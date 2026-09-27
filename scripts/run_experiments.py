"""Run GRASP + Tabu Search experiments and save convergence charts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sscflp_solver import (
    compute_lp_relaxation_lb,
    load_instance_from_excel,
    run_grasp,
    tabu_search,
)


def main() -> None:
    excel_file = PROJECT_ROOT / "data" / "Modeling data.xlsx"
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    sheet_names = ["Problem Instance1", "Problem Instance2", "Problem Instance3"]
    max_grasp_iterations = 5
    penalty_lambda = 10_000.0
    seed = 103

    print("Running GRASP-initialized Tabu Search experiments...")

    for sheet in sheet_names:
        print(f"\nProcessing {sheet}")
        instance = load_instance_from_excel(excel_file, sheet)
        lower_bound = compute_lp_relaxation_lb(instance)
        solutions = run_grasp(
            instance,
            max_iterations=max_grasp_iterations,
            penalty_lambda=penalty_lambda,
            rng_seed=seed,
            verbose=False,
        )

        if not solutions:
            print("No feasible GRASP solution was produced.")
            continue

        plt.figure(figsize=(10, 6))
        best_cost = float("inf")
        for index, solution in enumerate(solutions, start=1):
            improved_solution, history = tabu_search(
                instance,
                solution,
                penalty_lambda=penalty_lambda,
                verbose=False,
            )
            best_cost = min(best_cost, improved_solution.base_cost)
            plt.plot(history, label=f"GRASP start {index}")

        if lower_bound is not None:
            plt.axhline(lower_bound, color="crimson", linestyle="--", label=f"LP lower bound ({lower_bound:.2f})")

        plt.title(f"Tabu Search Convergence: {sheet}")
        plt.xlabel("Iteration")
        plt.ylabel("Objective value")
        plt.grid(alpha=0.3)
        plt.legend()
        chart_path = output_dir / f"{sheet.replace(' ', '_')}_convergence.png"
        plt.savefig(chart_path, dpi=160, bbox_inches="tight")
        plt.close()

        print(f"Best Tabu Search cost: {best_cost:.4f}")
        print(f"Convergence chart: {chart_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
