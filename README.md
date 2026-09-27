# GRASP-Initialized Tabu Search for SSCFLP

This project solves the **Single-Source Capacitated Facility Location Problem (SSCFLP)** with a GRASP-initialized, two-level Tabu Search heuristic. It also includes LP-relaxation lower bounds and an exact mixed-integer programming model for comparison on solvable instances.

## Problem

SSCFLP selects which facilities to open and assigns every customer to exactly one open facility. Each facility has a fixed opening cost and a capacity limit; every customer has a demand and an assignment cost for each facility.

The objective minimizes total fixed and assignment cost while enforcing:

- exactly one facility assignment per customer;
- facility capacity limits;
- assignments only to opened facilities.

## Algorithm

### GRASP initialization

The construction phase scores facilities using fixed cost per unit of capacity and average assignment cost. A restricted candidate list is formed with a randomized `alpha` threshold, facilities are opened until capacity covers total demand, and customers are assigned to affordable open facilities. A repair procedure moves customers when needed to restore capacity feasibility.

### Two-level Tabu Search

The intensification phase starts from each feasible GRASP solution and explores two neighborhoods:

- **Customer reassignment:** move one customer between currently open facilities when capacity allows.
- **Facility moves:** open a closed facility and move a profitable group of customers, or close an open facility and feasibly reassign its customers.

Tabu tenure prevents immediate reversals, an aspiration rule permits tabu moves that improve the global best solution, and a move-frequency penalty diversifies the search after prolonged stagnation.

## Repository layout

```text
src/sscflp_solver.py  Canonical GRASP + Tabu Search implementation
src/optimal_mip.py    Exact binary MIP formulation using PuLP/CBC
scripts/              Reproducible experiment runner
data/                 Excel workbook with three SSCFLP instances
results/              Generated convergence charts (created locally, ignored by Git)
```

## Installation

```bash
git clone https://github.com/yzc0000/sscflp-grasp-tabu-search.git
cd sscflp-grasp-tabu-search
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run experiments

Run the heuristic on all three Excel instances. The script computes an LP lower bound, applies GRASP followed by Tabu Search, and saves one convergence chart per instance under `results/`.

```bash
python scripts/run_experiments.py
```

## Example results

The canonical solver was run on all three sheets in `data/Modeling data.xlsx` with random seed `103`, five GRASP constructions per instance, and Tabu Search refinement from each feasible construction. Every reported solution had zero capacity violation.

| Instance | Facilities / customers | Best GRASP cost | Best Tabu Search cost | LP lower bound | Gap above LP bound | Open facilities |
|---|---:|---:|---:|---:|---:|---:|
| Problem Instance1 | 51 / 51 | 1,632.00 | **1,208.00** | 1,158.2204 | **4.30%** | 10 |
| Problem Instance2 | 55 / 55 | 8,655.00 | **7,678.00** | 7,220.1838 | **6.34%** | 16 |
| Problem Instance3 | 63 / 63 | 11,240.00 | **10,695.00** | 9,908.5600 | **7.94%** | 21 |

The Tabu Search improves the best GRASP construction by 424 cost units on Instance 1, 977 on Instance 2, and 545 on Instance 3. The LP lower bound is a relaxation rather than a feasible integer solution, so the reported gap measures how close the heuristic result is to that bound—not proof of optimality.

The experiment runner also writes a convergence chart for each instance, showing the best objective value as Tabu Search progresses from each GRASP starting solution.

## Exact comparator

Run the binary MIP model on the first instance with CBC:

```bash
python src/optimal_mip.py
```

The exact model is useful for validating heuristic solution quality on instances that can be solved within the configured time limit.

## Dependencies

Python, Pandas, PuLP/CBC, and Matplotlib.

## Author

Ahmed Yazıcı
