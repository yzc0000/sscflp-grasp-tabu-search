import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import pulp


@dataclass
class SSCFLPInstance:
    m: int
    n: int
    capacities: List[float]
    demands: List[float]
    fixed_costs: List[float]
    cost: List[List[float]]


@dataclass
class Solution:
    chrom: List[int]
    fitness: float = float("inf")
    base_cost: float = float("inf")
    violation: float = float("inf")


@dataclass
class FacilityIndividual:
    y: List[int]
    solution: Solution


def load_instance_from_excel(excel_path, sheet_name):
    """
    Load SSCFLP instance from Excel file.
    
    Excel structure:
    - Row 0: Headers (Cap, Cost, Dem, Fix)
    - Row 1: Facility 1 capacity/demand/fixed + COST COLUMN HEADERS (1,2,3...)
    - Row 2: Facility 2 capacity/demand/fixed + Facility 1's costs (cost_index=1)
    - Row 3: Facility 3 capacity/demand/fixed + Facility 2's costs (cost_index=2)
    - ...
    - Row 51: Facility 51 capacity/demand/fixed + Facility 50's costs (cost_index=50)
    - Row 52: (no cap/dem/fix) + Facility 51's costs (cost_index=51)
    
    So there's a 1-row offset: facility i's costs are in row i+1 (relative to cap/dem/fix in row i)
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    
    # Find demand and fixed cost column indices from row 0 headers
    # Search from column 50 onwards to avoid false matches in early columns
    # (Instance 3 has 'Fix' as column 0 label which is not the fixed cost data)
    demand_col = None
    fixed_col = None
    for col in df.columns:
        if col < 50:  # Skip early columns that might have misleading headers
            continue
        val = df.iloc[0, col]
        if isinstance(val, str):
            if demand_col is None and 'dem' in val.lower():
                demand_col = col + 1  # The actual values are in the next column
            elif fixed_col is None and 'fix' in val.lower():
                fixed_col = col + 1
        if demand_col is not None and fixed_col is not None:
            break
    
    if demand_col is None or fixed_col is None:
        raise ValueError(f"Could not find 'Dem' or 'Fix' headers in sheet '{sheet_name}'")
    

    # Find cost rows (column 4 non-null) - these define the number of facilities
    # These are rows 2-52 (row 1 has column headers, not cost data)
    cost_mask = df[4].notna()
    cost_rows = df.loc[cost_mask].copy()
    cost_rows.sort_values(by=4, inplace=True)  # Sort by cost index in column 4
    
    n_cost = len(cost_rows)
    n = n_cost  # Number of facilities/customers
    m = n
    
    # Find capacity rows (column 1 non-null, excluding row 0 header)
    # Need to match the number of cost rows
    capacity_mask = df[1].notna() & (df.index > 0)
    all_capacity_rows = df.loc[capacity_mask]
    
    # Take only the first n capacity rows (in case there's an extra row)
    if len(all_capacity_rows) > n:
        # Use the first n rows (by index)
        capacity_rows = all_capacity_rows.head(n)
    else:
        capacity_rows = all_capacity_rows
    
    if len(capacity_rows) != n:
        raise ValueError(
            f"Cannot match capacity rows to cost rows in sheet '{sheet_name}': "
            f"{len(capacity_rows)} capacity rows vs {n} cost rows"
        )
    
    # Load capacities from capacity rows (rows 1-51)
    capacities = capacity_rows[1].astype(float).to_list()
    
    # Load demands from capacity rows
    demands = capacity_rows[demand_col].astype(float).to_list()
    
    # Load fixed costs from capacity rows  
    fixed_costs = capacity_rows[fixed_col].astype(float).to_list()
    
    # Load cost matrix from cost rows (rows 2-52)
    # The cost matrix for facility i is in cost_rows[i] (after sorting by cost index)
    first_cost_col = 5
    last_cost_col = first_cost_col + n - 1
    cost_block = cost_rows.loc[:, first_cost_col:last_cost_col].astype(float)
    
    if cost_block.shape != (n, n):
        raise ValueError(
            f"Cost block on sheet '{sheet_name}' has shape {cost_block.shape}, expected ({n}, {n})."
        )
    
    cost = [row.tolist() for _, row in cost_block.iterrows()]
    
    print(f"Loaded: m={m}, n={n}, cost_cols={first_cost_col}-{last_cost_col}, demand_col={demand_col}, fixed_col={fixed_col}")
    
    return SSCFLPInstance(
        m=m,
        n=n,
        capacities=capacities,
        demands=demands,
        fixed_costs=fixed_costs,
        cost=cost,
    )


def evaluate_solution(sol, inst, penalty_lambda):
    m, n = inst.m, inst.n
    capacities = inst.capacities
    demands = inst.demands
    fixed = inst.fixed_costs
    cost = inst.cost

    load = [0.0] * m
    used = [0] * m
    assignment_cost = 0.0

    for j in range(n):
        i = sol.chrom[j]
        d_j = demands[j]
        c_ij = cost[i][j]
        load[i] += d_j
        used[i] = 1
        assignment_cost += c_ij  # Cost matrix already includes demand, don't multiply again

    fixed_cost = sum(fixed[i] for i in range(m) if used[i])
    base_cost = assignment_cost + fixed_cost

    total_violation = 0.0
    for i in range(m):
        viol = max(0.0, load[i] - capacities[i])
        total_violation += viol

    sol.base_cost = base_cost
    sol.violation = total_violation
    sol.fitness = base_cost + penalty_lambda * total_violation


def repair_solution(sol, inst, max_moves=1000):
    m, n = inst.m, inst.n
    capacities = inst.capacities
    demands = inst.demands
    fixed = inst.fixed_costs
    cost = inst.cost

    load = [0.0] * m
    used = [0] * m
    for j in range(n):
        i = sol.chrom[j]
        d_j = demands[j]
        load[i] += d_j
        used[i] = 1

    moves = 0
    improved = True
    while moves < max_moves and improved:
        improved = False

        overloaded = [i for i in range(m) if load[i] > capacities[i] + 1e-9]
        if not overloaded:
            break

        for i in overloaded:
            customers_i = [j for j in range(n) if sol.chrom[j] == i]
            customers_i.sort(key=lambda j: demands[j], reverse=True)

            moved_here = False
            for j in customers_i:
                d_j = demands[j]

                best_delta = None
                best_fac = None
                for k in range(m):
                    if k == i:
                        continue
                    if load[k] + d_j <= capacities[k] + 1e-9:
                        delta_assign = (cost[k][j] - cost[i][j])
                        delta_fixed = 0.0
                        if not used[k]:
                            delta_fixed += fixed[k]
                        if load[i] - d_j <= 1e-9:
                            delta_fixed -= fixed[i]
                        delta = delta_assign + delta_fixed

                        if best_delta is None or delta < best_delta:
                            best_delta = delta
                            best_fac = k

                if best_fac is not None:
                    sol.chrom[j] = best_fac
                    load[i] -= d_j
                    load[best_fac] += d_j
                    if load[i] <= 1e-9:
                        used[i] = 0
                    used[best_fac] = 1
                    moves += 1
                    improved = True
                    moved_here = True
                    break

            if moved_here:
                break


def tabu_search(
    inst,
    initial_sol,
    penalty_lambda,
    max_iter=30000,
    min_iter=10000,
    tabu_tenure=8,
    candidate_customers=None,
    no_improve_limit=7500,
    verbose=True,
    facility_move_freq=1,
):
    m, n = inst.m, inst.n
    capacities = inst.capacities
    demands = inst.demands
    fixed = inst.fixed_costs
    cost = inst.cost

    current = Solution(chrom=initial_sol.chrom.copy())
    evaluate_solution(current, inst, penalty_lambda)
    if current.violation > 1e-9:
        raise ValueError("Tabu search requires a feasible starting solution.")

    load = [0.0] * m
    used = [0] * m
    for j in range(n):
        i = current.chrom[j]
        d_j = demands[j]
        load[i] += d_j
        used[i] = 1

    best = Solution(
        chrom=current.chrom.copy(),
        fitness=current.fitness,
        base_cost=current.base_cost,
        violation=current.violation,
    )
    best_cost = best.base_cost

    # History tracking for visualization
    cost_history = [best_cost]
    
    tabu_customer = {}
    tabu_facility = {}
    move_frequency = {}  # (customer, facility) -> count
    current_tenure = tabu_tenure

    if candidate_customers is None or candidate_customers > n:
        candidate_customers = n

    no_improve = 0

    if verbose:
        print(f"Two-Level Tabu Search: start cost = {current.base_cost:.4f}")

    for it in range(1, max_iter + 1):
        # Periodic Re-Sync (Every 500 iters) - negligible perf impact, stops drift
        if it % 500 == 0:
            evaluate_solution(current, inst, penalty_lambda)
            load = [0.0] * m
            used = [0] * m
            for j in range(n):
                i = current.chrom[j]
                d_j = demands[j]
                load[i] += d_j
                used[i] = 1

        current_cost = current.base_cost
        best_move = None
        best_move_cost = float("inf")
        move_type = None

        customers = list(range(n))
        random.shuffle(customers)
        if candidate_customers < n:
            customers = customers[:candidate_customers]

        # REASSIGNMENT NEIGHBORHOOD: Move customer j from facility i0 to i1
        for j in customers:
            i0 = current.chrom[j]
            d_j = demands[j]

            for i1 in range(m):
                if i1 == i0:
                    continue
                if not used[i1]:
                    continue
                if load[i1] + d_j > capacities[i1] + 1e-9:
                    continue

                # Removed d_j multiplication to match problem definition (cost already includes demand)
                delta_assign = cost[i1][j] - cost[i0][j]
                delta_fixed = 0.0
                if load[i0] - d_j <= 1e-9:
                    delta_fixed -= fixed[i0]

                freq_penalty = 0.0
                if no_improve > 30:
                    freq = move_frequency.get((j, i1), 0)
                    freq_penalty = freq * (current_cost / 500.0)

                new_cost = current_cost + delta_assign + delta_fixed + freq_penalty
                real_cost = current_cost + delta_assign + delta_fixed

                attr = (j, i1)
                is_tabu = (attr in tabu_customer) and (tabu_customer[attr] >= it)
                if is_tabu and real_cost >= best_cost - 1e-6:
                    continue

                if new_cost < best_move_cost - 1e-9:
                    best_move_cost = new_cost
                    best_move = ('customer', j, i0, i1, real_cost)
                    move_type = 'customer'

        if it % facility_move_freq == 0:
            closed_facilities = [i for i in range(m) if not used[i]]
            random.shuffle(closed_facilities)
            
            for i_new in closed_facilities[:10]:
                potential_moves = []
                remaining_cap = capacities[i_new]
                
                for j in range(n):
                    i_old = current.chrom[j]
                    d_j = demands[j]
                    if d_j <= remaining_cap:
                        benefit = (cost[i_old][j] - cost[i_new][j])
                        if benefit > 0:
                            potential_moves.append((benefit, j, i_old, d_j))
                
                if not potential_moves:
                    continue
                
                potential_moves.sort(reverse=True)
                selected_customers = []
                total_demand = 0
                total_benefit = 0
                temp_load = load.copy()
                
                for benefit, j, i_old, d_j in potential_moves:
                    if total_demand + d_j <= capacities[i_new]:
                        selected_customers.append((j, i_old))
                        total_demand += d_j
                        total_benefit += benefit
                        temp_load[i_old] -= d_j
                
                if not selected_customers:
                    continue
                
                fixed_savings_from_donors = 0
                counted_donors = set()
                for j, i_old in selected_customers:
                    if i_old not in counted_donors and temp_load[i_old] <= 1e-9 and load[i_old] > 1e-9:
                        fixed_savings_from_donors += fixed[i_old]
                        counted_donors.add(i_old)
                
                delta = fixed[i_new] - total_benefit - fixed_savings_from_donors
                new_cost = current_cost + delta
                
                is_tabu = (i_new in tabu_facility) and (tabu_facility[i_new] >= it)
                if is_tabu and new_cost >= best_cost - 1e-6:
                    continue
                
                if new_cost < best_move_cost - 1e-9:
                    best_move_cost = new_cost
                    best_move = ('open_facility', i_new, selected_customers)
                    move_type = 'facility'
            
            open_facilities = [i for i in range(m) if used[i]]
            
            for i_close in open_facilities:
                customers_at_i = [j for j in range(n) if current.chrom[j] == i_close]
                if not customers_at_i:
                    continue
                
                reassignment = {}
                total_cost_increase = 0
                feasible = True
                
                temp_load = load.copy()
                temp_load[i_close] = 0
                
                for j in customers_at_i:
                    d_j = demands[j]
                    best_alt = None
                    best_alt_cost = float("inf")
                    
                    for k in range(m):
                        if k == i_close or not used[k]:
                            continue
                        if temp_load[k] + d_j <= capacities[k] + 1e-9:
                            alt_cost = cost[k][j]
                            if alt_cost < best_alt_cost:
                                best_alt_cost = alt_cost
                                best_alt = k
                    
                    if best_alt is None:
                        feasible = False
                        break
                    
                    reassignment[j] = best_alt
                    total_cost_increase += (cost[best_alt][j] - cost[i_close][j])
                    temp_load[best_alt] += d_j
                
                if not feasible:
                    continue
                
                delta = total_cost_increase - fixed[i_close]
                new_cost = current_cost + delta
                
                is_tabu = (i_close in tabu_facility) and (tabu_facility[i_close] >= it)
                if is_tabu and new_cost >= best_cost - 1e-6:
                    continue
                
                if new_cost < best_move_cost - 1e-9:
                    best_move_cost = new_cost
                    best_move = ('close_facility', i_close, reassignment)
                    move_type = 'facility'

        if best_move is None:
            if verbose:
                print(f"Tabu Search: no admissible moves at iteration {it}, stopping.")
            break

        if best_move[0] == 'customer':
            _, j, i0, i1, real_cost = best_move
            d_j = demands[j]

            current.chrom[j] = i1
            load[i0] -= d_j
            load[i1] += d_j
            if load[i0] <= 1e-9:
                used[i0] = 0
            used[i1] = 1

            current.base_cost = real_cost
            current.violation = 0.0
            current.fitness = real_cost

            tabu_customer[(j, i0)] = it + current_tenure
            move_frequency[(j, i1)] = move_frequency.get((j, i1), 0) + 1
            
        elif best_move[0] == 'open_facility':
            _, i_new, selected_customers = best_move
            
            used[i_new] = 1
            for j, i_old in selected_customers:
                d_j = demands[j]
                current.chrom[j] = i_new
                load[i_old] -= d_j
                load[i_new] += d_j
                if load[i_old] <= 1e-9:
                    used[i_old] = 0

            current.base_cost = best_move_cost
            current.violation = 0.0
            current.fitness = best_move_cost
            
            tabu_facility[i_new] = it + current_tenure * 2
            
        elif best_move[0] == 'close_facility':
            _, i_close, reassignment = best_move
            
            for j, k in reassignment.items():
                d_j = demands[j]
                current.chrom[j] = k
                load[i_close] -= d_j
                load[k] += d_j
            
            used[i_close] = 0
            load[i_close] = 0.0

            current.base_cost = best_move_cost
            current.violation = 0.0
            current.fitness = best_move_cost
            
            tabu_facility[i_close] = it + current_tenure * 2

        to_delete = [key for key, exp in tabu_customer.items() if exp < it]
        for key in to_delete:
            del tabu_customer[key]
        to_delete = [key for key, exp in tabu_facility.items() if exp < it]
        for key in to_delete:
            del tabu_facility[key]

        if current.base_cost < best_cost - 1e-6:
            # "Trust but Verify": Re-calc only on alleged new best to confirm it's real
            evaluate_solution(current, inst, penalty_lambda)
            load = [0.0] * m
            used = [0] * m
            for j in range(n):
                i = current.chrom[j]
                d_j = demands[j]
                load[i] += d_j
                used[i] = 1
            
            # Check if it's still best after correction
            if current.base_cost < best_cost - 1e-6:
                best_cost = current.base_cost
                best = Solution(
                    chrom=current.chrom.copy(),
                    fitness=current.base_cost,
                    base_cost=current.base_cost,
                    violation=0.0,
                )
                no_improve = 0
                if verbose and move_type == 'facility':
                    print(f"Tabu it {it}: NEW BEST via {best_move[0]} = {best_cost:.4f}")
            else:
                # It was a drift phantom, ignore and continue
                no_improve += 1
        else:
            no_improve += 1
            
        # Append best cost of this iteration
        cost_history.append(best.base_cost)

        if verbose and it % 100 == 0:
            num_open = sum(used)
            print(
                f"Tabu it {it}: current={current.base_cost:.4f}, "
                f"best={best_cost:.4f}, open_facilities={num_open}"
            )



        if no_improve >= no_improve_limit and it >= min_iter:
            if verbose:
                print(
                    f"Tabu Search: early stop at iteration {it} "
                    f"after {no_improve} iterations without improvement."
                )
            break

    if verbose:
        print(f"Two-Level Tabu Search: final best cost = {best.base_cost:.4f}")

    evaluate_solution(best, inst, penalty_lambda)
    return best, cost_history


def ensure_capacity(y, inst, min_factor=1.0):
    m = inst.m
    total_demand = sum(inst.demands)
    cap = sum(inst.capacities[i] for i in range(m) if y[i] == 1)

    if cap >= min_factor * total_demand and any(y):
        return

    indices_closed = [i for i in range(m) if y[i] == 0]

    if not indices_closed and cap == 0:
        best = min(
            range(m),
            key=lambda i: inst.fixed_costs[i] / max(inst.capacities[i], 1e-6)
        )
        y[best] = 1
        return

    indices_closed.sort(
        key=lambda i: inst.fixed_costs[i] / max(inst.capacities[i], 1e-6)
    )

    for i in indices_closed:
        if cap >= min_factor * total_demand:
            break
        y[i] = 1
        cap += inst.capacities[i]


def decode_y_to_solution(y, inst, penalty_lambda):
    m, n = inst.m, inst.n
    y = y.copy()

    ensure_capacity(y, inst, min_factor=1.0)

    capacities_left = [
        inst.capacities[i] if y[i] == 1 else 0.0 for i in range(m)
    ]
    demands = inst.demands
    cost = inst.cost

    chrom = [-1] * n

    customers = list(range(n))
    customers.sort(key=lambda j: demands[j], reverse=True)

    for j in customers:
        d_j = demands[j]

        candidates = []
        for i in range(m):
            if y[i] == 1 and capacities_left[i] >= d_j:
                candidates.append((cost[i][j], i))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, i_sel = candidates[0]
        else:
            open_fac = [i for i in range(m) if y[i] == 1]
            if not open_fac:
                i_sel = min(
                    range(m),
                    key=lambda i: inst.fixed_costs[i] / max(inst.capacities[i], 1e-6)
                )
                y[i_sel] = 1
                capacities_left[i_sel] = inst.capacities[i_sel]
                open_fac = [i_sel]
            i_sel = min(open_fac, key=lambda i: cost[i][j])

        chrom[j] = i_sel
        capacities_left[i_sel] -= d_j

    sol = Solution(chrom=chrom)
    repair_solution(sol, inst, max_moves=500)
    evaluate_solution(sol, inst, penalty_lambda)

    return sol



def compute_facility_attractiveness(inst):
    m, n = inst.m, inst.n
    scores = []
    
    for i in range(m):
        fixed_per_cap = inst.fixed_costs[i] / max(inst.capacities[i], 1e-6)
        total_demand = sum(inst.demands)
        avg_trans_cost = sum(
            inst.demands[j] * inst.cost[i][j] for j in range(n)
        ) / max(total_demand, 1e-6)
        scores.append(fixed_per_cap + avg_trans_cost)
    
    return scores



def grasp_construct_solution(inst, alpha, penalty_lambda):
    m = inst.m
    total_demand = sum(inst.demands)
    scores = compute_facility_attractiveness(inst)
    
    y = [0] * m
    current_capacity = 0.0
    available = list(range(m))
    
    while current_capacity < total_demand and available:
        min_score = min(scores[i] for i in available)
        max_score = max(scores[i] for i in available)
        threshold = min_score + alpha * (max_score - min_score)
        
        rcl = [i for i in available if scores[i] <= threshold]
        
        if not rcl:
            rcl = [min(available, key=lambda i: scores[i])]
        
        selected = random.choice(rcl)
        y[selected] = 1
        current_capacity += inst.capacities[selected]
        available.remove(selected)
    
    ensure_capacity(y, inst, min_factor=1.0)
    sol = decode_y_to_solution(y, inst, penalty_lambda)
    
    return FacilityIndividual(y=y, solution=sol)



def run_grasp(
    inst,
    max_iterations=200,
    penalty_lambda=10000.0,
    rng_seed=None,
    log_interval=10,
    verbose=True,
    alpha_min=0.1,
    alpha_max=0.4,
):
    if rng_seed is not None:
        random.seed(rng_seed)
    
    all_feasible_solutions = []
    best_cost = float("inf")
    
    if verbose:
        print(f"GRASP Construction: Generating {max_iterations} starting solutions")
    
    for iteration in range(1, max_iterations + 1):
        current_alpha = random.uniform(alpha_min, alpha_max)
        ind = grasp_construct_solution(inst, current_alpha, penalty_lambda)
        sol = ind.solution
        
        if sol.violation < 1e-9:
            all_feasible_solutions.append(Solution(
                chrom=sol.chrom.copy(),
                fitness=sol.fitness,
                base_cost=sol.base_cost,
                violation=sol.violation,
            ))
            if sol.base_cost < best_cost:
                best_cost = sol.base_cost
        
        if verbose and log_interval > 0 and iteration % log_interval == 0:
            print(
                f"GRASP iteration {iteration}: "
                f"constructed_cost={sol.base_cost:.4f}, "
                f"feasible_count={len(all_feasible_solutions)}, "
                f"best_so_far={best_cost:.4f}"
            )
    
    if verbose:
        print(f"GRASP Construction: Completed. Generated {len(all_feasible_solutions)} feasible solutions.")
        print(f"       Best constructed cost = {best_cost:.4f}")
    
    return all_feasible_solutions



def compute_lp_relaxation_lb(inst):
    m, n = inst.m, inst.n
    capacities = inst.capacities
    demands = inst.demands
    fixed = inst.fixed_costs
    cost = inst.cost
    prob = pulp.LpProblem("SSCFLP_LP_RELAX", pulp.LpMinimize)

    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0.0, upBound=1.0)
        for i in range(m) for j in range(n)
    }
    y = {
        i: pulp.LpVariable(f"y_{i}", lowBound=0.0, upBound=1.0)
        for i in range(m)
    }

    prob += (
        pulp.lpSum(
            cost[i][j] * x[(i, j)]  # No demand multiplication here
            for i in range(m) for j in range(n)
        )
        + pulp.lpSum(
            fixed[i] * y[i] for i in range(m)
        )
    )

    for i in range(m):
        prob += (
            pulp.lpSum(demands[j] * x[(i, j)] for j in range(n))
            <= capacities[i] * y[i]
        )

    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(m)) == 1.0

    for i in range(m):
        for j in range(n):
            prob += x[(i, j)] <= y[i]

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        print("Warning: LP relaxation did not find an optimal solution.")
        return None

    return pulp.value(prob.objective)


if __name__ == "__main__":
    excel_file = Path(__file__).resolve().parents[1] / "data" / "Modeling data.xlsx"
    sheet_names = ["Problem Instance1", "Problem Instance2", "Problem Instance3"]

    max_iterations = 5
    penalty_lambda = 10000.0
    log_interval = 25
    alpha_min = 0.1
    alpha_max = 0.4

    seeds = [103]
    
    summary_results = []  # To store (Instance, Best GRASP, Best Tabu, LB, Gap)

    for sheet in sheet_names:
        print("=" * 70)
        print(f"Sheet / instance: {sheet}")

        inst = load_instance_from_excel(excel_file, sheet)

        lb = compute_lp_relaxation_lb(inst)
        if lb is not None:
            print(f"LP relaxation lower bound: {lb:.4f}")
        else:
            print("LP relaxation lower bound: not available.")

        best_global = None

        for run_idx, seed in enumerate(seeds, start=1):
            print(f"\n--- GRASP Construction with seed {seed} ---")

            all_solutions = run_grasp(
                inst,
                max_iterations=max_iterations,
                penalty_lambda=penalty_lambda,
                rng_seed=seed,
                log_interval=log_interval,
                verbose=True,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )

            if not all_solutions:
                print("No feasible solutions found by GRASP.")
                continue

            # Find best constructed solution cost (Best GRASP)
            best_grasp_cost = min(s.base_cost for s in all_solutions)

            print(f"\n--- Running Tabu Search on {len(all_solutions)} starting solutions ---")
            print(f"{'Sol#':<6} {'Start Cost':<12} {'Final Cost':<12} {'Improvement':<12}")
            print("-" * 44)
            
            for sol_idx, start_sol in enumerate(all_solutions, start=1):
                start_cost = start_sol.base_cost
                
                best_after_ts, _ = tabu_search(
                    inst,
                    start_sol,
                    penalty_lambda=penalty_lambda,
                    verbose=False,  
                )
                
                final_cost = best_after_ts.base_cost
                improvement = start_cost - final_cost
                print(f"{sol_idx:<6} {start_cost:<12.2f} {final_cost:<12.2f} {improvement:<12.2f}")

                if best_global is None or best_after_ts.base_cost < best_global.base_cost:
                    best_global = Solution(
                        chrom=best_after_ts.chrom.copy(),
                        fitness=best_after_ts.fitness,
                        base_cost=best_after_ts.base_cost,
                        violation=best_after_ts.violation,
                    )
                    print(f"  ^^^ NEW GLOBAL BEST: {best_global.base_cost:.4f}")

        print("\n=== Summary over all runs (GRASP + Tabu Search) ===")
        if best_global is not None:
            print(f"Best cost over all runs (after TS): {best_global.base_cost:.4f}")
            print(f"Total capacity violation: {best_global.violation:.4f}")
            if lb is not None and lb > 0:
                gap = (best_global.base_cost - lb) / lb * 100.0
                print(f"Relative gap (GRASP+TS vs LP LB): {gap:.2f}%")

            chrom = best_global.chrom
            m, n = inst.m, inst.n

            open_facilities = sorted(set(chrom))

            print("\nOpened facilities (0-based indices):")
            print(", ".join(str(i) for i in open_facilities))

            print("Opened facilities (1-based indices):")
            print(", ".join(str(i + 1) for i in open_facilities))

            print("\nCustomer assignments (customer -> facility, 0-based indices):")
            for j in range(n):
                print(f"  Customer {j} -> Facility {chrom[j]}")

        else:
            print("No feasible solution found in any run.")
            
        # Store results for summary
        final_best = best_global.base_cost if best_global else float('inf')
        gap = None
        if lb is not None and lb > 0 and best_global:
            gap = (best_global.base_cost - lb) / lb * 100.0
            
        summary_results.append({
            "Instance": sheet,
            "Best Tabu": final_best,
            "LB": lb if lb else float('nan'),
            "Gap": gap if gap is not None else float('nan')
        })

    print("=" * 80)
    print(f"{'Instance':<25} {'Best Tabu':<15} {'LB':<15} {'Gap (%)':<10}")
    print("-" * 80)
    for res in summary_results:
        gap_str = f"{res['Gap']:.2f}%" if not pd.isna(res['Gap']) else "N/A"
        lb_str = f"{res['LB']:.2f}" if not pd.isna(res['LB']) else "N/A"
        print(f"{res['Instance']:<25} {res['Best Tabu']:<15.2f} {lb_str:<15} {gap_str:<10}")
    print("=" * 80)
