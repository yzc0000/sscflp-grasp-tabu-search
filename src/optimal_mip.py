"""
Optimal MIP solver for SSCFLP (Single Source Capacitated Facility Location Problem)
Uses PuLP with CBC solver to find the exact optimal solution.
"""

from pathlib import Path

import pulp

try:
    from .sscflp_solver import load_instance_from_excel
except ImportError:
    from sscflp_solver import load_instance_from_excel


def solve_sscflp_optimal(inst, time_limit=300, verbose=True):
    """
    Solve SSCFLP to optimality using Mixed Integer Programming.
    
    Args:
        inst: SSCFLPInstance with m facilities and n customers
        time_limit: Maximum solve time in seconds
        verbose: Print progress information
    
    Returns:
        Dictionary with optimal solution details
    """
    m, n = inst.m, inst.n
    capacities = inst.capacities
    demands = inst.demands
    fixed = inst.fixed_costs
    cost = inst.cost
    
    if verbose:
        print(f"Setting up MIP for SSCFLP with {m} facilities and {n} customers...")
    
    # Create the problem
    prob = pulp.LpProblem("SSCFLP_Optimal", pulp.LpMinimize)
    
    # Decision variables
    # x[i][j] = 1 if customer j is assigned to facility i (BINARY for single source)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
        for i in range(m) for j in range(n)
    }
    
    # y[i] = 1 if facility i is open
    y = {
        i: pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary)
        for i in range(m)
    }
    
    # Objective: minimize total cost (assignment + fixed)
    prob += (
        pulp.lpSum(
            cost[i][j] * x[(i, j)]  # Cost already includes demand
            for i in range(m) for j in range(n)
        )
        + pulp.lpSum(
            fixed[i] * y[i] for i in range(m)
        ),
        "Total_Cost"
    )
    
    # Constraint 1: Each customer must be assigned to exactly one facility
    for j in range(n):
        prob += (
            pulp.lpSum(x[(i, j)] for i in range(m)) == 1,
            f"Single_Assignment_{j}"
        )
    
    # Constraint 2: Capacity constraints for each facility
    for i in range(m):
        prob += (
            pulp.lpSum(demands[j] * x[(i, j)] for j in range(n)) <= capacities[i] * y[i],
            f"Capacity_{i}"
        )
    
    # Constraint 3: Can only assign to open facilities
    for i in range(m):
        for j in range(n):
            prob += (
                x[(i, j)] <= y[i],
                f"Link_{i}_{j}"
            )
    
    if verbose:
        print(f"MIP has {len(x) + len(y)} variables and {m*n + m + n} constraints")
        print(f"Solving with CBC (time limit = {time_limit}s)...")
    
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    
    if status not in ["Optimal", "Not Solved"]:
        if verbose:
            print(f"Solver status: {status}")
        if status == "Infeasible":
            return {"status": "Infeasible", "optimal_cost": None}
    
    # Extract solution
    optimal_cost = pulp.value(prob.objective)
    
    # Extract which facilities are open
    open_facilities = [i for i in range(m) if pulp.value(y[i]) > 0.5]
    
    # Extract customer assignments
    assignments = {}
    for j in range(n):
        for i in range(m):
            if pulp.value(x[(i, j)]) > 0.5:
                assignments[j] = i
                break
    
    # Calculate cost breakdown
    # Cost matrix already includes demand weighting
    assignment_cost = sum(
        cost[assignments[j]][j] for j in range(n)
    )
    fixed_cost = sum(fixed[i] for i in open_facilities)
    
    # Calculate facility loads
    loads = {i: 0.0 for i in open_facilities}
    for j, i in assignments.items():
        loads[i] = loads.get(i, 0) + demands[j]
    
    result = {
        "status": status,
        "optimal_cost": optimal_cost,
        "assignment_cost": assignment_cost,
        "fixed_cost": fixed_cost,
        "open_facilities": open_facilities,
        "num_open": len(open_facilities),
        "assignments": assignments,
        "loads": loads,
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("OPTIMAL SOLUTION FOUND")
        print("=" * 60)
        print(f"Status: {status}")
        print(f"Optimal Cost: {optimal_cost:.4f}")
        print(f"  - Assignment Cost: {assignment_cost:.4f}")
        print(f"  - Fixed Cost: {fixed_cost:.4f}")
        print(f"\nOpen Facilities ({len(open_facilities)}): {open_facilities}")
        print("\nFacility Details:")
        print(f"{'Facility':<10} {'Capacity':<12} {'Load':<12} {'Utilization':<12} {'Fixed Cost':<12}")
        print("-" * 58)
        for i in sorted(open_facilities):
            util = (loads[i] / capacities[i]) * 100 if capacities[i] > 0 else 0
            print(f"{i:<10} {capacities[i]:<12.2f} {loads[i]:<12.2f} {util:<11.1f}% {fixed[i]:<12.2f}")
        
        print("\nCustomer Assignments:")
        print(f"{'Customer':<10} {'Facility':<10} {'Demand':<12} {'Unit Cost':<12} {'Total Cost':<12}")
        print("-" * 56)
        for j in range(n):
            i = assignments[j]
            total = cost[i][j]  # Already includes demand
            print(f"{j:<10} {i:<10} {demands[j]:<12.2f} {cost[i][j]:<12.4f} {total:<12.4f}")
    
    return result


if __name__ == "__main__":
    excel_file = Path(__file__).resolve().parents[1] / "data" / "Modeling data.xlsx"
    sheet = "Problem Instance1"
    
    print("=" * 70)
    print(f"Loading instance: {sheet}")
    print("=" * 70)
    
    inst = load_instance_from_excel(excel_file, sheet)
    
    print(f"\nInstance size: {inst.m} facilities, {inst.n} customers")
    print(f"Total demand: {sum(inst.demands):.2f}")
    print(f"Total capacity: {sum(inst.capacities):.2f}")
    
    result = solve_sscflp_optimal(inst, time_limit=300, verbose=True)
    
    if result["status"] == "Optimal":
        print("\n" + "=" * 70)
        print("VERIFICATION")
        print("=" * 70)
        
        # Verify feasibility
        all_assigned = len(result["assignments"]) == inst.n
        capacity_ok = all(
            result["loads"].get(i, 0) <= inst.capacities[i] + 1e-6 
            for i in result["open_facilities"]
        )
        
        print(f"All customers assigned: {all_assigned}")
        print(f"All capacity constraints satisfied: {capacity_ok}")
        print(f"\nOptimal objective value: {result['optimal_cost']:.4f}")
