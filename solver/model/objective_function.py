from np_solver.core import BaseSolution, BaseProblemInstance

def build_objective(problem: BaseProblemInstance, sol: BaseSolution) -> float:
    """
    Args:
        problem: Object with the problem variables calculated through the instance (.d, .K, .c_p, .F_p, .C, .M, etc).
        sol (BaseSolution[List[int]]): The solution (list of routes) to be evaluated.

    Returns:
        float: The total calculated cost for the objective function Z.
    """
    total_variable_cost = 0.0
    total_fixed_cost = 0.0

    all_visited_customers = set()

    num_customers_to_visit = len(problem.C) 

    for k, route in enumerate(sol):
        if not route or len(route) <= 2:
            continue

        vehicle_tuple = problem.K[k] 
        vehicle_type = vehicle_tuple[0] 
        
        fix_cost = problem.F_p[vehicle_type]
        var_cost_rate = problem.c_p[vehicle_type]
        
        total_fixed_cost += fix_cost
                
        for arc_index in range(len(route) - 1):
            i = route[arc_index]
            j = route[arc_index + 1]
            
            if i == j: 
                continue
            
            distance_ij = problem.d[i][j]
            
            total_variable_cost += var_cost_rate * distance_ij
                  
            if j in problem.C: 
                all_visited_customers.add(j)

    num_actually_visited = len(all_visited_customers)
    num_unvisited = num_customers_to_visit - num_actually_visited
  
    penalty_cost = problem.M * num_unvisited if num_unvisited > 0 else 0.0
    
    total_cost = total_variable_cost + total_fixed_cost + penalty_cost
    
    sol.cost = total_cost
    
    return total_cost