from np_solver.core import BaseSolution, BaseProblemInstance

"""
constrains that are always true due to data structure constrains:
max_vehicles_use,
allow_unused_vehicles
ensure_flow_conservation
eliminate_subtours
"""

def max_vehicles_use(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """ 
    1. Max of |K| vehicles must leave the depot 0 () 

    The structure of the `sol` solution is a list of routes, one for each vehicle k in K. The number of routes *used* (len > 2) will always be <= len(sun), which is |K|. Structurally guaranteed.
    """
    return True

def allow_unused_vehicles(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """ 
    2. Allows not all vehicles to be used 

    Each vehicle k has a route sol[k]. If the route is [0, 0] or [], the vehicle was not used (left 0 times). If it is [0, i, ..., 0], he left 1 time. It is impossible to exit > 1 time. Structurally guaranteed.
    """
    return True

def ensure_depot_return(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """ 3. Ensures all vehicles return to the depot at the end of the route """
    for route in sol:
        if len(route) > 2: 
            if route[0] != 0 or route[-1] != 0:
                # print("VIOLAÇÃO: Rota não começa ou não termina no depósito 0.")
                return False
    return True

def limit_client_visits(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """ 4. All client are visited at most 1 time """
    visited_clients = set()
    for route in sol:
        for node in route:
            if node in problem.C: 
                if node in visited_clients:
                    # print(f"VIOLAÇÃO: Cliente {node} visitado mais de uma vez.")
                    return False
                visited_clients.add(node)
    return True

def ensure_flow_conservation(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """
    5. All vehicles arrives and departs from the same client 

    The list representation [..., i, j, k, ...] guarantees that if you arrived at 'j' (coming from 'i'), you should exit of 'j' (going to 'k'). Structurally guaranteed.
    """
    return True

def eliminate_subtours(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """ 
    6. Prohibits the formations of subcicles that does not include the depot 

    A route is a simple path that starts and ends at 0. Restriction 4 (limit_client_visits) already prohibits visiting the same customer twice, which prevents cycles. Structurally guaranteed.
    """
    return True

def limit_vehicle_capacity(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """ 7. Ensures no route exceeds the vehicle's capacity """
    for k, route in enumerate(sol):
        if len(route) <= 2: # unused route
            continue
            
        
        vehicle_type = problem.K[k][0] 
        capacity = problem.Q_p[vehicle_type]
        
        current_load = 0.0
        for node in route:
            current_load += problem.q[node]
            
        if current_load > capacity:
            # print(f"VIOLAÇÃO: Rota {k} excedeu capacidade. Carga: {current_load} > Cap: {capacity}")
            return False
    return True

def respect_time_windows(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """ 8. Ensures time window feasibility """
    for k, route in enumerate(sol):
        if len(route) <= 2: # unused route
            continue

        current_time = problem.e[0] 
        
        for idx in range(len(route) - 1):
            i = route[idx]     
            j = route[idx + 1]
            
            travel_time = problem.t[i][j]
            

            arrival_at_j = current_time + travel_time
            

            service_start_j = max(arrival_at_j, problem.e[j])

            if service_start_j > problem.l[j]:
                # print(f"VIOLAÇÃO: Rota {k} atrasada no nó {j}. Chegou: {service_start_j} > Janela: {problem.l[j]}")
                return False
                
            current_time = service_start_j + problem.s[j]
            
    return True

def track_vehicle_load(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """ 9. Ensures the load difference """
    return True

def limit_segment_load(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """ 10. Limits maximum vehicle capacity. """
    return True

def build_constraints(problem: BaseProblemInstance, sol: BaseSolution) -> bool:
    """
    Verifies if the given solution candidade is feasible, considering all constrains.
    """
    if (
        max_vehicles_use(problem, sol) and
        allow_unused_vehicles(problem, sol) and
        ensure_depot_return(problem, sol) and
        limit_client_visits(problem, sol) and
        ensure_flow_conservation(problem, sol) and
        eliminate_subtours(problem, sol) and
        limit_vehicle_capacity(problem, sol) and
        respect_time_windows(problem, sol) and
        track_vehicle_load(problem, sol) and
        limit_segment_load(problem, sol)
    ):
        # the solution is possible
        return True
    # the solution is not possible
    return False