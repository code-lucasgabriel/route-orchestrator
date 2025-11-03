from np_solver.core import BaseEvaluator, BaseProblemInstance, BaseSolution
from np_solver.metaheuristics.alns import ALNS
from np_solver.metaheuristics.alns.components import ALNSAcceptance, ALNSDestroy, ALNSRepair, ALNSWeightManager, SimulatedAnnealingAcceptance, RouletteWheelManager
from np_solver.metaheuristics.alns.interface import ALNSDestroy, ALNSRepair
import random
from typing import List, Set, Tuple, Any


def _get_clients_in_solution(solution: BaseSolution) -> List[int]:
    """Helper to get all visited clients, excluding depots."""
    clients = []
    for route in solution:
        for node in route:
            if node != 0: # Assuming 0 is the depot
                clients.append(node)
    return clients

def _get_unvisited_clients(solution: BaseSolution, problem: BaseProblemInstance) -> List[int]:
    """Helper to get all clients in the problem that are not in the solution."""
    all_problem_clients = set(problem.C)
    clients_in_solution_set = set(_get_clients_in_solution(solution))
    return list(all_problem_clients - clients_in_solution_set)

def _get_nodes_in_used_routes(solution: BaseSolution) -> List[int]:
    """Helper to get all nodes (clients + depots) in *active* routes."""
    nodes = []
    for route in solution:
        if len(route) > 2: # A route with > 2 nodes is used (e.g., [0, A, 0])
            nodes.extend(route)
    return nodes

def _get_unused_vehicles(solution: BaseSolution) -> List[int]:
    """Helper to get indices of empty routes [0, 0]."""
    empty_route = [0, 0]
    unused_vehicles = []
    for idx, route in enumerate(solution):
        if route == empty_route:
            unused_vehicles.append(idx)
    return unused_vehicles


# --- Destroy Operators ---

class RandomDestroy(ALNSDestroy):
    """
    A concrete ALNSDestroy strategy.
    
    It removes a fixed number (`num_to_remove`) of randomly selected
    clients from the solution.
    """
    def __init__(self, num_to_remove: int):
        if num_to_remove < 1:
            raise ValueError("num_to_remove must be at least 1.")
        self.num_to_remove = num_to_remove
        
    def destroy(self, solution: BaseSolution, problem: BaseProblemInstance) -> BaseSolution:
        """
        Removes `num_to_remove` random clients.
        
        ! We must assume the BaseSolution has a `remove_client(client_id)`
        ! method that removes the client from its route.
        """
        new_sol = solution.copy()
        clients_in_solution = _get_clients_in_solution(new_sol)
        
        if not clients_in_solution:
            return new_sol # Nothing to remove
            
        # Ensure we don't try to remove more clients than exist
        n = min(self.num_to_remove, len(clients_in_solution))
        
        clients_to_remove = random.sample(clients_in_solution, n)
        
        for client in clients_to_remove:
            try:
                # This is the assumed method, similar to your `swap`, `relocate`, etc.
                new_sol.remove_client(client) 
            except Exception as e:
                print(f"Warning: Could not remove client {client}. Error: {e}")
                
        return new_sol

class RouteDestroy(ALNSDestroy):
    """
    A concrete ALNSDestroy strategy.
    
    It selects one *used* route at random and removes all clients
    from it, adding them to the unvisited pool.
    """
    def destroy(self, solution: BaseSolution, problem: BaseProblemInstance) -> BaseSolution:
        """Removes all clients from one randomly selected active route."""
        new_sol = solution.copy()
        
        used_route_indices = [
            i for i, route in enumerate(new_sol) if len(route) > 2
        ]
        
        if not used_route_indices:
            return new_sol # No used routes to destroy
            
        route_idx_to_destroy = random.choice(used_route_indices)
        
        # Get clients, but skip depots (0)
        clients_to_remove = [
            node for node in new_sol[route_idx_to_destroy] if node != 0
        ]
        
        for client in clients_to_remove:
            try:
                new_sol.remove_client(client)
            except Exception as e:
                print(f"Warning: Could not remove client {client}. Error: {e}")
        
        return new_sol

# --- Repair Operators ---

class GreedyRepair(ALNSRepair):
    """
    A concrete ALNSRepair strategy.
    
    It iterates through all unvisited clients (in a random order)
    and inserts each one into its *best possible position*.
    
    The "best position" is found by checking all 'insert' and 'insert_use'
    moves, identical to the logic in your TSNeighborhood.
    """

    def _find_best_insertion(self, 
                             client: int, 
                             solution: BaseSolution, 
                             evaluator: BaseEvaluator) -> Tuple[float, Any]:
        """
        Finds the best move ('insert' or 'insert_use') for a single client.
        Returns (best_cost, best_move_tuple).
        """
        best_move = None
        
        # We must use the 'sense' to initialize the best cost
        if evaluator.sense == BaseEvaluator.ObjectiveSense.MINIMIZE:
            best_cost = float('inf')
            is_better = lambda new, old: new < old
        else:
            best_cost = float('-inf')
            is_better = lambda new, old: new > old

        # 1. Check 'insert' moves (into *used* routes)
        nodes_in_used_routes = _get_nodes_in_used_routes(solution)
        for n_neighbor in nodes_in_used_routes:
            move = ('insert', client, n_neighbor)
            try:
                # Use the evaluator from your TS example
                cost = evaluator.evaluate_insertion_cost(
                    elem_to_insert=client, elem_new_neighbor=n_neighbor, sol=solution
                )
                
                # Note: your evaluate_move returned solution.cost + delta
                # We assume evaluate_insertion_cost does the same.
                
                if is_better(cost, best_cost):
                    best_cost = cost
                    best_move = move
            except:
                continue # Move is infeasible (e.g., TW violation)

        # 2. Check 'insert_use' moves (into *unused* vehicles)
        unused_vehicle_indices = _get_unused_vehicles(solution)
        for v_idx in unused_vehicle_indices:
            move = ('insert_use', client, v_idx)
            try:
                cost = evaluator.evaluate_insert_use_cost(
                    client=client, vehicle_index=v_idx, sol=solution
                )
                if is_better(cost, best_cost):
                    best_cost = cost
                    best_move = move
            except:
                continue # Move is infeasible
                
        return best_cost, best_move

    def repair(self, 
             partial_solution: BaseSolution, 
             problem: BaseProblemInstance, 
             evaluator: BaseEvaluator) -> BaseSolution:
        
        repaired_sol = partial_solution.copy()
        
        # 1. Find all clients that need to be re-inserted
        clients_to_insert = _get_unvisited_clients(repaired_sol, problem)
        
        # Randomize the insertion order
        random.shuffle(clients_to_insert)
        
        # 2. Iteratively insert each client
        while clients_to_insert:
            client = clients_to_insert.pop(0)
            
            # 3. Find the best possible move for this client
            best_cost, best_move = self._find_best_insertion(
                client, repaired_sol, evaluator
            )
            
            # 4. Apply the best move, if one was found
            if best_move:
                move_type, elem1, elem2 = best_move
                
                # Apply the move using the methods from your TS example
                if move_type == 'insert':
                    repaired_sol.insert_element(elem_to_insert=elem1, elem_new_neighbor=elem2)
                elif move_type == 'insert_use':
                    repaired_sol.insert_into_vehicle(client=elem1, vehicle_index=elem2)
                
                # CRITICAL: Update the solution's cost for the next iteration's
                # delta-evaluation to be correct.
                repaired_sol.cost = best_cost
            else:
                # Client could not be inserted (e.g., no feasible position)
                # The final solution will be "partial" and the main
                # evaluator.evaluate() must assign a high penalty.
                # We can print a warning.
                print(f"Warning: Could not repair solution. Client {client} is unroutable.")
                pass 
                
        return repaired_sol
    
# File: model_definitions.py (or your main script)


# --- 1. Define the pool of operators ---

# A list of destroy operators to choose from
destroy_operators = [
    RandomDestroy(num_to_remove=5),
    RandomDestroy(num_to_remove=10),
    RouteDestroy()
]

# A list of repair operators to choose from
# (We only defined one, but you could add RegretRepair, etc.)
repair_operators = [
    GreedyRepair()
]

# --- 2. Create the ALNS Models ---

# Model 1: A standard ALNS with Simulated Annealing and adaptive weights
alns_adaptive_sa = ALNS(
    destroy_operators=destroy_operators,
    repair_operators=repair_operators,
    
    # Strategy for managing operator weights and selection
    weight_manager=RouletteWheelManager(
        segment_size=100,  # Recalculate weights every 100 iterations
        decay=0.8,         # 80% of old weight, 20% of new score
        reward_points={
            "new_best": 10,
            "better_than_current": 5,
            "accepted": 2,
            "rejected": 0
        }
    ),
    
    # Strategy for accepting/rejecting solutions
    acceptance_criteria=SimulatedAnnealingAcceptance(
        initial_temp=5000,
        cooling_rate=0.998, # Slow cooling
        min_temp=0.1
    ),
    
    # Arguments for BaseMetaheuristic
    time_limit=600,
    max_iterations=10000
)


# Model 2: A "Greedy" LNS (Large Neighborhood Search)
# This model only accepts *improving* solutions and has a
# very simple, non-adaptive operator selection.
alns_greedy_lns = ALNS(
    destroy_operators=[RandomDestroy(num_to_remove=8)], # Only one operator
    repair_operators=[GreedyRepair()],                  # Only one operator
    
    # A "dummy" weight manager (it will always pick the only operators)
    weight_manager=RouletteWheelManager(
        segment_size=1, 
        decay=1.0 # Never update weights, stick to initial
    ),
    
    # A "Hill Climbing" acceptance criterion (temp=0)
    acceptance_criteria=SimulatedAnnealingAcceptance(
        initial_temp=0, # Never accept worse solutions
        cooling_rate=1.0, 
        min_temp=0
    ),
    
    # Arguments for BaseMetaheuristic
    time_limit=600,
    max_iterations=10000
)

# You can now use these models just like your Tabu Search models:
#
# best_sol = alns_adaptive_sa.solve(
#     problem=my_problem_instance,
#     evaluator=my_evaluator,
#     initial_solution=my_initial_solution
# )
