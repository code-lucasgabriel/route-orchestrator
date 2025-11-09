from np_solver.core import BaseEvaluator, BaseProblemInstance, BaseSolution
from np_solver.metaheuristics.alns import ALNS
from np_solver.metaheuristics.alns.components import ALNSAcceptance, ALNSDestroy, ALNSRepair, ALNSWeightManager, SimulatedAnnealingAcceptance, RouletteWheelManager
from np_solver.metaheuristics.alns.interface import ALNSDestroy, ALNSRepair
import random
from typing import List, Set, Tuple, Any
import math
from settings import TIME_LIMIT

class ShawDestroy(ALNSDestroy):
    """
    Shaw Removal (Relatedness-Based) Operator.
    
    This operator removes clients that are "related" to each other,
    based on the idea that related clients are easier to reschedule
    together in new ways.
    
    Relatedness is measured by a weighted sum of:
    1. Distance (closer = more related)
    2. Time Window Overlap (more overlap = more related)
    
    Inspired by Shaw (1998) and mentioned in Pereira et al..
    """
    def __init__(self, num_to_remove: int, 
                 weight_distance: float = 1.0, 
                 weight_time: float = 0.5,
                 determinism_param: float = 3.0):
        self.num_to_remove = num_to_remove
        self.weight_distance = weight_distance
        self.weight_time = weight_time
        self.determinism_param = determinism_param # Controls greediness

    def _calculate_relatedness(self, client1: int, client2: int, 
                               problem: BaseProblemInstance) -> float:
        """Calculates relatedness between two clients."""
        
        # 1. Normalized Distance
        dist = problem.d[client1][client2]
        max_dist = problem.max_distance  # Assumes problem has this attr
        norm_dist = dist / max_dist
        
        # 2. Normalized Time Window Overlap
        e1, l1 = problem.e[client1], problem.l[client1]
        e2, l2 = problem.e[client2], problem.l[client2]
        
        # Calculate overlap
        overlap = max(0, min(l1, l2) - max(e1, e2))
        
        # Normalize by the total time span (e.g., max end time)
        max_time = problem.max_tw_end # Assumes problem has this attr
        norm_time_overlap = overlap / max_time
        
        # Relatedness (lower is more related)
        relatedness = (self.weight_distance * norm_dist + 
                       self.weight_time * (1 - norm_time_overlap)) # 1-overlap
        
        return relatedness

    def destroy(self, solution: BaseSolution, problem: BaseProblemInstance) -> BaseSolution:
        new_sol = solution.copy()
        clients_in_solution = _get_clients_in_solution(new_sol)
        
        if not clients_in_solution:
            return new_sol
        
        n = min(self.num_to_remove, len(clients_in_solution))
        
        # Start with a random client
        first_client = random.choice(clients_in_solution)
        clients_to_remove = [first_client]
        clients_in_solution.remove(first_client)
        
        while len(clients_to_remove) < n and clients_in_solution:
            # Pick a random client *from the removed list* to be the seed
            seed_client = random.choice(clients_to_remove)
            
            # Find client in solution most related to the seed
            relatedness_scores = []
            for other_client in clients_in_solution:
                score = self._calculate_relatedness(seed_client, other_client, problem)
                relatedness_scores.append((score, other_client))
            
            # Sort by relatedness (ascending, lower is better)
            relatedness_scores.sort(key=lambda x: x[0])
            
            # Pick from the top related clients using determinism param
            # (higher param = more greedy/deterministic)
            idx = int(len(relatedness_scores) ** self.determinism_param * random.random())
            idx = min(idx, len(relatedness_scores) - 1) # Bound check
            
            chosen_client = relatedness_scores[idx][1]
            
            clients_to_remove.append(chosen_client)
            clients_in_solution.remove(chosen_client)

        # Remove all chosen clients from the solution
        for client in clients_to_remove:
            try:
                new_sol.remove_client(client)
            except Exception as e:
                print(f"Warning: Could not remove client {client}. Error: {e}")
                
        return new_sol


class RegretRepair(ALNSRepair):
    """
    Regret-k Repair (Regret-Based) Operator.
    
    This operator inserts clients in "regret" order.
    The regret for a client is the cost difference between its
    *best* insertion position and its *k-th best* insertion position.
    
    The idea is to prioritize clients with high regret, as they
    have few good options and "must" be placed soon.
    """
    def __init__(self, k_regret: int = 3):
        # k_regret=1 is equivalent to GreedyRepair
        # k_regret=2 calculates best_cost - 2nd_best_cost
        self.k_regret = max(2, k_regret) 

    def repair(self, 
             partial_solution: BaseSolution, 
             problem: BaseProblemInstance, 
             evaluator: BaseEvaluator) -> BaseSolution:
        
        repaired_sol = partial_solution.copy()
        
        clients_to_insert = _get_unvisited_clients(repaired_sol, problem)
        
        while clients_to_insert:
            client_regrets = []

            # 1. Find insertion costs for ALL clients in ALL positions
            for client in clients_to_insert:
                insertion_options = []
                
                # CORRIGIDO: Checa 'insert' moves em rotas existentes
                nodes_in_used_routes = _get_nodes_in_used_routes(repaired_sol)
                for n_neighbor in nodes_in_used_routes:
                    cost = evaluator.evaluate_insertion_cost(
                        elem_to_insert=client, elem_new_neighbor=n_neighbor, sol=repaired_sol
                    )
                    if cost < float('inf'):
                        move = ('insert', client, n_neighbor)
                        insertion_options.append((cost, move))
                
                # CORRIGIDO: Sempre checa 'insert_use', não apenas se insertion_options estiver vazio
                # Isso permite comparar o custo de abrir nova rota vs inserir em rota existente
                unused_vehicle_indices = _get_unused_vehicles(repaired_sol)
                for v_idx in unused_vehicle_indices:
                    # Calculate actual cost without penalty subtraction
                    new_route = [0, client, 0]
                    route_cost, is_feasible = evaluator._get_route_cost_and_feasibility(new_route, v_idx)
                    
                    if is_feasible:
                        move = ('insert_use', client, v_idx)
                        insertion_options.append((route_cost, move))

                # If no feasible insertions, client is unroutable
                if not insertion_options:
                    continue

                # 2. Calculate regret for this client
                insertion_options.sort(key=lambda x: x[0])
                
                best_cost, best_move = insertion_options[0]
                
                regret = 0.0
                k = min(self.k_regret, len(insertion_options))
                for i in range(1, k):
                    regret += (insertion_options[i][0] - best_cost)
                
                # Store (regret, best_cost_tiebreak, client, best_move)
                # We sort by highest regret, then lowest cost
                client_regrets.append((-regret, best_cost, client, best_move))
            
            # 3. If no clients can be inserted, stop
            if not client_regrets:
                unroutable = [c for c in clients_to_insert]
                print(f"Warning: Could not repair solution. Unroutable clients: {unroutable}")
                break
                
            # 4. Select the client with the highest regret (and best cost)
            client_regrets.sort() # Sorts by -regret (so highest is first)
            
            best_regret_data = client_regrets[0]
            best_cost = best_regret_data[1]
            client_to_insert = best_regret_data[2]
            best_move = best_regret_data[3]
            
            # 5. Apply the best move for that client
            move_type, elem1, elem2 = best_move
            
            if move_type == 'insert':
                repaired_sol.insert_element(elem_to_insert=elem1, elem_new_neighbor=elem2)
            elif move_type == 'insert_use':
                repaired_sol.insert_into_vehicle(client=elem1, vehicle_index=elem2)
            
            clients_to_insert.remove(client_to_insert)

        # Final evaluation to ensure cost is set correctly
        repaired_sol.cost = evaluator.evaluate(repaired_sol)
        return repaired_sol


class WorstDestroy(ALNSDestroy):
    """
    Worst Removal (Cost-Based) Operator.
    
    This operator removes clients that contribute the most to the
    solution's cost. This is a greedy way to try and remove
    "bad" decisions.
    
    Since the destroy operator doesn't get the evaluator, we use
    a proxy for cost: the extra distance a client adds to its route.
    
    Mentioned in Pereira et al..
    """
    def __init__(self, num_to_remove: int, determinism_param: float = 3.0):
        self.num_to_remove = num_to_remove
        self.determinism_param = determinism_param # Controls greediness

    def _calculate_removal_cost_proxy(self, client: int, 
                                      solution: BaseSolution, 
                                      problem: BaseProblemInstance) -> float:
        """
        Calculates the cost (distance) saved by removing the client.
        Returns a *higher* value for "worse" clients.
        """
        r_idx, n_idx = solution._find_node(client)
        
        if r_idx is None:
            return -float('inf') # Should not happen

        route = solution[r_idx]
        
        # Get neighbors
        prev_node = route[n_idx - 1]
        next_node = route[n_idx + 1]
        
        # Cost of having the client
        cost_with_client = problem.d[prev_node][client] + problem.d[client][next_node]
        
        # Cost of *not* having the client
        cost_without_client = problem.d[prev_node][next_node]
        
        # Return the difference. Higher = worse.
        return cost_with_client - cost_without_client

    def destroy(self, solution: BaseSolution, problem: BaseProblemInstance) -> BaseSolution:
        new_sol = solution.copy()
        
        n = min(self.num_to_remove, len(_get_clients_in_solution(new_sol)))
        
        for _ in range(n):
            clients_in_solution = _get_clients_in_solution(new_sol)
            if not clients_in_solution:
                break
            
            # Calculate cost for all clients *currently* in the solution
            costs = []
            for client in clients_in_solution:
                cost = self._calculate_removal_cost_proxy(client, new_sol, problem)
                costs.append((cost, client))
            
            # Sort by cost (descending, higher is worse)
            costs.sort(key=lambda x: x[0], reverse=True)
            
            # Pick from the top worst clients using determinism param
            idx = int(len(costs) ** self.determinism_param * random.random())
            idx = min(idx, len(costs) - 1)
            
            worst_client = costs[idx][1]
            
            try:
                new_sol.remove_client(worst_client)
            except Exception as e:
                print(f"Warning: Could not remove client {worst_client}. Error: {e}")
        
        return new_sol

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
        """
        new_sol = solution.copy()
        clients_in_solution = _get_clients_in_solution(new_sol)
        
        if not clients_in_solution:
            return new_sol # Nothing to remove
    
        n = min(self.num_to_remove, len(clients_in_solution))
        
        clients_to_remove = random.sample(clients_in_solution, n)
        
        for client in clients_to_remove:
            try:
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
        
        IMPORTANTE: Avalia TODOS os movimentos (insert e insert_use) e escolhe
        o de menor custo, não priorizando arbitrariamente rotas existentes.
        """
        best_move = None
        
        if evaluator.sense == BaseEvaluator.ObjectiveSense.MINIMIZE:
            best_cost = float('inf')
            is_better = lambda new, old: new < old
        else:
            best_cost = float('-inf')
            is_better = lambda new, old: new > old

        # 1. Avalia a inserção em rotas existentes
        nodes_in_used_routes = _get_nodes_in_used_routes(solution)
        for n_neighbor in nodes_in_used_routes:
            move = ('insert', client, n_neighbor)
            
            cost = evaluator.evaluate_insertion_cost(
                elem_to_insert=client, elem_new_neighbor=n_neighbor, sol=solution
            )
            
            if is_better(cost, best_cost):
                best_cost = cost
                best_move = move

        # 2. Avalia a abertura de novas rotas (e compara com o best_cost atual)
        # CORRIGIDO: Agora sempre avalia insert_use, não apenas quando best_cost == inf
        unused_vehicle_indices = _get_unused_vehicles(solution)
        for v_idx in unused_vehicle_indices:
            move = ('insert_use', client, v_idx)
            
            # Calcula o custo real de abrir uma nova rota
            new_route = [0, client, 0]
            route_cost, is_feasible = evaluator._get_route_cost_and_feasibility(new_route, v_idx)
            
            if is_feasible and is_better(route_cost, best_cost):
                best_cost = route_cost
                best_move = move
                
        return best_cost, best_move

    def repair(self, 
             partial_solution: BaseSolution, 
             problem: BaseProblemInstance, 
             evaluator: BaseEvaluator) -> BaseSolution:
        
        repaired_sol = partial_solution.copy()
        
        # 1. Find all clients that need to be re-inserted
        clients_to_insert = _get_unvisited_clients(repaired_sol, problem)
        
        # Sort clients by time window start (earlier clients first)
        # This helps maintain feasibility when reinserting
        clients_to_insert.sort(key=lambda c: problem.e[c])
        
        # Track unroutable clients to prevent infinite loops
        unroutable_clients = []
        max_attempts = 3  # Maximum attempts to insert each client
        attempt_count = {}
        
        # 2. Iteratively insert each client
        while clients_to_insert:
            client = clients_to_insert.pop(0)
            
            # Track attempts for this client
            attempt_count[client] = attempt_count.get(client, 0) + 1
            
            # If we've tried too many times, mark as unroutable
            if attempt_count[client] > max_attempts:
                unroutable_clients.append(client)
                continue
            
            best_cost, best_move = self._find_best_insertion(
                client, repaired_sol, evaluator
            )
            
            if best_move:
                move_type, elem1, elem2 = best_move
                
                # Apply the move using the methods from your TS example
                if move_type == 'insert':
                    repaired_sol.insert_element(elem_to_insert=elem1, elem_new_neighbor=elem2)
                elif move_type == 'insert_use':
                    repaired_sol.insert_into_vehicle(client=elem1, vehicle_index=elem2)
            else:
                # No feasible move found, try again later if attempts remain
                if attempt_count[client] < max_attempts:
                    clients_to_insert.append(client)
                else:
                    unroutable_clients.append(client)
        
        # Warn if there are unroutable clients
        if unroutable_clients:
            print(f"Warning: GreedyRepair could not insert {len(unroutable_clients)} clients: {unroutable_clients[:10]}")
        
        # Final evaluation to ensure cost is set correctly
        repaired_sol.cost = evaluator.evaluate(repaired_sol)
        return repaired_sol
    
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

# Model 1: ALNS with Simulated Annealing and adaptive weights

destroy_operators = [
    RandomDestroy(num_to_remove=5),
    RandomDestroy(num_to_remove=10),
    RouteDestroy(),
    ShawDestroy(num_to_remove=8, determinism_param=4),
    WorstDestroy(num_to_remove=6, determinism_param=3)
]

# A list of repair operators to choose from
repair_operators = [
    GreedyRepair(),
    RegretRepair(k_regret=3),
    RegretRepair(k_regret=5)
]

alns_adaptive_sa = ALNS(
    destroy_operators=destroy_operators,
    repair_operators=repair_operators,

    weight_manager=RouletteWheelManager(
        segment_size=100,
        decay=0.8,
        reward_points={
            "new_best": 10,
            "better_than_current": 5,
            "accepted": 2,
            "rejected": 0
        }
    ),
    acceptance_criteria=SimulatedAnnealingAcceptance(
        initial_temp=5000,
        cooling_rate=0.998,
        min_temp=0.1
    ),
    time_limit=TIME_LIMIT,
    max_iterations=10000
)


# Model 2: A "Greedy" LNS (Large Neighborhood Search)
# This model only accepts *improving* solutions and has a
# very simple, non-adaptive operator selection.
alns_greedy_lns = ALNS(
    destroy_operators=[RandomDestroy(num_to_remove=8)], # Simple random removal
    repair_operators=[GreedyRepair()],                  # Simple greedy repair
    weight_manager=RouletteWheelManager(
        segment_size=1, 
        decay=1.0 # No decay: each operator treated equally always
    ),
    # Greedy acceptance: only accept improving solutions
    acceptance_criteria=SimulatedAnnealingAcceptance(
        initial_temp=0, # Temp = 0 means only accept better solutions
        cooling_rate=1.0, 
        min_temp=0
    ),
    # 10-minute time limit (same as Tabu Search)
    time_limit=TIME_LIMIT,
    max_iterations=10000
)

