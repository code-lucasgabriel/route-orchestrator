import random
from typing import List, Tuple
from np_solver.core import BaseSolution, BaseProblemInstance, BaseEvaluator
from solver.problem.hffvrptw_solution import HFFVRPTWSolution


class HFFVRPTWConstructiveHeuristic:
    """
    Builds an initial solution using a Greedy Insertion heuristic.

    It iterates through all clients in a random order and inserts
    each client into the "cheapest" feasible position found across all routes.
    """
    
    def __init__(self, problem: BaseProblemInstance, evaluator: BaseEvaluator, seed=42):
        self.problem = problem
        self.evaluator = evaluator
        self.seed = seed
        random.seed(self.seed)

    def build(self) -> BaseSolution:
        """
        Builds and returns an initial feasible solution.
        """
        problem = self.problem
        
        sol_list = [[0, 0] for _ in problem.K]
        
        # Cache the current cost of each route (all 0.0)
        route_costs = [0.0] * len(sol_list)

        # 2. Get all clients and shuffle them
        unvisited_clients = list(problem.C)
        random.shuffle(unvisited_clients)
        
        clients_not_inserted = []

        # 3. Loop through each client
        for client in unvisited_clients:
            best_insertion = {
                'route_idx': None,
                'insert_pos': None,
                'cost_delta': float('inf'),
                'new_cost': float('inf')
            }
            
            # 4. Find the best place (cheapest) to insert this client
            for k in range(len(sol_list)):
                current_route = sol_list[k]
                
                # 5. Try every possible insertion spot in the route
                # (from index 1 up to, and including, the last depot)
                for i in range(1, len(current_route)):
                    
                    # Create a candidate route
                    candidate_route = current_route[:i] + [client] + current_route[i:]
                    
                    # 6. Check if this new route is feasible
                    new_cost, is_feasible = self.evaluator._get_route_cost_and_feasibility(candidate_route,k)
                    
                    if is_feasible:
                        # 7. It's feasible. Is it the best one so far?
                        old_cost = route_costs[k]
                        cost_delta = new_cost - old_cost
                        
                        if cost_delta < best_insertion['cost_delta']:
                            best_insertion = {
                                'route_idx': k,
                                'insert_pos': i,
                                'cost_delta': cost_delta,
                                'new_cost': new_cost
                            }
            
            # 8. After checking all spots, make the best insertion if found
            if best_insertion['route_idx'] is not None:
                k = best_insertion['route_idx']
                i = best_insertion['insert_pos']
                
                # Insert the client into the best-found route
                sol_list[k].insert(i, client)
                # Update the cached cost for that route
                route_costs[k] = best_insertion['new_cost']
            else:
                # No feasible spot was found for this client
                clients_not_inserted.append(client)
        
        if clients_not_inserted:
            print(f"[ConstructiveHeuristic] Warning: Could not insert {len(clients_not_inserted)} clients.")
        
        # 9. Return the final solution object
        return HFFVRPTWSolution.from_list(sol_list, self.evaluator)