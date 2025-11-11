from np_solver.core import BaseSolution, BaseEvaluator, BaseProblemInstance
from np_solver.metaheuristics.ts import TabuSearch
from np_solver.metaheuristics.ts.interface import TSNeighborhood
from typing import List, Any, Tuple
import random
from settings import TIME_LIMIT

class HFFVRPTW_TSNeighborhood(TSNeighborhood):
    """
    A HFFVRPTW neighborhood generator for Tabu Search.

    This class is designed to work with the provided BaseSolution.
    It generates moves based on *client IDs*, which are then passed to the
    solution's custom "swap" and "relocate" methods.

    It generates two types of moves:
    1. Swap ('swap', c1, c2): Swaps the positions of two clients.
    2. Relocate ('relocate', c_move, n_after): Moves a client to be after another node.
    3. Insert ('insert', c_insert, n_after): Inserts an unvisited client after a node
       in an *already used* route.
    4. Exchange ('exchange', c_in, c_out): Swaps an unvisited client for a
       visited one.
    5. Insert-Use ('insert_use', c_insert, v_idx): Inserts an unvisited client
       into a *previously unused* vehicle, activating it.
    """

    def _get_unused_vehicles(self, solution: BaseSolution) -> List[int]:
        """
        Helper method to get an index list of all unused vehicles (with a route [0, 0])
        """
        empty_route = [0, 0]
        unused_vehicles = []

        for idx, route in enumerate(solution):
            if route == empty_route:
                unused_vehicles.append(idx)
        
        return unused_vehicles
    
    def generate_moves(self, solution: BaseSolution, problem: BaseProblemInstance) -> List[Any]:
        """
        Generates all possible Swap, Relocate, Insert, Exchange, and Insert-Use moves.
        """
        
        clients_in_solution = []
        nodes_in_solution = [] # Includes clients AND depot visits for *used routes only*
        
        unused_vehicle_indices = self._get_unused_vehicles(solution)
        
        for v_idx, route in enumerate(solution):
            # Only iterate nodes in *USED* routes
            if v_idx not in unused_vehicle_indices:
                for node in route:
                    nodes_in_solution.append(node)
                    if node != 0: # Assuming 0 is the depot
                        clients_in_solution.append(node)
        
        all_problem_clients = set(problem.C)
        clients_in_solution_set = set(clients_in_solution)

        unvisited_clients = list(all_problem_clients - clients_in_solution_set)

        moves = []
        
        #* Intensification moves (on clients in *used* routes)
        # 1. Generate 'swap' moves
        for i in range(len(clients_in_solution)):
            for j in range(i + 1, len(clients_in_solution)):
                c1 = clients_in_solution[i]
                c2 = clients_in_solution[j]
                moves.append(('swap', c1, c2))

        # 2. Generate 'relocate' moves
        for c_to_move in clients_in_solution:
            for n_neighbor in nodes_in_solution:
                if c_to_move == n_neighbor:
                    continue
                moves.append(('relocate', c_to_move, n_neighbor))
        
        #* Diversification moves
        for c_to_insert in unvisited_clients:
            
            # 1. 'insert': Insert into *already used* routes
            #    (nodes_in_solution only contains nodes from used routes now)
            for n_neighbor in nodes_in_solution:
                moves.append(('insert', c_to_insert, n_neighbor))

            # 2. 'insert_use': Insert into an *unused* vehicle
            for v_idx in unused_vehicle_indices:
                moves.append(('insert_use', c_to_insert, v_idx))

            # 3. 'exchange': Swap unvisited for visited
            for c_to_remove in clients_in_solution:
                moves.append(('exchange', c_to_insert, c_to_remove))

        random.shuffle(moves)

        return moves

    def apply_move(self, solution: BaseSolution, move: Any) -> BaseSolution:
        """
        Applies a move and returns a *new* solution object.
        """
        new_sol = solution.copy()
        
        move_type, elem1, elem2 = move

        if move_type == 'swap':
            new_sol.swap(elem1, elem2)
        elif move_type == 'relocate':
            new_sol.relocate(elem_to_move=elem1, elem_new_neighbor=elem2)
        elif move_type == 'insert':
            new_sol.insert_element(elem_to_insert=elem1, elem_new_neighbor=elem2)
        elif move_type == 'exchange':
            new_sol.exchange(element_to_add=elem1, element_to_remove=elem2)
            
        elif move_type == 'insert_use':
            # elem1 = client_to_insert
            # elem2 = vehicle_index
            new_sol.insert_into_vehicle(client=elem1, vehicle_index=elem2)
            
        else:
            raise ValueError(f"Unknown move type: {move_type}")
            
        return new_sol

    def evaluate_move(self, solution: BaseSolution, move: Any, 
                      evaluator: BaseEvaluator) -> float:
        """
        Efficiently calculates the *cost of the new solution* after the move.
        """
        move_type, elem1, elem2 = move
        delta_cost = 0.0

        if move_type == 'swap':
            delta_cost = evaluator.evaluate_swap_cost(
                elem1, elem2, solution
            )
        elif move_type == 'relocate':
            delta_cost = evaluator.evaluate_relocation_cost(
                elem_to_move=elem1, elem_new_neighbor=elem2, sol=solution
            )
        elif move_type == 'insert':
            delta_cost = evaluator.evaluate_insertion_cost(
                elem_to_insert=elem1, elem_new_neighbor=elem2, sol=solution
            )
        elif move_type == 'exchange':
            delta_cost = evaluator.evaluate_exchange_cost(
                elem_in=elem1, elem_out=elem2, sol=solution
            )
            
        elif move_type == 'insert_use':
            # elem1 = client_to_insert
            # elem2 = vehicle_index
            delta_cost = evaluator.evaluate_insert_use_cost(
                client=elem1, vehicle_index=elem2, sol=solution
            )
            
        else:
            raise ValueError(f"Unknown move type: {move_type}")

        # Return the *full cost* of the new solution
        return solution.cost + delta_cost
    
    
ts_tenure5 = TabuSearch(
    tenure=5,
    neighborhood_strategy=HFFVRPTW_TSNeighborhood(),
    time_limit=TIME_LIMIT
)

ts_tenure0 = TabuSearch(
    tenure=0,
    neighborhood_strategy=HFFVRPTW_TSNeighborhood(),
    time_limit=TIME_LIMIT
)
