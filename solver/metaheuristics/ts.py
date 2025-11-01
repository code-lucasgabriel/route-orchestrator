from np_solver.core import BaseSolution, BaseEvaluator, BaseProblemInstance
from np_solver.metaheuristics.ts import TabuSearch
from np_solver.metaheuristics.ts import TSNeighborhood
from typing import List, Any, Tuple
import random


class HFFVRPTW_TSNeighborhood(TSNeighborhood):
    """
    A HFFVRPTW neighborhood generator for Tabu Search.

    This class is designed to work with the provided BaseSolution.
    It generates moves based on *client IDs*, which are then passed to the
    solution's custom "swap" and "relocate" methods.

    It generates two types of moves:
    1. Swap ('swap', c1, c2): Swaps the positions of two clients.
    2. Relocate ('relocate', c_move, n_after): Moves a client to be after another node.
    """
    
    def generate_moves(self, solution: BaseSolution, problem: BaseProblemInstance) -> List[Any]:
        """
        Generates all possible Swap and Relocate moves.
        - Assumes the depot is node "0".
        """
        
        clients_in_solution = []
        nodes_in_solution = [] # Includes clients AND all depot visits
        
        # This assumes solution is a list of routes, e.g., [[0, 1, 2, 0], [0, 3, 0]]
        for route in solution:
            for node in route:
                nodes_in_solution.append(node)
                if node != 0: # Assuming 0 is the depot
                    clients_in_solution.append(node)
        
        all_problem_clients = set(problem.C)
        clients_in_solution_set = set(clients_in_solution)

        unvisited_clients = list(all_problem_clients - clients_in_solution_set)

        moves = []
        

        #* Intensification moves:
        # 1. Generate 'swap' moves (between any two clients)
        # O(N_clients^2)
        for i in range(len(clients_in_solution)):
            for j in range(i + 1, len(clients_in_solution)):
                c1 = clients_in_solution[i]
                c2 = clients_in_solution[j]
                # Move is hashable tuple: (type, client1, client2)
                moves.append(('swap', c1, c2))

        # 2. Generate 'relocate' moves
        # O(N_clients * N_nodes)
        for c_to_move in clients_in_solution:
            for n_neighbor in nodes_in_solution:
                if c_to_move == n_neighbor:
                    # Can't relocate a node to be after itself,
                    # as it's removed before the neighbor is found.
                    continue
                
                # Move is hashable tuple: (type, client_to_move, new_neighbor_node)
                moves.append(('relocate', c_to_move, n_neighbor))
        
        #* Diversification moves
        for c_to_insert in unvisited_clients:
            # Insert after any node already in the solution
            for n_neighbor in nodes_in_solution:
                moves.append(('insert', c_to_insert, n_neighbor))
                
        # 'exchange' moves (swap *unvisited* for *visited*)
        for c_to_insert in unvisited_clients:
            for c_to_remove in clients_in_solution:
                moves.append(('exchange', c_to_insert, c_to_remove))

        # Shuffling is a good idea when the move list is huge
        random.shuffle(moves)

        return moves

    def apply_move(self, solution: BaseSolution, move: Any) -> BaseSolution:
        """
        Applies a move and returns a *new* solution object.
        It calls the custom "swap" and "relocate" methods of the
        BaseSolution.
        """
        # CRITICAL: Start from a deep copy
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
            
        else:
            raise ValueError(f"Unknown move type: {move_type}")
            
        return new_sol

    def evaluate_move(self, solution: BaseSolution, move: Any, 
                      evaluator: BaseEvaluator) -> float:
        """
        Efficiently calculates the *cost of the new solution* after the move
        using the evaluator's delta-evaluation methods.
        
        This assumes the evaluator's delta methods also expect *client IDs*.
        """
        move_type, elem1, elem2 = move
        delta_cost = 0.0

        if move_type == 'swap':
            # Assumes evaluator.evaluate_swap_cost(c1, c2, sol)
            delta_cost = evaluator.evaluate_swap_cost(
                elem1, elem2, solution
            )
        elif move_type == 'relocate':
            # Assumes evaluator.evaluate_relocation_cost(c_move, n_after, sol)
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
        else:
            raise ValueError(f"Unknown move type: {move_type}")

        # Return the *full cost* of the new solution
        return solution.cost + delta_cost

ts_tenure5 = TabuSearch(
    tenure=5,
    neighborhood_strategy=HFFVRPTW_TSNeighborhood()
)

ts_tenure0 = TabuSearch(
    tenure=0,
    neighborhood_strategy=HFFVRPTW_TSNeighborhood()
)
