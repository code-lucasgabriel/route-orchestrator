from np_solver.core import BaseSolution
from typing import Tuple

class HFFVRPTWSolution(BaseSolution):
    # helper to find a client
    def _find_node(self, node_to_find) -> Tuple[int, int]:
        """Finds (route_idx, node_idx) of a node. Returns (None, None) if not found."""
        for r_idx, route in enumerate(self):
            try:
                n_idx = route.index(node_to_find)
                return (r_idx, n_idx)
            except ValueError:
                continue
        return (None, None)

    def copy(self):
        """Creates a deep copy of the routes."""
        new_sol = HFFVRPTWSolution()
        for route in self:
            new_sol.append(route[:])
        new_sol.cost = self.cost
        return new_sol
        
    def swap(self, elem1, elem2):
        """
        Overrides BaseSolution.swap to work on nested routes.
        Finds elem1 and elem2 and swaps their positions.
        """
        r1_idx, n1_idx = self._find_node(elem1)
        r2_idx, n2_idx = self._find_node(elem2)
        
        if r1_idx is None or r2_idx is None:
            raise ValueError("Element not found in solution")
            
        self[r1_idx][n1_idx], self[r2_idx][n2_idx] = self[r2_idx][n2_idx], self[r1_idx][n1_idx]
        
    def relocate(self, elem_to_move, elem_new_neighbor):
        """
        Overrides BaseSolution.relocate to work on nested routes.
        Moves elem_to_move to be immediately after elem_new_neighbor.
        """
        if elem_to_move == elem_new_neighbor:
            return

        r_move_idx, n_move_idx = self._find_node(elem_to_move)
        if r_move_idx is None:
            raise ValueError("Element to move not found")
            
        node = self[r_move_idx].pop(n_move_idx) 
        
        r_neighbor_idx, n_neighbor_idx = self._find_node(elem_new_neighbor)
        if r_neighbor_idx is None:
            raise ValueError("Neighbor element not found")
        
        self[r_neighbor_idx].insert(n_neighbor_idx + 1, node)

    def insert_element(self, elem_to_insert, elem_new_neighbor):
        """
        Inserts a new element to be immediately after elem_new_neighbor.
        """
        r_neighbor_idx, n_neighbor_idx = self._find_node(elem_new_neighbor)
        
        if r_neighbor_idx is None:
            r_neighbor_idx = 0
            n_neighbor_idx = 0 
            
        self[r_neighbor_idx].insert(n_neighbor_idx + 1, elem_to_insert)

    def exchange(self, element_to_add, element_to_remove):
        """
        Swaps an element inside the solution (elem_to_remove) with an element outside of it (elem_to_add).
        """
        r_out_idx, n_out_idx = self._find_node(element_to_remove)
        
        if r_out_idx is None:
             raise ValueError(f"Element to remove {element_to_add} not found in solution")
        
        self[r_out_idx][n_out_idx] = element_to_add