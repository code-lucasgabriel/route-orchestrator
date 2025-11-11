from np_solver.core import BaseSolution
from typing import Tuple, Optional

class HFFVRPTWSolution(BaseSolution):
    # helper to find a client
    def _find_node(self, node_to_find) -> Tuple[Optional[int], Optional[int]]:
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
        
        # Explicitly copy the cost to the new solution
        new_sol.cost = self.cost
        
        # If you track unvisited clients on the solution, copy them too
        # e.g., if hasattr(self, 'unvisited_clients'):
        #     new_sol.unvisited_clients = self.unvisited_clients[:]
            
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
        self.cost = None # Invalidate cost cache

    def relocate(self, elem_to_move, elem_new_neighbor):
        """
        Overrides BaseSolution.relocate to work on nested routes.
        Moves elem_to_move to be immediately after elem_new_neighbor.
        """
        if elem_to_move == elem_new_neighbor:
            return

        r_move_idx, n_move_idx = self._find_node(elem_to_move)
        if r_move_idx is None:
            raise ValueError(f"Element to move {elem_to_move} not found")
            
        node = self[r_move_idx].pop(n_move_idx) 
        
        r_neighbor_idx, n_neighbor_idx = self._find_node(elem_new_neighbor)
        if r_neighbor_idx is None or n_neighbor_idx is None:
            # If neighbor not found, it might have been the node we just popped
            # In that case, we can't relocate relative to itself.
            # But if it's a different node, it's an error.
            # We'll re-add the node to prevent solution corruption
            self[r_move_idx].insert(n_move_idx, node)
            raise ValueError(f"Neighbor element {elem_new_neighbor} not found")
        
        self[r_neighbor_idx].insert(n_neighbor_idx + 1, node)
        self.cost = None # Invalidate cost cache

    def insert_element(self, elem_to_insert, elem_new_neighbor):
        """
        Inserts a new element to be immediately after elem_new_neighbor.
        """
        r_neighbor_idx, n_neighbor_idx = self._find_node(elem_new_neighbor)
        
        if r_neighbor_idx is None or n_neighbor_idx is None:
            # If neighbor isn't found, default to inserting in the
            # first position of the first route.
            r_neighbor_idx = 0
            n_neighbor_idx = 0 
            
        self[r_neighbor_idx].insert(n_neighbor_idx + 1, elem_to_insert)
        self.cost = None # Invalidate cost cache

    def exchange(self, element_to_add, element_to_remove):
        """
        Swaps an element inside the solution (elem_to_remove) with an element 
        outside of it (elem_to_add).
        """
        r_out_idx, n_out_idx = self._find_node(element_to_remove)
        
        if r_out_idx is None:
             raise ValueError(f"Element to remove {element_to_remove} not found in solution")
        
        self[r_out_idx][n_out_idx] = element_to_add
        self.cost = None # Invalidate cost cache
    
    def insert_into_vehicle(self, client: int, vehicle_index: int):
        """
        Activates a new vehicle by inserting a client into its route.
        Changes the route from [0, 0] to [0, client, 0].
        """
        
        # Check if vehicle_index is valid
        if vehicle_index < 0 or vehicle_index >= len(self):
            raise IndexError(f"Vehicle index {vehicle_index} is out of bounds.")
            
        route = self[vehicle_index]
        
        # Check if the route is actually empty
        if route == [0, 0]:
            # Assign the client to the route
            self[vehicle_index] = [0, client, 0]
            self.cost = None # Invalidate cost cache
        else:
            # This should not happen if the neighborhood generator is correct
            raise RuntimeError(
                f"Attempted 'insert_use' move on a non-empty route. "
                f"Vehicle {vehicle_index} has route: {route}"
            )
        
    def remove_client(self, client_node: int):
        (route_idx, client_idx) = self._find_node(client_node)
        if route_idx is None or client_idx is None:
            return
        self[route_idx].pop(client_idx)
