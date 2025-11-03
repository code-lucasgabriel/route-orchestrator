from np_solver.core import BaseEvaluator, BaseSolution
from solver.problem.model import build_constraints, build_objective
from typing import Tuple
from solver.problem.hffvrptw_solution import HFFVRPTWSolution
# Assuming HFFVRPTWProblem is imported or available in the scope
# from solver.problem.hffvrptw_problem import HFFVRPTWProblem 

class HFFVRPTWEvaluator(BaseEvaluator):
    sense = BaseEvaluator.ObjectiveSense.MINIMIZE

    def constraints(self, sol: HFFVRPTWSolution):
        """
        Note: This is a high-level constraint check for a complete solution.
        The local search relies on _get_route_cost_and_feasibility.
        """
        return build_constraints(self.problem, sol)

    def objective_function(self, sol: HFFVRPTWSolution):
        """
        Note: This is a high-level objective function for a complete solution.
        The local search relies on _get_route_cost_and_feasibility.
        """
        return build_objective(self.problem, sol)

    def _get_route_cost_and_feasibility(self, route: list, vehicle_idx: int) -> Tuple[float, bool]:
        """
        The core logic of the evaluator.
        Calculates the total cost (fixed + variable) and checks
        all capacity and time window constraints for a *single* route.
        """
        # --- 1. Empty route is feasible and free ---
        if len(route) <= 2:  # e.g., [0, 0]
            return (0.0, True)

        try:
            # FIX: Get the vehicle tuple (type, instance_idx) from the problem's list K
            vehicle_tuple = self.problem.K[vehicle_idx]
            vehicle_type = vehicle_tuple[0]
        except IndexError:
            # This route doesn't have a corresponding vehicle
            return (self._get_infeasible_cost(), False)

        # FIX: Get fixed cost from F_p dict using the vehicle *type*
        cost = self.problem.F_p[vehicle_type]
        
        # FIX: Get vehicle properties from problem dicts using vehicle *type*
        vehicle_capacity = self.problem.Q_p[vehicle_type]
        vehicle_var_cost = self.problem.c_p[vehicle_type]

        total_demand = 0
        travel_distance = 0
        current_time = self.problem.e[0]  # Start time at the depot (node 0)

        for i in range(len(route) - 1):
            u = route[i]
            v = route[i+1]

            # --- 2. Add travel cost & time ---
            # FIX: Get distance and travel time from problem matrices d and t
            dist = self.problem.d[u][v]
            travel_time = self.problem.t[u][v]
            
            travel_distance += dist
            arrival_at_v = current_time + travel_time

            # --- 3. Check Time Window Feasibility ---
            # FIX: Get client data from problem lists (e, l, s, q)
            tw_start_v = self.problem.e[v]
            tw_end_v = self.problem.l[v]
            service_time_v = self.problem.s[v]
            demand_v = self.problem.q[v]

            # Wait if arriving early
            service_begin_at_v = max(arrival_at_v, tw_start_v)

            # Check if late
            if service_begin_at_v > tw_end_v:
                return (self._get_infeasible_cost(), False)  # Infeasible: Time window violation

            # --- 4. Add to demand ---
            total_demand += demand_v

            # --- 5. Update time for next leg ---
            current_time = service_begin_at_v + service_time_v

        # --- 6. Final Capacity Check ---
        # FIX: Compare against the vehicle_capacity fetched earlier
        if total_demand > vehicle_capacity:
            return (self._get_infeasible_cost(), False)  # Infeasible: Capacity violation

        # --- 7. All checks passed ---
        # FIX: Use the vehicle_var_cost fetched earlier
        cost += travel_distance * vehicle_var_cost
        return (cost, True)

    # --- Delta-Evaluation Methods ---
    # (These methods were logically correct and relied on the 
    # _get_route_cost_and_feasibility method. No changes needed here.)

    def evaluate_swap_cost(self, elem1: int, elem2: int, sol: BaseSolution) -> float:
        """
        Fast O(L) delta-evaluation for a swap.
        """
        loc1 = sol._find_node(elem1)
        loc2 = sol._find_node(elem2)

        if loc1[0] is None or loc2[0] is None:
            return 0.0  # Invalid move, no cost change

        r1_idx, n1_idx = loc1
        r2_idx, n2_idx = loc2

        if r1_idx == r2_idx:
            # --- Intra-route swap ---
            route = sol[r1_idx]
            old_cost, _ = self._get_route_cost_and_feasibility(route, r1_idx)
            
            new_route = route[:]
            new_route[n1_idx], new_route[n2_idx] = new_route[n2_idx], new_route[n1_idx]
            
            new_cost, _ = self._get_route_cost_and_feasibility(new_route, r1_idx)
            return new_cost - old_cost
            
        else:
            # --- Inter-route swap ---
            route1 = sol[r1_idx]
            route2 = sol[r2_idx]
            old_cost1, _ = self._get_route_cost_and_feasibility(route1, r1_idx)
            old_cost2, _ = self._get_route_cost_and_feasibility(route2, r2_idx)
            old_total = old_cost1 + old_cost2

            new_route1, new_route2 = route1[:], route2[:]
            new_route1[n1_idx], new_route2[n2_idx] = new_route2[n2_idx], new_route1[n1_idx]
            
            new_cost1, _ = self._get_route_cost_and_feasibility(new_route1, r1_idx)
            new_cost2, _ = self._get_route_cost_and_feasibility(new_route2, r2_idx)
            new_total = new_cost1 + new_cost2
            
            return new_total - old_total

    def evaluate_relocation_cost(
        self, elem_to_move: int, elem_new_neighbor: int, sol: BaseSolution
    ) -> float:
        """
        Fast O(L) delta-evaluation for a relocate.
        """
        loc_move = sol._find_node(elem_to_move)
        loc_neighbor = sol._find_node(elem_new_neighbor)

        if loc_move[0] is None or loc_neighbor[0] is None:
            return 0.0  # Invalid move

        r_move_idx, n_move_idx = loc_move
        r_neighbor_idx, n_neighbor_idx = loc_neighbor

        if r_move_idx == r_neighbor_idx:
            # --- Intra-route relocate ---
            route = sol[r_move_idx]
            old_cost, _ = self._get_route_cost_and_feasibility(route, r_move_idx)
            
            new_route = route[:]
            node = new_route.pop(n_move_idx)
            
            # Re-find index, as it may have shifted after pop
            new_neighbor_idx = new_route.index(elem_new_neighbor)
            new_route.insert(new_neighbor_idx + 1, node)

            new_cost, _ = self._get_route_cost_and_feasibility(new_route, r_move_idx)
            return new_cost - old_cost
        else:
            # --- Inter-route relocate ---
            route_from = sol[r_move_idx]
            route_to = sol[r_neighbor_idx]
            old_cost_from, _ = self._get_route_cost_and_feasibility(route_from, r_move_idx)
            old_cost_to, _ = self._get_route_cost_and_feasibility(route_to, r_neighbor_idx)
            old_total = old_cost_from + old_cost_to
            
            new_route_from, new_route_to = route_from[:], route_to[:]
            node = new_route_from.pop(n_move_idx)
            new_route_to.insert(n_neighbor_idx + 1, node)
            
            new_cost_from, _ = self._get_route_cost_and_feasibility(new_route_from, r_move_idx)
            new_cost_to, _ = self._get_route_cost_and_feasibility(new_route_to, r_neighbor_idx)
            new_total = new_cost_from + new_cost_to

            return new_total - old_total

    def evaluate_insertion_cost(
        self, elem_to_insert: int, elem_new_neighbor: int, sol: BaseSolution
    ) -> float:
        """
        Fast O(L) delta-evaluation for an insert.
        """
        loc_neighbor = sol._find_node(elem_new_neighbor)

        if loc_neighbor[0] is None:
            # Fallback: if neighbor not found (e.g., empty route [0,0]),
            # just try to insert in the first route.
            r_idx = 0
            n_idx = 0
        else:
            r_idx, n_idx = loc_neighbor

        route = sol[r_idx]
        old_cost, _ = self._get_route_cost_and_feasibility(route, r_idx)
        
        new_route = route[:]
        new_route.insert(n_idx + 1, elem_to_insert)
        
        new_cost, _ = self._get_route_cost_and_feasibility(new_route, r_idx)
        return new_cost - old_cost

    def evaluate_exchange_cost(
        self, elem_in: int, elem_out: int, sol: BaseSolution
    ) -> float:
        """
        Fast O(L) delta-evaluation for an exchange (elem_in replaces elem_out).
        """
        loc_out = sol._find_node(elem_out)

        if loc_out[0] is None:
            return 0.0  # elem_out not in solution, invalid move
            
        r_idx, n_idx = loc_out
        route = sol[r_idx]
        old_cost, _ = self._get_route_cost_and_feasibility(route, r_idx)
        
        new_route = route[:]
        new_route[n_idx] = elem_in  # Replace elem_out with elem_in
        
        new_cost, _ = self._get_route_cost_and_feasibility(new_route, r_idx)
        return new_cost - old_cost

    def evaluate_removal_cost(self, elem: int, sol: BaseSolution) -> float:
        """
        Fast O(L) delta-evaluation for a removal.
        """
        loc = sol._find_node(elem)

        if loc[0] is None:
            return 0.0  # elem not in solution, no cost change
            
        r_idx, n_idx = loc
        route = sol[r_idx]
        old_cost, _ = self._get_route_cost_and_feasibility(route, r_idx)
        
        new_route = route[:]
        new_route.pop(n_idx)
        
        new_cost, _ = self._get_route_cost_and_feasibility(new_route, r_idx)
        return new_cost - old_cost
    
    def evaluate_insert_use_cost(
        self, client: int, vehicle_index: int, sol: BaseSolution
    ) -> float:
        """
        Fast O(L) delta-evaluation for an 'insert_use' move.
        Calculates the cost of a new route [0, client, 0] and
        subtracts the penalty for not visiting that client.
        """
        
        # 1. Define the new route for the previously unused vehicle
        new_route = [0, client, 0]
        
        # 2. Calculate the cost of this new route
        # Your _get_route_cost_and_feasibility function is perfect for this.
        # It correctly calculates fixed_cost + variable_cost
        # and checks all time window and capacity constraints.
        (new_route_cost, is_feasible) = self._get_route_cost_and_feasibility(
            new_route, vehicle_index
        )
        
        # 3. If the move is infeasible (e.g., client can't be
        # served by this vehicle within TWs), return a high cost.
        if not is_feasible:
            return self._get_infeasible_cost() # Or float('inf')
            
        # 4. Get the cost that is being *removed*.
        # The cost of the old route [0, 0] is 0.0 (per your function).
        # The main cost being removed is the "Big M" penalty
        # for not visiting this client.
        try:
            # This is the "Big M" penalty cost you save.
            penalty_saved = self.problem.M 
        except AttributeError:
            raise AttributeError(
                "Your problem instance (self.problem) must have an attribute "
                "'M' that defines the 'Big M' penalty for an unvisited client."
            )

        # 5. Calculate the delta cost.
        # delta = (cost_added) - (cost_removed)
        # cost_added = new_route_cost
        # cost_removed = penalty_saved
        delta_cost = new_route_cost - penalty_saved
        
        return delta_cost
