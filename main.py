from solver.hffvrptw import HFFVRPTWSolution, HFFVRPTWProblem, HFFVRPTWEvaluator, HFFVRPTWConstructiveHeuristic
from solver.metaheuristics.ts import ts_tenure5, ts_tenure0
from solver.metaheuristics.alns import alns_greedy_lns, alns_adaptive_sa
from settings import INSTANCES
import os
import time
import multiprocessing
from functools import partial
from collections import defaultdict


def group_routes_by_fleet(solution, problem):
    """
    Group routes by fleet type, excluding unused vehicles.
    
    Args:
        solution: HFFVRPTWSolution with routes
        problem: HFFVRPTWProblem instance with fleet information
    
    Returns:
        dict: Fleet type -> list of routes for that fleet type
    """
    fleet_routes = defaultdict(list)
    
    # Map vehicle index to fleet type
    for idx, route in enumerate(solution):
        # Skip unused vehicles (routes that are just [0, 0])
        if route == [0, 0]:
            continue
        
        # Get the fleet type for this vehicle index
        # problem.K is a list of (fleet_type, vehicle_index_within_type) tuples
        if idx < len(problem.K):
            fleet_type, _ = problem.K[idx]
            fleet_routes[fleet_type].append(route)
    
    return dict(fleet_routes)


def run_solver_for_instance(instance_path: str, solver_name: str):
    """
    Run the solver for a single instance and log results.
    
    Args:
        instance_path: Path to the instance file (e.g., "100_customers/C1_1_01.csv")
        solver_name: Name of the solver to use ('ts_tenure5', 'ts_tenure0', 'alns_greedy_lns', 'alns_adaptive_sa')
    """
    # Import solver inside function for multiprocessing compatibility
    if solver_name == 'ts_tenure5':
        solver = ts_tenure5
    elif solver_name == 'ts_tenure0':
        solver = ts_tenure0
    elif solver_name == 'alns_greedy_lns':
        solver = alns_greedy_lns
    elif solver_name == 'alns_adaptive_sa':
        solver = alns_adaptive_sa
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
    
    # Create log directories based on solver name
    execution_dir = f"logs/{solver_name}/execution"
    results_dir = f"logs/{solver_name}/results"
    os.makedirs(execution_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get instance name without extension
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    
    # Create file paths
    execution_file = os.path.join(execution_dir, f"{instance_name}.txt")
    results_file = os.path.join(results_dir, f"{instance_name}.txt")
    
    # Check if results already exist
    if os.path.exists(results_file):
        print(f"Skipping {instance_name} ({solver_name}) - results already exist")
        return {
            'instance': instance_name,
            'solver': solver_name,
            'skipped': True
        }
    
    print(f"\n{'='*80}")
    print(f"Processing instance: {instance_path} with {solver_name}")
    print(f"{'='*80}\n")
    
    # Create and load the problem instance data
    problem = HFFVRPTWProblem()
    problem.read_instance(instance_path)
    
    # Create the evaluator and pass the problem instance data
    evaluator = HFFVRPTWEvaluator(problem)
    
    # Generate initial solution
    constructor = HFFVRPTWConstructiveHeuristic(problem, evaluator)
    initial_sol = constructor.build()
    
    print(f"Initial solution cost: {evaluator.evaluate(initial_sol)}")
    print(f"Initial solution valid: {evaluator.constraints(initial_sol)}")
    
    # Open execution log file
    with open(execution_file, 'w') as exec_log:
        # Track best solution found and when it was found
        best_cost = evaluator.evaluate(initial_sol)
        best_solution = initial_sol.copy()
        best_time = 0.0
        
        # Start timer
        start_time = time.time()
        
        # Log initial solution
        current_time = time.time() - start_time
        exec_log.write(f"[{best_cost:.2f}, {current_time:.2f}]\n")
        
        # Group routes by fleet and write them
        fleet_routes = group_routes_by_fleet(best_solution, problem)
        for fleet_type in sorted(fleet_routes.keys()):
            exec_log.write(f"{fleet_type}: {fleet_routes[fleet_type]}\n")
        exec_log.flush()
        
        # Run the solver with custom callback to log progress
        print(f"\nRunning {solver_name} for {instance_name}...")
        
        # Check solver attributes
        has_update_method = hasattr(solver, '_update_solution')
        
        # Wrap _update_solution to log improvements
        if has_update_method:
            # Track best internally
            tracked_best_cost = best_cost
            tracked_best_solution = best_solution
            
            # CORRIGIDO: Armazena o MÉTODO BOUND original, não a função unbound
            original_bound_method = solver._update_solution
            
            def logged_update(self, candidate_sol):
                """Wrapper to log each improvement"""
                nonlocal best_cost, best_solution, best_time, tracked_best_cost, tracked_best_solution
                
                # Check if this is an improvement BEFORE calling original
                if candidate_sol is not None and hasattr(candidate_sol, 'cost') and candidate_sol.cost is not None:
                    if candidate_sol.cost < tracked_best_cost:
                        # This IS an improvement!
                        tracked_best_cost = candidate_sol.cost
                        tracked_best_solution = candidate_sol.copy()
                        best_cost = tracked_best_cost
                        best_solution = tracked_best_solution
                        best_time = time.time() - start_time
                        
                        # Log to execution file
                        exec_log.write(f"[{best_cost:.2f}, {best_time:.2f}]\n")
                        fleet_routes = group_routes_by_fleet(best_solution, problem)
                        for fleet_type in sorted(fleet_routes.keys()):
                            exec_log.write(f"{fleet_type}: {fleet_routes[fleet_type]}\n")
                        exec_log.flush()
                                        
                # CORRIGIDO: Chama o MÉTODO BOUND original (ele já sabe quem é o 'self')
                original_bound_method(candidate_sol)
            
            # Bind the wrapped method to the instance
            import types
            solver._update_solution = types.MethodType(logged_update, solver)
            
            try:
                print(f"  → Calling solve()...")
                final_solution = solver.solve(problem, evaluator, initial_sol)
                print(f"  → Solve completed")
            except Exception as e:
                print(f"  → ERROR during solve: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                # CORRIGIDO: Restaura o MÉTODO BOUND original, não a função unbound
                solver._update_solution = original_bound_method
        else:
            print(f"  → No _update_solution method, running solve directly")
            try:
                final_solution = solver.solve(problem, evaluator, initial_sol)
                print(f"  → Solve completed")
            except Exception as e:
                print(f"  → ERROR during solve: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        total_time = time.time() - start_time
        
        # Ensure we have the final best solution from the solver
        if hasattr(solver, 'best_solution') and solver.best_solution is not None:
            final_cost = evaluator.evaluate(solver.best_solution)
            if final_cost < best_cost:
                best_cost = final_cost
                best_solution = solver.best_solution.copy()
                best_time = total_time  # If we missed it during iteration, mark as found at end
        
        # Also check the returned solution
        if final_solution is not None:
            final_cost = evaluator.evaluate(final_solution)
            if final_cost < best_cost:
                best_cost = final_cost
                best_solution = final_solution.copy()
                best_time = total_time
    
    # Write results file with new format
    with open(results_file, 'w') as res_log:
        # First line: [total_cost, time_found] - changed from tuple to list
        res_log.write(f"[{best_cost:.2f}, {best_time:.2f}]\n")
        
        # Group routes by fleet and write them
        fleet_routes = group_routes_by_fleet(best_solution, problem)
        for fleet_type in sorted(fleet_routes.keys()):
            res_log.write(f"{fleet_type}: {fleet_routes[fleet_type]}\n")
    
    print(f"\n{'='*80}")
    print(f"Completed: {instance_name} ({solver_name})")
    print(f"Best cost: {best_cost:.2f}")
    print(f"Found at: {best_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Execution log: {execution_file}")
    print(f"Results log: {results_file}")
    print(f"{'='*80}\n")
    
    return {
        'instance': instance_name,
        'solver': solver_name,
        'best_cost': best_cost,
        'best_time': best_time,
        'total_time': total_time,
        'skipped': False
    }


def _process_task_wrapper(task):
    """
    Wrapper function for multiprocessing. Must be at module level for pickling.
    
    Args:
        task: Tuple of (instance_path, solver_name)
    
    Returns:
        dict: Result dictionary from run_solver_for_instance
    """
    instance_path, solver_name = task
    return run_solver_for_instance(instance_path, solver_name)


def run_batch(num_workers: int | None = None):
    """
    Run solvers for all instances in the list using parallel processing.
    Runs all four metaheuristics for each instance:
    - alns_adaptive_sa
    - alns_greedy_lns
    - ts_tenure5
    - ts_tenure0
    
    Args:
        num_workers: Number of parallel workers. If None, uses CPU count - 1.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Define all solver configurations
    solver_configs = [
        'alns_adaptive_sa',
        'alns_greedy_lns',
        'ts_tenure5',
        'ts_tenure0',
    ]
    
    # Create list of all tasks (instance, solver_name)
    all_tasks = []
    for instance in INSTANCES:
        for solver_name in solver_configs:
            all_tasks.append((instance, solver_name))
    
    print(f"\n{'#'*80}")
    print(f"Starting batch processing")
    print(f"Instances: {len(INSTANCES)}")
    print(f"Solvers: {len(solver_configs)} ({', '.join(solver_configs)})")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Using {num_workers} parallel workers")
    print(f"{'#'*80}\n")
    
    # Start timer for overall batch
    batch_start_time = time.time()
    
    # Run tasks in parallel
    results_summary = []
    completed = 0
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance and progress tracking
        for result in pool.imap_unordered(_process_task_wrapper, all_tasks):
            completed += 1
            results_summary.append(result)
            
            # Print progress
            if result.get('skipped'):
                print(f"[{completed}/{len(all_tasks)}] Skipped: {result['instance']} ({result['solver']})")
            elif 'error' not in result:
                print(f"[{completed}/{len(all_tasks)}] Completed: {result['instance']} ({result['solver']}) - Cost: {result['best_cost']:.2f}")
            else:
                print(f"[{completed}/{len(all_tasks)}] ERROR: {result['instance']} ({result['solver']}) - {result['error']}")
    
    batch_total_time = time.time() - batch_start_time
    
    # Sort results by instance name and solver
    results_summary.sort(key=lambda x: (x['instance'], x['solver']))
    
    # Print summary
    print(f"\n{'#'*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Total batch time: {batch_total_time:.2f}s")
    print(f"{'#'*80}\n")
    print(f"{'Instance':<30} {'Solver':<20} {'Best Cost':<15} {'Found At (s)':<15} {'Total Time (s)':<15}")
    print(f"{'-'*100}")
    
    for result in results_summary:
        if result.get('skipped'):
            print(f"{result['instance']:<30} {result['solver']:<20} SKIPPED")
        elif 'error' in result:
            print(f"{result['instance']:<30} {result['solver']:<20} ERROR: {result['error']}")
        else:
            print(f"{result['instance']:<30} {result['solver']:<20} {result['best_cost']:<15.2f} {result['best_time']:<15.2f} {result['total_time']:<15.2f}")
    
    print(f"\n{'#'*80}\n")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    multiprocessing.set_start_method('spawn', force=True)
    run_batch()
