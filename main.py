from solver.hffvrptw import HFFVRPTWSolution, HFFVRPTWProblem, HFFVRPTWEvaluator, HFFVRPTWConstructiveHeuristic
from solver.metaheuristics.ts import ts_tenure5, ts_tenure0
from solver.metaheuristics.alns import alns_greedy_lns, alns_adaptive_sa
from settings import INSTANCES
import os
import time
import multiprocessing
from functools import partial

def run_solver_for_instance(instance_path: str, solver_name: str = 'ts_tenure5'):
    """
    Run the solver for a single instance and log results.
    
    Args:
        instance_path: Path to the instance file (e.g., "100_customers/C1_1_01.csv")
        solver_name: Name of the solver to use ('ts_tenure5', 'ts_tenure0', 'alns_greedy_lns')
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
        solver = ts_tenure5
    # Create log directories
    execution_dir = "logs/execution"
    results_dir = "logs/results"
    os.makedirs(execution_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get instance name without extension
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    
    # Create file paths
    execution_file = os.path.join(execution_dir, f"{instance_name}.txt")
    results_file = os.path.join(results_dir, f"{instance_name}.txt")
    
    print(f"\n{'='*80}")
    print(f"Processing instance: {instance_path}")
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
        exec_log.write(f"{best_cost}\n")
        exec_log.write(f"{list(best_solution)}\n")
        exec_log.flush()
        
        # Run the solver with custom callback to log progress
        print(f"\nRunning solver for {instance_name}...\n")
        
        # Store original solve method
        original_update = solver._update_solution
        
        def logged_update(candidate_sol):
            """Wrapper to log each improvement"""
            nonlocal best_cost, best_solution, best_time
            
            # Call original update
            original_update(candidate_sol)
            
            # Check if this is a new best
            if solver.best_solution.cost < best_cost:
                best_cost = solver.best_solution.cost
                best_solution = solver.best_solution.copy()
                best_time = time.time() - start_time
                
                # Log to execution file: cost and elements
                exec_log.write(f"{best_cost}\n")
                exec_log.write(f"{list(best_solution)}\n")
                exec_log.flush()
                
                print(f"New best solution found: {best_cost:.2f} (at {best_time:.2f}s)")
        
        # Temporarily replace the update method
        solver._update_solution = logged_update
        
        try:
            final_solution = solver.solve(problem, evaluator, initial_sol)
        finally:
            # Restore original method
            solver._update_solution = original_update
        
        total_time = time.time() - start_time
    
    # Write results file (3 lines: cost, elements, time)
    with open(results_file, 'w') as res_log:
        res_log.write(f"{best_cost}\n")
        # HFFVRPTWSolution is a list of routes, so we write it directly
        res_log.write(f"{list(best_solution)}\n")
        res_log.write(f"{best_time}\n")
    
    print(f"\n{'='*80}")
    print(f"Completed: {instance_name}")
    print(f"Best cost: {best_cost:.2f}")
    print(f"Found at: {best_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Execution log: {execution_file}")
    print(f"Results log: {results_file}")
    print(f"{'='*80}\n")
    
    return {
        'instance': instance_name,
        'best_cost': best_cost,
        'best_time': best_time,
        'total_time': total_time
    }

def run_batch(num_workers: int | None = None):
    """
    Run solver for all instances in the list using parallel processing.
    
    Args:
        num_workers: Number of parallel workers. If None, uses CPU count - 1.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"\n{'#'*80}")
    print(f"Starting batch processing of {len(INSTANCES)} instances")
    print(f"Using {num_workers} parallel workers")
    print(f"{'#'*80}\n")
    
    # Select solver
    solver_name = 'alns__adaptive_sa'
    
    # Create partial function with solver_name bound
    process_instance = partial(run_solver_for_instance, solver_name=solver_name)
    
    # Start timer for overall batch
    batch_start_time = time.time()
    
    # Run instances in parallel
    results_summary = []
    completed = 0
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance and progress tracking
        for result in pool.imap_unordered(process_instance, INSTANCES):
            completed += 1
            results_summary.append(result)
            
            # Print progress
            if 'error' not in result:
                print(f"[{completed}/{len(INSTANCES)}] Completed: {result['instance']} - Cost: {result['best_cost']:.2f}")
            else:
                print(f"[{completed}/{len(INSTANCES)}] ERROR: {result['instance']} - {result['error']}")
    
    batch_total_time = time.time() - batch_start_time
    
    # Sort results by instance name for consistent display
    results_summary.sort(key=lambda x: x['instance'])
    
    # Print summary
    print(f"\n{'#'*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Total batch time: {batch_total_time:.2f}s")
    print(f"{'#'*80}\n")
    print(f"{'Instance':<30} {'Best Cost':<15} {'Found At (s)':<15} {'Total Time (s)':<15}")
    print(f"{'-'*80}")
    
    for result in results_summary:
        if 'error' in result:
            print(f"{result['instance']:<30} ERROR: {result['error']}")
        else:
            print(f"{result['instance']:<30} {result['best_cost']:<15.2f} {result['best_time']:<15.2f} {result['total_time']:<15.2f}")
    
    print(f"\n{'#'*80}\n")

if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    multiprocessing.set_start_method('spawn', force=True)
    run_batch()
