# Route Orchestrator

A high-performance optimization suite for solving the **Heterogeneous Fixed Fleet Vehicle Routing Problem with Time Windows (HFFVRPTW)** using state-of-the-art metaheuristic algorithms.

## Overview

Route Orchestrator implements a comprehensive metaheuristic optimization framework featuring **Adaptive Large Neighborhood Search (ALNS)** and **Tabu Search (TS)** algorithms, with intelligent parallel batch processing to efficiently solve complex vehicle routing problems with heterogeneous fleets and strict time window constraints.

**Key Features:**

- **Four Metaheuristic Variants** - ALNS with Simulated Annealing, ALNS with Greedy LNS, Tabu Search with Tenure 5, and Tabu Search with Tenure 0
- **Parallel Batch Processing** - Multi-core CPU utilization with intelligent task distribution and process-safe state management
- **Advanced ALNS Operators** - Shaw (relatedness-based) and Worst (cost-based) destroy operators, Greedy and Regret-k repair operators
- **Fleet-Aware Routing** - Support for heterogeneous vehicle fleets with varying capacities and costs
- **Time Window Constraints** - Strict adherence to customer service time windows
- **Comprehensive Logging** - Detailed execution tracking with solution evolution and fleet-grouped route output
- **Scalable Architecture** - Handles instances from 100 to 1000+ customers

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Structure](#data-structure)
- [Logging System](#logging-system)
- [Parallel Processing](#parallel-processing)
- [Solver Configuration](#solver-configuration)
- [Advanced Usage](#advanced-usage)
- [Algorithm Details](#algorithm-details)
- [Performance](#performance)

---

## Features

### Optimization Algorithms

**Adaptive Large Neighborhood Search (ALNS)**

- **Two ALNS variants**:
  - `alns_adaptive_sa` - Adaptive operator selection with Simulated Annealing acceptance
  - `alns_greedy_lns` - Greedy acceptance (only improving solutions)
- **Five destroy operators**:
  - RandomDestroy (5 and 10 customers) - Random removal
  - RouteDestroy - Removes entire route
  - ShawDestroy - Relatedness-based removal (distance + time window similarity)
  - WorstDestroy - Cost-based removal (removes expensive customers)
- **Three repair operators**:
  - GreedyRepair - Inserts at cheapest feasible position
  - RegretRepair (k=3) - Prioritizes customers with high regret
  - RegretRepair (k=5) - Higher-order regret calculation
- **Adaptive operator selection** - Roulette Wheel mechanism with dynamic weight management
- **Simulated Annealing acceptance** - Temperature-based solution acceptance

**Tabu Search (TS)**

- **Two TS variants**:
  - `ts_tenure5` - Tabu tenure of 5 iterations
  - `ts_tenure0` - No tabu restrictions (greedy search)
- **Comprehensive neighborhood exploration**:
  - Swap - Exchange two customer positions
  - Relocate - Move customer to new position
  - Insert - Add unvisited customer to route
  - Exchange - Swap visited ↔ unvisited customers
  - Insert-use - Activate unused vehicle (diversification)
- **Efficient delta evaluation** - Fast move cost estimation
- **Diversification support** - Automatic vehicle activation when needed

### System Capabilities

- **Intelligent Result Caching** - Automatically skips instances with existing solutions to save computation time
- **Real-time Progress Tracking** - Live updates during batch processing with completion counters
- **Fleet-Grouped Output** - Routes organized by vehicle type for easy analysis
- **Unused Vehicle Suppression** - Clean output showing only active routes (excludes [0, 0] routes)
- **Centralized Configuration** - Single `settings.py` file for time limits and instance selection
- **Multiprocessing-Safe State** - Proper method binding preservation for parallel execution
- **Comprehensive Error Handling** - Graceful handling of infeasible instances with detailed warnings

---

## Project Structure

```
route-orchestrator/
├── main.py                      # Main batch processing engine
├── settings.py                  # Configuration (TIME_LIMIT, instances)
├── README.md                    # Documentation
│
├── data/
│   ├── instances/              # Problem instance files (CSV format)
│   │   ├── 100_customers/      # 56 instances with 100 customers
│   │   ├── 400_customers/      # 60 instances with 400 customers
│   │   ├── 800_customers/      # 60 instances with 800 customers
│   │   └── 1000_customers/     # 60 instances with 1000 customers
│   ├── fleets/                 # Fleet configuration files (JSON)
│   │   ├── C1.json             # Type C1 fleet configuration
│   │   ├── C2.json             # Type C2 fleet configuration
│   │   ├── R1.json             # Type R1 fleet configuration
│   │   ├── R2.json             # Type R2 fleet configuration
│   │   ├── RC1.json            # Type RC1 fleet configuration
│   │   └── RC2.json            # Type RC2 fleet configuration
│   └── raw_instances/          # Original Solomon benchmark data
│
├── logs/                        # Algorithm execution logs (auto-created)
│   ├── alns_adaptive_sa/       # ALNS with SA acceptance
│   │   ├── execution/          # Improvement trajectory logs
│   │   └── results/            # Final solution logs
│   ├── alns_greedy_lns/        # ALNS with greedy acceptance
│   │   ├── execution/          # Improvement trajectory logs
│   │   └── results/            # Final solution logs
│   ├── ts_tenure5/             # Tabu Search (tenure 5)
│   │   ├── execution/          # Improvement trajectory logs
│   │   └── results/            # Final solution logs
│   └── ts_tenure0/             # Tabu Search (tenure 0)
│       ├── execution/          # Improvement trajectory logs
│       └── results/            # Final solution logs
│
├── solver/
│   ├── __init__.py
│   ├── hffvrptw.py            # Core solver classes and imports
│   ├── metaheuristics/
│   │   ├── __init__.py
│   │   ├── alns.py            # ALNS implementation with operators
│   │   └── ts.py              # Tabu Search implementation
│   └── problem/
│       ├── hffvrptw_problem_instance.py  # Problem data structure
│       ├── hffvrptw_solution.py          # Solution representation
│       ├── hffvrptw_evaluator.py         # Objective and constraint evaluation
│       ├── hffvrptw_initial_solution.py  # Constructive heuristic
│       └── model/
│           ├── __init__.py
│           ├── constraints.py            # Time window and capacity constraints
│           └── objective_function.py     # Cost calculation
│
└── utils/
    ├── __init__.py
    ├── adj_matrix.py          # Distance matrix utilities
    ├── capture_output.py      # Output redirection utilities
    ├── data_loader.py         # High-level data loading
    ├── instance_reader.py     # CSV parsing and fleet matching
    └── results_logger.py      # Result output formatting
```

---

## Installation

### Prerequisites

- **Python 3.10+** (tested with Python 3.14)
- **pip** package manager
- **Virtual environment** (recommended)

### Setup

1. Clone the repository

```bash
git clone <repository-url>
cd route-orchestrator
```

2. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `numpy` - Numerical computations
- `np-solver` - Metaheuristic framework
- Standard library: `multiprocessing`, `time`, `json`, `csv`

---

## Configuration

All configuration is centralized in `settings.py`:

### Time Limit

```python
TIME_LIMIT = 600  # Solver time limit in seconds (default: 10 minutes)
```

### Instance Selection

```python
INSTANCES = [
    "100_customers/C1_1_01.csv",
    "400_customers/C1_4_1.csv",
    # ... add instances to process
]
```

### Path Configuration

```python
FLEETS_PATH = "data/fleets"
INSTANCES_PATH = "data/instances"
RESULTS_PATH = "data/results"
```

---

## Usage

### Running Batch Processing

Process all configured instances with both metaheuristics in parallel:

```bash
source .venv/bin/activate
python main.py
```

**Behavior:**

- Processes each instance with all four solvers: `alns_adaptive_sa`, `alns_greedy_lns`, `ts_tenure5`, `ts_tenure0`
- Uses `CPU_COUNT - 1` workers by default (configurable)
- Automatically skips instances with existing result files
- Creates separate log directories for each solver variant
- Progress tracking shows `[completed/total]` with instance name and solver

**Example output:**

```
[1/96] Completed: C1_1_01 (alns_adaptive_sa) - Cost: 2244.29
[2/96] Skipped: C1_1_01 (alns_greedy_lns) - results already exist
[3/96] Completed: C1_1_01 (ts_tenure5) - Cost: 2391.49
```

### Running Individual Instances

```python
from main import run_solver_for_instance

# Run ALNS on a single instance
result = run_solver_for_instance(
    instance_path="100_customers/C1_1_01.csv",
    solver_name="alns_adaptive_sa"
)

print(f"Best cost: {result['best_cost']:.2f}")
print(f"Time found: {result['best_time']:.2f}s")
print(f"Total time: {result['total_time']:.2f}s")
```

**Available solvers:**

- `alns_adaptive_sa` - ALNS with Simulated Annealing and adaptive operator weights
- `alns_greedy_lns` - ALNS with greedy acceptance (only improving solutions)
- `ts_tenure5` - Tabu Search with tabu tenure of 5 iterations
- `ts_tenure0` - Tabu Search with no tabu restrictions (greedy search)

### Custom Batch with Specific Instances

```python
from main import run_batch
from settings import INSTANCES

# Override INSTANCES temporarily
import settings
settings.INSTANCES = [
    "100_customers/C1_1_01.csv",
    "100_customers/C1_1_02.csv",
]

# Run with custom worker count
run_batch(num_workers=4)
```

---

## Data Structure

### Instance Files

CSV format with customer and depot information:

```csv
CUST_NO, XCOORD, YCOORD, DEMAND, READY_TIME, DUE_DATE, SERVICE_TIME
0,40,50,0,0,230,0
1,45,68,10,0,70,10
```

- Row 0: Depot information
- Rows 1-N: Customer information
- Euclidean coordinates for distance
- Time windows: `[READY_TIME, DUE_DATE]`

### Fleet Configuration Files

JSON format defining available vehicle types:

```json
{
  "fleet_name": "C1",
  "vehicles": [
    {
      "type": "A",
      "fixed_cost": 0.0,
      "variable_cost": 1.0,
      "capacity": 200,
      "number_of_vehicles": 25
    }
  ]
}
```

---

## Logging System

### Log Organization

Four solver variants, each with separate execution and results logs:

```
logs/
├── alns_adaptive_sa/       # ALNS with Simulated Annealing
│   ├── execution/          # Improvement trajectory for each instance
│   └── results/            # Final best solutions
├── alns_greedy_lns/        # ALNS with greedy acceptance
│   ├── execution/
│   └── results/
├── ts_tenure5/             # Tabu Search with tenure 5
│   ├── execution/
│   └── results/
└── ts_tenure0/             # Tabu Search with tenure 0 (greedy)
    ├── execution/
    └── results/
```

### Log Format

**Execution Logs** (`logs/{solver}/execution/{instance}.txt`)

Tracks the complete improvement trajectory with timestamps:

```
[3763.29, 0.00]
A: [[0, 43, 42, 71, 92, 48, 51, 66, 0]]
B: [[0, 67, 65, 41, 53, 60, 64, 68, 69, 0]]
[2891.49, 15.32]
A: [[0, 43, 42, 40, 44, 45, 48, 51, 0]]
B: [[0, 67, 65, 41, 53, 60, 64, 68, 69, 0], [0, 71, 92, 48, 51, 66, 0]]
[2384.79, 45.67]
A: [[0, 43, 42, 40, 44, 45, 48, 51, 66, 0]]
B: [[0, 67, 65, 41, 0], [0, 71, 92, 48, 51, 0]]
```

- First line of each block: `[cost, timestamp_seconds]`
- Following lines: Fleet-grouped routes (e.g., `A:`, `B:`, `C:`)
- Only shows used vehicles (excludes empty `[0, 0]` routes)
- New block logged each time a better solution is found

**Results Logs** (`logs/{solver}/results/{instance}.txt`)

Stores the final best solution:

```
[2384.79, 45.67]
A: [[0, 43, 42, 40, 44, 45, 48, 51, 66, 0]]
B: [[0, 67, 65, 41, 0], [0, 71, 92, 48, 51, 0]]
```

- Line 1: `[best_cost, time_found_seconds]`
- Lines 2+: Fleet-grouped routes of best solution

### Parsing Logs

```python
import ast

with open('logs/alns/results/C1_1_01.txt', 'r') as f:
    best_cost, best_time = ast.literal_eval(f.readline().strip())
    
    fleet_routes = {}
    for line in f:
        fleet_type, routes = line.strip().split(': ', 1)
        fleet_routes[fleet_type] = ast.literal_eval(routes)
```

---

## Parallel Processing

### Architecture

Route Orchestrator uses Python's `multiprocessing` module with process-safe state management:

```python
# Spawn method ensures clean process state
multiprocessing.set_start_method('spawn', force=True)

num_workers = multiprocessing.cpu_count() - 1

# Module-level wrapper for pickling compatibility
def _process_task_wrapper(task):
    instance_path, solver_name = task
    return run_solver_for_instance(instance_path, solver_name)

with multiprocessing.Pool(processes=num_workers) as pool:
    for result in pool.imap_unordered(_process_task_wrapper, all_tasks):
        # Process results as they complete
        results_summary.append(result)
```

**Key implementation details:**

- **Spawn method** - Ensures clean process state (critical for solver instances)
- **Method binding preservation** - Proper handling of bound methods in monkey-patching
- **Module-level wrappers** - Required for pickling tasks across processes
- **Unordered execution** - Results processed as they complete for better progress tracking

### Task Distribution

Each instance generates 4 tasks (one per solver variant):

```
24 instances × 4 solvers = 96 total tasks

Example task distribution (8-core system, 7 workers):
- Worker 1: C1_1_01 + alns_adaptive_sa
- Worker 2: C1_1_01 + alns_greedy_lns  
- Worker 3: C1_1_01 + ts_tenure5
- Worker 4: C1_1_01 + ts_tenure0
- Worker 5: C1_1_02 + alns_adaptive_sa
- Worker 6: C1_1_02 + alns_greedy_lns
- Worker 7: C1_1_02 + ts_tenure5
... (tasks redistributed as workers complete)
```

### Performance Characteristics

**8-core system example:**

- 24 instances × 4 solvers = 96 tasks
- 7 parallel workers
- ~600s per task (10-minute time limit)
- Total wall time: ~96 / 7 × 600s ≈ 137 minutes
- Speedup: ~7x compared to sequential execution

---

## Solver Configuration

### ALNS Configuration

Located in `solver/metaheuristics/alns.py`:

**Model 1: Adaptive SA (alns_adaptive_sa)**

```python
alns_adaptive_sa = ALNS(
    destroy_operators=[
        RandomDestroy(num_to_remove=5),
        RandomDestroy(num_to_remove=10),
        RouteDestroy(),
        ShawDestroy(num_to_remove=8, determinism_param=4),
        WorstDestroy(num_to_remove=6, determinism_param=3)
    ],
    repair_operators=[
        GreedyRepair(),
        RegretRepair(k_regret=3),
        RegretRepair(k_regret=5)
    ],
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
    time_limit=TIME_LIMIT,  # 600 seconds
    max_iterations=10000
)
```

**Model 2: Greedy LNS (alns_greedy_lns)**

```python
alns_greedy_lns = ALNS(
    destroy_operators=[RandomDestroy(num_to_remove=8)],
    repair_operators=[GreedyRepair()],
    weight_manager=RouletteWheelManager(
        segment_size=1, 
        decay=1.0  # No decay
    ),
    acceptance_criteria=SimulatedAnnealingAcceptance(
        initial_temp=0,    # Only accept improving solutions
        cooling_rate=1.0, 
        min_temp=0
    ),
    time_limit=TIME_LIMIT,
    max_iterations=10000
)
```

### Tabu Search Configuration

Located in `solver/metaheuristics/ts.py`:

**Model 1: Tenure 5 (ts_tenure5)**

```python
ts_tenure5 = TabuSearch(
    tenure=5,
    neighborhood_strategy=HFFVRPTW_TSNeighborhood(),
    time_limit=TIME_LIMIT
)
```

**Model 2: Tenure 0 (ts_tenure0)**

```python
ts_tenure0 = TabuSearch(
    tenure=0,  # No tabu restrictions (greedy)
    neighborhood_strategy=HFFVRPTW_TSNeighborhood(),
    time_limit=TIME_LIMIT
)
```

**Neighborhood moves:**

- `swap` - Exchange two customer positions within routes
- `relocate` - Move customer to different position
- `insert` - Add unvisited customer to route  
- `exchange` - Swap visited ↔ unvisited customers
- `insert_use` - Activate unused vehicle for diversification

---

## Advanced Usage

### Testing and Validation

The project includes comprehensive test suites:

```bash
# Run full validation suite
source .venv/bin/activate
python3.14 test_corrections.py

# Run single instance test
python3.14 test_single_run.py
```

**Test coverage:**

- GreedyRepair operator correctness
- RegretRepair operator correctness  
- Multiprocessing state preservation
- End-to-end solver execution

### Analyzing Solution Quality

```python
import os
import ast
from collections import defaultdict

def analyze_results(solver='alns_adaptive_sa'):
    """Aggregate results by instance type"""
    results_dir = f'logs/{solver}/results'
    costs = defaultdict(list)
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath) as f:
                cost, _ = ast.literal_eval(f.readline().strip())
                instance_type = filename.split('_')[0]  # C1, R1, RC1, etc.
                costs[instance_type].append(cost)
    
    for instance_type, cost_list in sorted(costs.items()):
        avg = sum(cost_list) / len(cost_list)
        min_cost = min(cost_list)
        max_cost = max(cost_list)
        print(f"{instance_type}: avg={avg:.2f}, min={min_cost:.2f}, max={max_cost:.2f} (n={len(cost_list)})")

# Usage
analyze_results('alns_adaptive_sa')
analyze_results('alns_greedy_lns')
```

### Comparing Solver Performance

```python
def compare_solvers(instance_name):
    """Compare all four solvers on a single instance"""
    solvers = ['alns_adaptive_sa', 'alns_greedy_lns', 'ts_tenure5', 'ts_tenure0']
    results = {}
    
    for solver in solvers:
        filepath = f'logs/{solver}/results/{instance_name}.txt'
        with open(filepath) as f:
            cost, time = ast.literal_eval(f.readline().strip())
            results[solver] = {'cost': cost, 'time': time}
    
    print(f"\nInstance: {instance_name}\n")
    for solver, data in sorted(results.items(), key=lambda x: x[1]['cost']):
        print(f"{solver:20s}: cost={data['cost']:8.2f}, found_at={data['time']:6.2f}s")
    
    best_solver = min(results.items(), key=lambda x: x[1]['cost'])
    print(f"\nBest: {best_solver[0]} with cost {best_solver[1]['cost']:.2f}")

# Usage
compare_solvers('C1_1_01')
```

### Tracking Solution Evolution

```python
def plot_convergence(instance_name, solver):
    """Extract and plot improvement trajectory"""
    import matplotlib.pyplot as plt
    
    filepath = f'logs/{solver}/execution/{instance_name}.txt'
    
    costs = []
    times = []
    
    with open(filepath) as f:
        for line in f:
            if line.startswith('['):
                cost, time = ast.literal_eval(line.strip())
                costs.append(cost)
                times.append(time)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, costs, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Solution Cost')
    plt.title(f'{instance_name} - {solver}\nConvergence Trajectory')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'convergence_{solver}_{instance_name}.png', dpi=300)
    plt.show()

# Usage
plot_convergence('C1_1_01', 'alns_adaptive_sa')
```

### Custom ALNS Configuration

```python
from solver.metaheuristics.alns import (
    ALNS, RandomDestroy, ShawDestroy, GreedyRepair, RegretRepair,
    RouletteWheelManager, SimulatedAnnealingAcceptance
)
from settings import TIME_LIMIT

# Create custom ALNS variant
custom_alns = ALNS(
    destroy_operators=[
        RandomDestroy(num_to_remove=15),  # More aggressive removal
        ShawDestroy(num_to_remove=12, determinism_param=2),  # More random
    ],
    repair_operators=[
        RegretRepair(k_regret=4),  # Different regret level
    ],
    weight_manager=RouletteWheelManager(
        segment_size=50,   # Faster adaptation
        decay=0.9,         # Higher decay
        reward_points={
            "new_best": 15,
            "better_than_current": 7,
            "accepted": 3,
            "rejected": 0
        }
    ),
    acceptance_criteria=SimulatedAnnealingAcceptance(
        initial_temp=10000,  # Higher initial temperature
        cooling_rate=0.995,   # Slower cooling
        min_temp=0.01
    ),
    time_limit=TIME_LIMIT,
    max_iterations=10000
)

# Use in run_solver_for_instance
result = run_solver_for_instance(
    instance_path="100_customers/C1_1_01.csv",
    solver_name="custom_alns"  # Will need to add to solver_name mapping
)
```

---

## Technical Implementation Notes

### Critical Corrections Implemented

The codebase includes several important correctness fixes:

**1. Repair Operator Logic**

Both `GreedyRepair` and `RegretRepair` now correctly evaluate **all** move types (`insert` and `insert_use`) before making decisions, rather than arbitrarily prioritizing existing routes. This ensures optimal moves are selected based on actual cost, not move type.

**2. Multiprocessing State Management**

The `main.py` monkey-patching preserves bound methods correctly to avoid `TypeError` in subsequent worker tasks. This ensures stable parallel execution across all worker processes.

**3. Operator Selection Strategy**

ALNS operators now properly compare opening new routes versus inserting into existing routes, leading to better solution quality through more informed decisions.

See `CORREÇÕES_IMPLEMENTADAS.md` for detailed technical documentation.

---

## Algorithm Details

### ALNS Destroy Operators

**1. RandomDestroy(num_to_remove)**

Randomly selects and removes `num_to_remove` customers from the solution.

- Simple diversification mechanism
- Used with both 5 and 10 customers in `alns_adaptive_sa`
- Fast execution, good for exploration

**2. RouteDestroy()**

Selects one active route randomly and removes all its customers.

- Preserves spatial structure within removed customers
- Good for route-level reorganization
- Balances solution exploration

**3. ShawDestroy(num_to_remove, determinism_param)**

Removes customers based on **relatedness** (Shaw, 1998):

```python
relatedness = weight_distance × norm_distance + 
              weight_time × (1 - norm_time_overlap)
```

- Selects seed customer randomly
- Iteratively removes most related customers
- `determinism_param` controls greediness (higher = more deterministic)
- Effective for customers with similar time windows or locations

**4. WorstDestroy(num_to_remove, determinism_param)**

Removes customers that contribute most to solution cost:

```python
removal_cost = distance(prev, customer) + distance(customer, next) 
               - distance(prev, next)
```

- Targets "expensive" routing decisions
- Greedy improvement potential
- `determinism_param` controls selection randomness

### ALNS Repair Operators

**1. GreedyRepair()**

Inserts each unvisited customer at its cheapest feasible position:

- Evaluates **both** `insert` (existing routes) and `insert_use` (new routes)
- Selects move with lowest cost (important fix: no arbitrary prioritization)
- Fast, effective baseline repair strategy

**2. RegretRepair(k_regret)**

Inserts customers based on **regret** value:

```python
regret = sum(cost[i] - cost[0] for i in range(1, k_regret))
```

- Prioritizes customers with few good insertion options
- `k_regret=3` uses difference between best and 3rd-best positions
- `k_regret=5` uses more positions for higher discrimination
- Leads to better global solutions by avoiding "easy" greedy choices

### Tabu Search Neighborhood

**Intensification Moves:**

- **Swap**: Exchange positions of two customers in routes
- **Relocate**: Move customer to different position (same or different route)

**Diversification Moves:**

- **Insert**: Add unvisited customer to an existing route
- **Exchange**: Swap a visited customer with an unvisited one
- **Insert-use**: Place customer in an unused vehicle (activates new route)

**Delta Evaluation:**

Fast cost estimation without full solution re-evaluation:

```python
delta_cost = new_cost - old_cost
if delta_cost < 0 or not is_tabu:
    apply_move()
```

---

## Performance

### Benchmark Results

Tested on Solomon HFFVRPTW instances (100-1000 customers):

**Typical improvements from initial solution:**

- **alns_adaptive_sa**: 35-45% cost reduction
- **alns_greedy_lns**: 40-50% cost reduction  
- **ts_tenure5**: 30-40% cost reduction
- **ts_tenure0**: 35-45% cost reduction

**Convergence patterns:**

- Most improvements occur in first 2-3 minutes
- Diminishing returns after 5 minutes
- 10-minute time limit provides good balance

### Scalability

**Instance size vs execution time:**

- **100 customers**: Reaches time limit, continues improving
- **400 customers**: Reaches time limit, solution quality varies
- **800 customers**: Reaches time limit, harder to find improvements
- **1000 customers**: Reaches time limit, mostly local search

**Resource usage:**

- Memory: ~50-100 MB per solver instance
- CPU: Scales linearly with number of workers
- Disk: ~10-50 KB per instance (log files)

### Solver Comparison

**ALNS Adaptive SA vs Greedy LNS:**

- Adaptive SA explores more diverse solutions (accepts worse solutions)
- Greedy LNS converges faster but may get stuck in local optima
- Adaptive SA generally finds better solutions given sufficient time

**Tabu Search Tenure 5 vs Tenure 0:**

- Tenure 5 avoids cycling, more thorough exploration
- Tenure 0 is pure greedy search, faster convergence
- Tenure 5 typically outperforms on complex instances

---

## License

MIT License - see LICENSE file for details.

---

## References

### Benchmark Instances

- **Solomon VRPTW benchmarks** - Original problem set adapted for heterogeneous fleets
- Instance types: C (clustered), R (random), RC (random-clustered)
- Sizes: 100, 400, 800, 1000 customers

### Algorithm References

**ALNS:**

- Ropke, S., & Pisinger, D. (2006). "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows." Transportation Science, 40(4), 455-472.
- Shaw, P. (1998). "Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems." CP-98, 417-431.

**Tabu Search:**

- Glover, F. (1989). "Tabu Search - Part I." ORSA Journal on Computing, 1(3), 190-206.
- Cordeau, J.-F., Laporte, G., & Mercier, A. (2001). "A Unified Tabu Search Heuristic for Vehicle Routing Problems with Time Windows." Journal of the Operational Research Society, 52(8), 928-936.

### Framework

- **np-solver** - Python metaheuristic optimization framework

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Ensure all tests pass (`python test_corrections.py`)
6. Submit a pull request

**Areas for contribution:**

- Additional destroy/repair operators
- New metaheuristic variants
- Performance optimizations
- Enhanced visualization tools
- Documentation improvements

---

## Contact

For questions, issues, or contributions:

- **Repository**: [route-orchestrator](https://github.com/code-lucasgabriel/route-orchestrator)
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

---

## Acknowledgments

- Solomon benchmark instances for standardized VRPTW testing
- np-solver framework for metaheuristic implementations
- Research community for ALNS and Tabu Search algorithms
- Contributors and testers

---

**Last Updated:** November 2025  
**Version:** 2.0.0  
**Python Version:** 3.10+ (tested with 3.14)
