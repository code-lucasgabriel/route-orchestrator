# Route Orchestrator

A high-performance optimization suite for solving the **Heterogeneous Fixed Fleet Vehicle Routing Problem with Time Windows (HFFVRPTW)** using state-of-the-art metaheuristic algorithms.

## Overview

Route Orchestrator implements dual metaheuristic optimization using both **Adaptive Large Neighborhood Search (ALNS)** and **Tabu Search (TS)** algorithms, with intelligent parallel batch processing to efficiently solve complex vehicle routing problems with heterogeneous fleets and strict time window constraints.

**Key Features:**

- **Dual Metaheuristic Engine** - Simultaneous ALNS and Tabu Search optimization
- **Parallel Batch Processing** - Multi-core CPU utilization with intelligent task distribution
- **Fleet-Aware Routing** - Support for heterogeneous vehicle fleets with varying capacities and costs
- **Time Window Constraints** - Strict adherence to customer service time windows
- **Comprehensive Logging** - Detailed execution tracking and solution evolution logs
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

- Multiple destroy operators: Random, Route-based, Shaw (relatedness), Worst removal
- Multiple repair operators: Greedy insertion, Regret-based insertion (k=3,5)
- Adaptive operator selection using Roulette Wheel mechanism
- Simulated Annealing acceptance criteria
- Dynamic operator weight management

**Tabu Search (TS)**

- Comprehensive neighborhood exploration: Swap, Relocate, Insert, Exchange
- Efficient delta evaluation for move cost estimation
- Support for activating unused vehicles (diversification)
- Configurable tabu tenure (tested with tenure 0 and 5)

### System Capabilities

- **Automatic Result Caching** - Skips instances with existing solutions
- **Real-time Progress Tracking** - Live updates during batch processing
- **Fleet Grouping** - Routes organized by vehicle type in output
- **Unused Vehicle Suppression** - Clean output with only active routes
- **Configurable Time Limits** - Centralized time budget management
- **Multiprocessing-Safe** - Module-level task wrappers for pickling compatibility

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
│   ├── raw_instances/          # Original benchmark data
│   └── results/                # Legacy results storage
│
├── logs/                        # Algorithm execution logs
│   ├── alns/                   # ALNS metaheuristic logs
│   │   ├── execution/          # Improvement trajectory logs
│   │   └── results/            # Final solution logs
│   └── ts/                     # Tabu Search metaheuristic logs
│       ├── execution/          # Improvement trajectory logs
│       └── results/            # Final solution logs
│
├── solver/
│   ├── hffvrptw.py            # Core solver classes
│   ├── metaheuristics/
│   │   ├── alns.py            # ALNS implementation
│   │   └── ts.py              # Tabu Search implementation
│   └── problem/
│       ├── hffvrptw_problem_instance.py
│       ├── hffvrptw_solution.py
│       ├── hffvrptw_evaluator.py
│       └── hffvrptw_initial_solution.py
│
└── utils/
    ├── data_loader.py         # Instance and fleet loading
    ├── instance_reader.py     # CSV parsing utilities
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
- Processes each instance with both ALNS and Tabu Search
- Uses `CPU_COUNT - 1` workers by default
- Automatically skips instances with existing result files
- Creates separate logs for each metaheuristic

### Running Individual Instances

```python
from main import run_solver_for_instance

# Run ALNS on a single instance
result = run_solver_for_instance(
    instance_path="100_customers/C1_1_01.csv",
    solver_name="alns_adaptive_sa",
    metaheuristic="alns"
)

print(f"Best cost: {result['best_cost']}")
print(f"Time: {result['best_time']:.2f}s")
```

**Available solvers:**
- `alns_adaptive_sa` - ALNS with Simulated Annealing
- `alns_greedy_lns` - ALNS with greedy acceptance
- `ts_tenure5` - Tabu Search with tenure 5
- `ts_tenure0` - Tabu Search with tenure 0

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

```
logs/
├── alns/
│   ├── execution/          # Improvement trajectory
│   └── results/            # Final solutions
└── ts/
    ├── execution/          # Improvement trajectory
    └── results/            # Final solutions
```

### Log Format

**Execution Logs** (`logs/{metaheuristic}/execution/{instance}.txt`)

```
[3763.29, 0.00]
A: [[0, 43, 42, 71, 92, 48, 51, 66, 0]]
B: [[0, 67, 65, 41, 53, 60, 64, 68, 69, 0]]
[2891.49, 15.32]
A: [[0, 43, 42, 40, 44, 45, 48, 51, 0]]
...
```

- Line 1: `[cost, time]` - Cost and timestamp (2 decimal places)
- Lines 2-N: Fleet-grouped routes
- New block for each improvement

**Results Logs** (`logs/{metaheuristic}/results/{instance}.txt`)

```
[2391.49, 120.45]
A: [[0, 43, 42, 40, 44, 45, 48, 51, 66, 0]]
B: [[0, 19, 16, 12, 2, 0]]
```

- Line 1: `[best_cost, time_found]` - Best cost and discovery time
- Lines 2-N: Fleet-grouped routes (only used vehicles)

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

Uses Python's `multiprocessing` module:

```python
num_workers = multiprocessing.cpu_count() - 1

def _process_task_wrapper(task):
    instance_path, solver_name, metaheuristic = task
    return run_solver_for_instance(instance_path, solver_name, metaheuristic)

with multiprocessing.Pool(processes=num_workers) as pool:
    for result in pool.imap_unordered(_process_task_wrapper, all_tasks):
        # Process results as they complete
        ...
```

### Task Distribution

- Each instance generates 2 tasks (ALNS + TS)
- 20 instances → 40 total tasks
- Tasks distributed across CPU cores
- Results processed as they complete

### Performance

Example (8-core system):
- 20 instances × 2 metaheuristics = 40 tasks
- 7 parallel workers
- ~600s per task
- Total time: ~3500s (≈58 minutes)

---

## Solver Configuration

### ALNS

Located in `solver/metaheuristics/alns.py`:

```python
alns_adaptive_sa = ALNS(
    destroy_operators=[
        RandomDestroy(num_to_remove=5),
        RouteDestroy(),
        ShawDestroy(num_to_remove=8),
        WorstDestroy(num_to_remove=6)
    ],
    repair_operators=[
        GreedyRepair(),
        RegretRepair(k_regret=3),
        RegretRepair(k_regret=5)
    ],
    weight_manager=RouletteWheelManager(...),
    acceptance_criteria=SimulatedAnnealingAcceptance(...),
    time_limit=TIME_LIMIT,
    max_iterations=10000
)
```

### Tabu Search

Located in `solver/metaheuristics/ts.py`:

```python
ts_tenure5 = TabuSearch(
    tenure=5,
    neighborhood_strategy=HFFVRPTW_TSNeighborhood(),
    time_limit=TIME_LIMIT
)
```

**Neighborhood moves:**
- `swap` - Exchange two customer positions
- `relocate` - Move customer to new position
- `insert` - Insert unvisited customer into route
- `exchange` - Swap visited/unvisited customers
- `insert_use` - Activate unused vehicle

---

## Advanced Usage

### Analyzing Solution Quality

```python
import os, ast
from collections import defaultdict

def analyze_results(metaheuristic='alns'):
    results_dir = f'logs/{metaheuristic}/results'
    costs = defaultdict(list)
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath) as f:
                cost, _ = ast.literal_eval(f.readline().strip())
                instance_type = filename.split('_')[0]
                costs[instance_type].append(cost)
    
    for instance_type, cost_list in sorted(costs.items()):
        avg = sum(cost_list) / len(cost_list)
        print(f"{instance_type}: {avg:.2f} (n={len(cost_list)})")
```

### Comparing Metaheuristics

```python
def compare_metaheuristics(instance_name):
    alns_file = f'logs/alns/results/{instance_name}.txt'
    ts_file = f'logs/ts/results/{instance_name}.txt'
    
    with open(alns_file) as f:
        alns_cost, alns_time = ast.literal_eval(f.readline().strip())
    with open(ts_file) as f:
        ts_cost, ts_time = ast.literal_eval(f.readline().strip())
    
    print(f"Instance: {instance_name}")
    print(f"ALNS: {alns_cost:.2f} (at {alns_time:.2f}s)")
    print(f"TS:   {ts_cost:.2f} (at {ts_time:.2f}s)")
    print(f"Winner: {'ALNS' if alns_cost < ts_cost else 'TS'}")
```

---

## Algorithm Details

### ALNS Operators

**Destroy Operators:**

1. **RandomDestroy** - Randomly removes N customers
2. **RouteDestroy** - Removes all customers from random route
3. **ShawDestroy** - Removes related customers (distance + time)
4. **WorstDestroy** - Removes high-cost customers

**Repair Operators:**

1. **GreedyRepair** - Inserts at cheapest position
2. **RegretRepair** - Prioritizes high-regret customers

### Tabu Search Neighborhood

**Intensification moves:**
- Swap: Exchange two customer positions
- Relocate: Move customer to new position

**Diversification moves:**
- Insert: Add unvisited customer to route
- Exchange: Swap visited ↔ unvisited
- Insert-use: Activate unused vehicle

---

## Performance

### Scalability

- **100 customers**: 8-10 minutes per instance
- **400 customers**: 10-12 minutes (reaches time limit)
- **800 customers**: 10-12 minutes (reaches time limit)
- **1000 customers**: 10-12 minutes (reaches time limit)

**Memory usage:**
- ~50-100 MB per solver instance
- Scales linearly with instance size

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- Solomon benchmark instances for VRPTW
- np-solver framework for metaheuristic implementations
- Research papers on ALNS and Tabu Search for VRP

---

## Contact

For questions or contributions, please open an issue on the repository.
