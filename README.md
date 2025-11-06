# route-orchestrator

An optimization suite for the **Heterogeneous Fixed Fleet Vehicle Routing Problem with Time Windows (HFFVRPTW)** featuring parallel batch processing with Tabu Search and ALNS metaheuristic solvers, comprehensive logging, and centralized configuration management.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running Batch Processing](#running-batch-processing)
  - [Understanding the Output](#understanding-the-output)
  - [Logging System](#logging-system)
- [Data Structure](#data-structure)
  - [Instance Files](#instance-files)
  - [Fleet Configuration Files](#fleet-configuration-files)
- [Parallel Processing](#parallel-processing)
- [Solver Configuration](#solver-configuration)
- [Advanced Usage](#advanced-usage)

---

## Features

**Key Capabilities:**
- **Parallel Batch Processing** - Processes multiple instances simultaneously using Python multiprocessing
- **Dual Logging System** - Execution logs (all improvements) and results logs (final summary)
- **Centralized Configuration** - Instance lists managed in `settings.py` for easy modification
- **Multiple Solvers** - Tabu Search (tenure 0 & 5) and ALNS with greedy LNS
- **Real-time Progress Tracking** - Live updates as instances complete
- **Optimized for Python 3.14** - Leverages improved multiprocessing capabilities

---

## Project Structure

```
route-orchestrator/
├── main.py                      # Main batch processing script
├── settings.py                  # Configuration and instance lists
├── README.md                    # This file
├── data/
│   ├── instances/              # Customer instance files
│   │   ├── 100_customers/      # 56 instances with 100 customers
│   │   ├── 400_customers/      # 60 instances with 400 customers
│   │   ├── 800_customers/      # 60 instances with 800 customers
│   │   └── 1000_customers/     # 60 instances with 1000 customers
│   ├── fleets/                 # Fleet configuration JSON files
│   │   ├── C1.json            # Clustered customers, short schedules (9 vehicle types)
│   │   ├── C2.json            # Clustered customers, long schedules (8 vehicle types)
│   │   ├── R1.json            # Random customers, short schedules (12 vehicle types)
│   │   ├── R2.json            # Random customers, long schedules (11 vehicle types)
│   │   ├── RC1.json           # Mixed random-clustered, short (8 vehicle types)
│   │   └── RC2.json           # Mixed random-clustered, long (8 vehicle types)
│   └── raw_instances/          # Original benchmark instances
├── logs/
│   ├── execution/              # Detailed improvement logs (created on first run)
│   └── results/                # Final summary logs (created on first run)
├── solver/
│   ├── hffvrptw.py            # Main solver classes
│   ├── metaheuristics/
│   │   ├── ts.py              # Tabu Search implementations
│   │   └── alns.py            # ALNS implementation
│   └── problem/
│       ├── hffvrptw_problem_instance.py
│       ├── hffvrptw_solution.py
│       ├── hffvrptw_evaluator.py
│       ├── hffvrptw_initial_solution.py
│       └── model/
│           ├── constraints.py
│           └── objective_function.py
├── utils/
│   ├── data_loader.py         # Instance and fleet loading utilities
│   ├── instance_reader.py     # CSV parsing for instances
│   ├── results_logger.py      # Result logging utilities
│   └── adj_matrix.py          # Distance matrix calculations
└── np-solver/                  # Metaheuristic framework (submodule/dependency)
    └── np_solver/
        ├── core/
        ├── metaheuristics/
        └── reporting/
```

---

## Installation

### Prerequisites

- **Python 3.14** (recommended for optimal multiprocessing performance)
- **Git** for cloning repositories
- **Virtual environment** support (`venv`)

### Step-by-Step Setup

1. **Clone the route-orchestrator repository:**
   ```bash
   git clone https://github.com/code-lucasgabriel/route-orchestrator.git
   cd route-orchestrator
   ```

2. **Clone the np-solver framework:**
   
   The project depends on the `np-solver` metaheuristic framework. Clone it into the project directory:
   ```bash
   git clone https://github.com/code-lucasgabriel/np-solver.git
   ```

3. **Create and activate a virtual environment:**
   ```bash
   python3.14 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Install np-solver in editable mode:**
   ```bash
   python -m pip install -e ./np-solver
   ```
   
   This installs the framework in development mode, making it available to the route-orchestrator project.

5. **Verify installation:**
   ```bash
   python -c "from settings import INSTANCES; print(f'✓ Loaded {len(INSTANCES)} instances')"
   python -c "import np_solver; print('✓ np-solver framework installed')"
   ```

---

## Configuration

### Instance Configuration (`settings.py`)

The `settings.py` file centralizes all configuration:

```python
# Path constants
FLEETS_PATH = os.path.join(_base_path, "data/fleets")
INSTANCES_PATH = os.path.join(_base_path, "data/instances")
RESULTS_PATH = os.path.join(_base_path, "data/results")

# List of instances to process
INSTANCES = [
    # 100_customers - first 5
    "100_customers/C1_1_01.csv",
    "100_customers/C1_1_02.csv",
    # ... add more instances here
]
```

**To modify which instances are processed:**
1. Open `settings.py`
2. Edit the `INSTANCES` list
3. Add or remove instance paths as needed
4. Save and run - no changes needed in `main.py`

---

## Usage

### Running Batch Processing

Process all instances defined in `settings.py` using parallel execution:

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Run with Python 3.14 for optimal multiprocessing
python3.14 main.py
```

**What happens during execution:**
1. **Initialization** - Detects CPU count and spawns worker processes (CPU count - 1)
2. **Parallel Processing** - Multiple instances solve simultaneously
3. **Real-time Updates** - Progress messages show completions with costs
4. **Log Generation** - Both execution and results logs created for each instance
5. **Summary Report** - Final table displays all results sorted by instance name

### Understanding the Output

**Console Output Example:**
```
################################################################################
Starting batch processing of 20 instances
Using 11 parallel workers
################################################################################

================================================================================
Processing instance: 100_customers/C1_1_01.csv
================================================================================

Initial solution cost: 2694.99
Initial solution valid: True

Running solver for C1_1_01...

New best solution found: 2491.96 (at 19.76s)
New best solution found: 2391.49 (at 20.20s)

================================================================================
Completed: C1_1_01
Best cost: 2391.49
Found at: 20.20s
Total time: 20.21s
Execution log: logs/execution/C1_1_01.txt
Results log: logs/results/C1_1_01.txt
================================================================================

[1/20] Completed: C1_1_01 - Cost: 2391.49
[2/20] Completed: C1_1_03 - Cost: 2681.87
...

################################################################################
BATCH PROCESSING COMPLETE
Total batch time: 52.43s
################################################################################

Instance                       Best Cost       Found At (s)    Total Time (s) 
--------------------------------------------------------------------------------
C1_1_01                        2391.49         20.20           20.21          
C1_1_02                        2358.55         20.37           20.38          
C1_1_03                        2681.87         20.22           20.22          
...
```

### Logging System

The system generates **two types of logs** for each instance:

#### 1. Execution Logs (`logs/execution/`)

**Purpose:** Track the solver's progress and every improvement found during optimization.

**Format:** Alternating lines of cost and solution elements
```
2694.99
[[0, 51, 42, 43, 92, 97, 99, 0], [0, 8, 10, 4, 2, 1, 75, 0], ...]
2491.96
[[0, 42, 43, 41, 10, 11, 51, 50, 22, 0], [0, 93, 100, 2, 1, 0], ...]
2391.49
[[0, 43, 42, 40, 44, 45, 48, 51, 50, 52, 0], [0, 19, 16, 12, 2, 0], ...]
```

**Use cases:**
- Analyze convergence behavior
- Study improvement patterns
- Debug solver performance
- Generate convergence plots

#### 2. Results Logs (`logs/results/`)

**Purpose:** Store the final best solution in a compact format.

**Format:** Exactly 3 lines
```
2391.49                                    # Line 1: Best cost found
[[0, 43, 42, 40, 44, 45, ...], ...]       # Line 2: Best solution routes
20.20                                      # Line 3: Time when best was found (seconds)
```

**Use cases:**
- Quick result lookup
- Batch result analysis
- Performance comparison
- Export to external tools

---

## Data Structure

### Instance Files

Customer instance files are organized in `data/instances/` by customer count:

| Directory | Instance Count | Description |
|-----------|----------------|-------------|
| `100_customers/` | 56 instances | Small-scale problems for testing |
| `400_customers/` | 60 instances | Medium-scale problems |
| `800_customers/` | 60 instances | Large-scale problems |
| `1000_customers/` | 60 instances | Extra-large problems |

**Instance Naming Convention:** `{TYPE}_{FLEET_SIZE}_{NUMBER}.csv`

**Instance Type Prefixes:**
- **C1/C2** - Clustered customers (C1: short schedules, C2: long schedules)
- **R1/R2** - Random customers (R1: short schedules, R2: long schedules)
- **RC1/RC2** - Mixed random-clustered (RC1: short, RC2: long)

**Examples:**
- `C1_1_01.csv` - Clustered, fleet size 1, instance 01, 100 customers
- `R2_4_05.csv` - Random long schedule, fleet size 4, instance 05, 400 customers
- `RC1_10_3.csv` - Mixed random-clustered, fleet size 10, instance 3, 1000 customers

**CSV Format:**
Each instance file contains customer data with columns:
```
CUST NO., XCOORD., YCOORD., DEMAND, READY TIME, DUE DATE, SERVICE TIME
0, 50.0, 50.0, 0, 0, 230, 0          # Depot (customer 0)
1, 52.0, 64.0, 10, 155, 175, 10      # Customer 1
2, 96.0, 26.0, 20, 56, 76, 10        # Customer 2
...
```

### Fleet Configuration Files

Fleet configurations are stored as JSON files in `data/fleets/`:

| Fleet File | Instance Type | Vehicle Types | Description |
|------------|---------------|---------------|-------------|
| `C1.json` | C1 | 9 types | Clustered customers, short schedules |
| `C2.json` | C2 | 8 types | Clustered customers, long schedules |
| `R1.json` | R1 | 12 types | Random customers, short schedules |
| `R2.json` | R2 | 11 types | Random customers, long schedules |
| `RC1.json` | RC1 | 8 types | Mixed random-clustered, short schedules |
| `RC2.json` | RC2 | 8 types | Mixed random-clustered, long schedules |

**Fleet File Structure:**
```json
[
    {
        "type": "A",
        "count": 5,
        "capacity": 200,
        "latest_return_time": 230,
        "fixed_cost": 100,
        "variable_cost": 1.0
    },
    {
        "type": "B",
        "count": 3,
        "capacity": 150,
        "latest_return_time": 230,
        "fixed_cost": 80,
        "variable_cost": 1.2
    }
]
```

**Field Descriptions:**
- `type`: Vehicle type identifier (A, B, C, etc.)
- `count`: Number of vehicles available of this type
- `capacity`: Maximum load capacity
- `latest_return_time`: Latest time vehicle can return to depot
- `fixed_cost`: Fixed cost for using this vehicle type
- `variable_cost`: Variable cost per distance unit

**Automatic Mapping:**
The system automatically maps instance names to fleet files:
- `C1_1_01.csv` → `C1.json`
- `R2_4_05.csv` → `R2.json`
- `RC1_10_3.csv` → `RC1.json`

---

## Parallel Processing

### How It Works

The batch processor uses **Python multiprocessing** to solve multiple instances simultaneously:

1. **Worker Pool Creation** - Spawns `CPU_COUNT - 1` worker processes
2. **Task Distribution** - Distributes instances across workers using `imap_unordered`
3. **Independent Execution** - Each worker runs a complete solver independently
4. **Result Collection** - Results collected as they complete (unordered for efficiency)
5. **Final Sorting** - Results sorted by instance name for consistent display

### Performance Benefits

**Sequential vs Parallel Execution:**

| Scenario | Sequential Time | Parallel Time (11 workers) | Speedup |
|----------|----------------|----------------------------|---------|
| 20 instances × 25s avg | ~500 seconds (8.3 min) | ~50-80 seconds (1.3 min) | **6-10x faster** |

**Key Advantages:**
- Near-linear speedup with available CPU cores
- CPU-intensive optimization work distributed efficiently
- No waiting time between instances
- Full CPU utilization during processing

### Customizing Worker Count

By default, the system uses `CPU_COUNT - 1` workers. To customize:

```python
# In main.py, modify the run_batch call:
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    run_batch(num_workers=4)  # Use exactly 4 workers
```

---

## Solver Configuration

### Available Solvers

The system includes three metaheuristic solvers:

1. **`ts_tenure5`** (Default) - Tabu Search with tenure 5
2. **`ts_tenure0`** - Tabu Search with tenure 0 (no tabu restrictions)
3. **`alns_greedy_lns`** - Adaptive Large Neighborhood Search with greedy LNS

### Changing the Solver

Edit `main.py` in the `run_batch()` function:

```python
def run_batch(num_workers: int | None = None):
    # ...
    
    # Change this line to switch solvers:
    solver_name = 'ts_tenure5'        # Default
    # solver_name = 'ts_tenure0'      # Alternative 1
    # solver_name = 'alns_greedy_lns' # Alternative 2
    
    # ...
```

### Solver Implementation Details

Each solver uses the **np-solver framework** and follows this workflow:

1. **Problem Loading** - Read instance CSV and fleet JSON
2. **Initial Solution** - Generate feasible starting solution using constructive heuristic
3. **Optimization** - Apply metaheuristic (TS or ALNS) with improvement tracking
4. **Logging** - Record each improvement to execution log
5. **Result Export** - Save final best solution to results log

---

## Advanced Usage

### Loading Data Programmatically

```python
from utils import load_instance_and_fleet

# Load a single instance with its fleet configuration
customers, fleet, fleet_type = load_instance_and_fleet(
    'data/instances/100_customers/C1_1_01.csv'
)

# Access customer data
for customer in customers:
    print(f"Customer {customer['CUST NO.']}: "
          f"Demand={customer['DEMAND']}, "
          f"Window=[{customer['READY TIME']}, {customer['DUE DATE']}]")

# Access fleet configuration
for vehicle in fleet:
    print(f"Vehicle {vehicle['type']}: "
          f"Count={vehicle['count']}, "
          f"Capacity={vehicle['capacity']}")
```

### Utility Functions

```python
from utils import (
    load_customer_data,
    load_fleet_data,
    get_all_instances,
    get_instances_by_size,
    get_fleet_type_from_instance_name
)

# Load customers and fleet separately
customers = load_customer_data('data/instances/100_customers/C1_1_01.csv')
fleet = load_fleet_data('data/fleets/C1.json')

# Get all instances for a specific size
instances_100 = get_instances_by_size('data/instances', 100)

# Get all instances across all sizes
all_instances = get_all_instances('data/instances')

# Extract fleet type from filename
fleet_type = get_fleet_type_from_instance_name('C1_1_01.csv')  # Returns 'C1'
```

### Processing a Single Instance

To run a single instance without batch processing:

```python
from main import run_solver_for_instance

# Process one instance
result = run_solver_for_instance(
    instance_path="100_customers/C1_1_01.csv",
    solver_name='ts_tenure5'
)

print(f"Best cost: {result['best_cost']}")
print(f"Total time: {result['total_time']}")
```

### Analyzing Log Files

```python
import os

# Read execution log to analyze convergence
with open('logs/execution/C1_1_01.txt', 'r') as f:
    lines = f.readlines()
    costs = [float(lines[i]) for i in range(0, len(lines), 2)]
    print(f"Improvements: {len(costs)}")
    print(f"Initial: {costs[0]:.2f}, Final: {costs[-1]:.2f}")
    print(f"Total improvement: {costs[0] - costs[-1]:.2f}")

# Read results log for final solution
with open('logs/results/C1_1_01.txt', 'r') as f:
    best_cost = float(f.readline().strip())
    best_solution = eval(f.readline().strip())
    best_time = float(f.readline().strip())
    print(f"Best: {best_cost:.2f} (found at {best_time:.2f}s)")
    print(f"Routes: {len(best_solution)}")
```

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'np_solver'`
- **Solution:** Install np-solver: `python -m pip install -e ./np-solver`

**Issue:** `AttributeError: 'Metaheuristic' object has no attribute 'report_experiment'`
- **Solution:** Update np-solver to latest version or apply the typo fix in `np-solver/np_solver/metaheuristics/metaheuristic.py` (line 144: change `report_experiments` to `report_experiment`)

**Issue:** Multiprocessing errors on Windows
- **Solution:** Ensure `multiprocessing.set_start_method('spawn', force=True)` is in the `if __name__ == "__main__"` block

**Issue:** Too many/too few workers
- **Solution:** Manually set worker count: `run_batch(num_workers=4)`

---

## Performance Tips

1. **Use Python 3.14** - Significantly improved multiprocessing stability
2. **Monitor CPU usage** - Ensure workers aren't exceeding available cores
3. **Adjust worker count** - Leave 1-2 cores free for system tasks
4. **Start with small instances** - Test configuration before running large batches
5. **Check log disk space** - Large batches generate many log files

---

## License

TO DO

## Contributors

TO DO

## Citation

If you use this code in your research, please cite:

```
TO DO
```
