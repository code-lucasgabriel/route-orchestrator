# Data Loading Implementation Summary

## What Was Implemented

### 1. Enhanced Data Loader (`utils/data_loader.py`)

Added the following new functions:

- **`get_fleet_type_from_instance_name(instance_name: str)`**
  - Extracts fleet type from instance filename
  - Examples: `C1_1_01.csv` → `C1`, `R2_4_05.csv` → `R2`

- **`load_fleet_data(fleet_file_path: str)`**
  - Loads fleet configuration from JSON files
  - Returns list of vehicle type dictionaries

- **`load_customer_data(instance_file_path: str)`**
  - Loads customer data from CSV files
  - Converts data to typed dictionaries

- **`load_instance_and_fleet(instance_path: str, fleet_dir: Optional[str] = None)`**
  - **Main function for loading both datasets together**
  - Automatically maps instance name to correct fleet file
  - Returns: `(customers, fleet, fleet_type)`

- **`get_all_instances(instances_dir: str)`**
  - Recursively finds all instance files
  - Returns sorted list of paths

- **`get_instances_by_size(instances_dir: str, customer_count: int)`**
  - Gets all instances for a specific customer count
  - Example: `get_instances_by_size('data/instances', 100)`

### 2. Updated Package Exports (`utils/__init__.py`)

All new functions are exposed at package level for easy import.

### 3. Main Script (`main.py`)

Demonstrates complete usage:
- Loading single instances
- Printing instance information
- Calculating statistics
- Placeholder for solver integration

### 4. Batch Processing Example (`example_batch_processing.py`)

Shows how to:
- Process multiple instances
- Filter by instance type
- Collect results

## Instance-to-Fleet Mapping

The system automatically maps instance files to fleet files:

| Instance Pattern | Fleet File | Description |
|-----------------|------------|-------------|
| `C1_*_*.csv`    | `C1.json`  | Type C1 instances (9 cases) |
| `C2_*_*.csv`    | `C2.json`  | Type C2 instances (8 cases) |
| `R1_*_*.csv`    | `R1.json`  | Type R1 instances (12 cases) |
| `R2_*_*.csv`    | `R2.json`  | Type R2 instances (11 cases) |
| `RC1_*_*.csv`   | `RC1.json` | Type RC1 instances (8 cases) |
| `RC2_*_*.csv`   | `RC2.json` | Type RC2 instances (8 cases) |

## Usage Examples

### Basic Usage

```python
from utils import load_instance_and_fleet

# Load everything you need in one call
customers, fleet, fleet_type = load_instance_and_fleet(
    'data/instances/100_customers/C1_1_01.csv'
)

# Now pass to your solver
solution = your_solver(customers, fleet)
```

### Batch Processing

```python
from utils import get_instances_by_size, load_instance_and_fleet

# Get all 100-customer instances
instances = get_instances_by_size('data/instances', 100)

for instance_path in instances:
    customers, fleet, fleet_type = load_instance_and_fleet(instance_path)
    solution = your_solver(customers, fleet)
    save_results(solution)
```

### Filtering by Type

```python
import os
from utils import get_instances_by_size, load_instance_and_fleet

# Get only C1 instances with 100 customers
all_instances = get_instances_by_size('data/instances', 100)
c1_instances = [i for i in all_instances if 'C1_' in os.path.basename(i)]

for instance_path in c1_instances:
    customers, fleet, fleet_type = load_instance_and_fleet(instance_path)
    # Process...
```

## Data Structures

### Customer Dictionary
```python
{
    'CUST NO.': int,        # Customer number (0 = depot)
    'XCOORD.': float,       # X coordinate
    'YCOORD.': float,       # Y coordinate
    'DEMAND': int,          # Demand quantity
    'READY TIME': int,      # Time window start
    'DUE DATE': int,        # Time window end
    'SERVICE TIME': int     # Service duration
}
```

### Vehicle Type Dictionary
```python
{
    'type': str,                    # Vehicle type ID (A, B, C)
    'count': int,                   # Number available
    'capacity': int,                # Vehicle capacity
    'latest_return_time': int,      # Max return time
    'fixed_cost': float,            # Fixed cost per vehicle
    'variable_cost': float          # Cost per distance unit
}
```

## Next Steps for Integration

1. **Implement your solver** in `main.py`:
   ```python
   def solve_instance(customers, fleet, fleet_type):
       # Your ALNS or Tabu Search implementation
       solution = your_algorithm(customers, fleet)
       return solution
   ```

2. **Test with small instances** first (100 customers)

3. **Scale up** to larger instances (400, 800, 1000)

4. **Save results** to a standardized format for comparison

## Testing

Run the examples:
```bash
# Test single instance loading
python3 main.py

# Test batch processing
python3 example_batch_processing.py
```

All tests pass successfully! ✅
