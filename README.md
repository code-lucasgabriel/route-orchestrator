# route-orchestrator
An optimization suite for the Heterogeneous Fixed Fleet Vehicle Routing Problem with Time Windows (HFFVRPTW) featuring TS & ALNS solvers, a REST API, and an interactive route visualization map.

## Data Structure

### Instance Files
Customer instance files are organized in `data/instances/` by customer count:
- `100_customers/` - 56 instances
- `400_customers/` - 60 instances  
- `800_customers/` - 60 instances
- `1000_customers/` - 60 instances

Instance file naming convention: `{TYPE}_{FLEET_SIZE}_{NUMBER}.csv`
- Examples: `C1_1_01.csv`, `R2_4_05.csv`, `RC1_10_3.csv`

### Fleet Configuration Files
Fleet configurations are stored as JSON files in `data/fleets/`:
- `C1.json` - Fleet for C1 type instances (9 cases)
- `C2.json` - Fleet for C2 type instances (8 cases)
- `R1.json` - Fleet for R1 type instances (12 cases)
- `R2.json` - Fleet for R2 type instances (11 cases)
- `RC1.json` - Fleet for RC1 type instances (8 cases)
- `RC2.json` - Fleet for RC2 type instances (8 cases)

Each fleet file contains vehicle type definitions with:
- `type`: Vehicle type identifier (A, B, C)
- `count`: Number of vehicles available
- `capacity`: Vehicle capacity
- `latest_return_time`: Latest time vehicle can return to depot
- `fixed_cost`: Fixed cost for using this vehicle type
- `variable_cost`: Variable cost per distance unit

## Usage

### Loading Data

```python
from utils import load_instance_and_fleet

# Load a single instance with its fleet configuration
customers, fleet, fleet_type = load_instance_and_fleet(
    'data/instances/100_customers/C1_1_01.csv'
)

# The system automatically maps instance names to fleet files:
# C1_1_01.csv -> C1.json
# R2_4_05.csv -> R2.json
# RC1_10_3.csv -> RC1.json
```

### Running the Example

```bash
python3 main.py
```

This will load and display information for an example instance, showing:
- Fleet configuration details
- Customer statistics
- Capacity utilization
- Time window information

### Advanced Usage

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

## Implementing Your Solver

To integrate your ALNS or Tabu Search algorithm, modify the `solve_instance()` function in `main.py`:

```python
def solve_instance(customers: List[Dict], fleet: List[Dict], fleet_type: str) -> Dict:
    """
    Solve a single VRPHETW instance.
    
    Args:
        customers: List of customer dictionaries with keys:
            - CUST NO.: Customer number (0 is depot)
            - XCOORD., YCOORD.: Coordinates
            - DEMAND: Customer demand
            - READY TIME, DUE DATE: Time window
            - SERVICE TIME: Service duration
        
        fleet: List of vehicle type dictionaries with keys:
            - type, count, capacity
            - latest_return_time, fixed_cost, variable_cost
        
        fleet_type: Fleet type identifier (C1, R1, RC1, etc.)
    
    Returns:
        Solution dictionary
    """
    # Your solver implementation here
    # Example: solution = your_alns_solver(customers, fleet)
    pass
```
