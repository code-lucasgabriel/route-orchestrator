from settings import FLEETS_PATH, INSTANCES_PATH
import os
from typing import List, Dict, Tuple, Optional
import csv
import json
import glob

def _get_fleet_type_from_instance_name(instance_name: str) -> str:
    """
    Extract the fleet type from an instance filename.
    
    Args:
        instance_name: The instance filename (e.g., 'C1_1_01.csv')
    
    Returns:
        The fleet type (e.g., 'C1', 'R2', 'RC1')
    """
    
    base_name = os.path.splitext(instance_name)[0]
    
    
    parts = base_name.split('_')
    
    if len(parts) >= 1:
        return parts[0]
    
    raise ValueError(f"Cannot extract fleet type from instance name: {instance_name}")


def _load_fleet_data(fleet_file_path: str) -> List[Dict]:
    """
    Load fleet configuration from a JSON file.
    
    Args:
        fleet_file_path: Path to the fleet JSON file
    
    Returns:
        List of vehicle type dictionaries with keys:
        - type: Vehicle type identifier (e.g., 'A', 'B', 'C')
        - count: Number of vehicles of this type
        - capacity: Vehicle capacity
        - latest_return_time: Latest time vehicle can return to depot
        - fixed_cost: Fixed cost for using this vehicle type
        - variable_cost: Variable cost per distance unit
    """
    try:
        with open(fleet_file_path, 'r', encoding='utf-8') as f:
            fleet_data = json.load(f)
        return fleet_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Fleet file not found: {fleet_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in fleet file {fleet_file_path}: {e}")


def _load_customer_data(instance_file_path: str) -> List[Dict]:
    """
    Load customer data from a CSV file.
    
    Args:
        instance_file_path: Path to the customer instance CSV file
    
    Returns:
        List of customer dictionaries with keys matching CSV headers:
        - CUST NO.: Customer number (0 is depot)
        - XCOORD.: X coordinate
        - YCOORD.: Y coordinate
        - DEMAND: Customer demand
        - READY TIME: Earliest service time
        - DUE DATE: Latest service time
        - SERVICE TIME: Service duration
    """
    customers = []
    
    try:
        with open(instance_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields to appropriate types
                customer = {
                    'CUST NO.': int(row['CUST NO.']),
                    'XCOORD.': float(row['XCOORD.']),
                    'YCOORD.': float(row['YCOORD.']),
                    'DEMAND': int(row['DEMAND']),
                    'READY TIME': int(row['READY TIME']),
                    'DUE DATE': int(row['DUE DATE']),
                    'SERVICE TIME': int(row['SERVICE TIME'])
                }
                customers.append(customer)
        
        return customers
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance file not found: {instance_file_path}")
    except KeyError as e:
        raise ValueError(f"Missing required column in instance file: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid data format in instance file: {e}")


def load_instance(
    instance_name: str,
    fleet_dir: Optional[str] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load both customer instance data and corresponding fleet configuration.
    
    Args:
        instance_path: Path to the customer instance CSV file
        fleet_dir: Directory containing fleet JSON files (default: ../fleets relative to instance)
    
    Returns:
        Tuple of (customers, fleet, fleet_type):
        - customers: List of customer dictionaries
        - fleet: List of vehicle type dictionaries
        - fleet_type: The fleet type identifier (e.g., 'C1', 'R2')
    """
    # Load customer data
    if instance_name is None:
        raise TypeError("Input 'instance_name' must be a valid file!")
    
    instance_path = os.path.join(INSTANCES_PATH, instance_name)
    if not os.path.isfile(instance_path):
        raise TypeError("Input 'instance_name' must be a valid file!")

    customers = _load_customer_data(instance_path)
    # Extract fleet type from instance filename
    instance_name = os.path.basename(instance_path)
    fleet_type = _get_fleet_type_from_instance_name(instance_name)
    
    # Determine fleet directory
    if fleet_dir is None:
        fleet_dir = FLEETS_PATH
    
    # Construct fleet file path
    fleet_file_path = os.path.join(fleet_dir, f"{fleet_type}.json")
    
    # Load fleet data
    fleet = _load_fleet_data(fleet_file_path)
    
    return customers, fleet

def get_all_instances(instances_dir: str, pattern: str = "*.csv") -> List[str]:
    """
    Get all instance files from the instances directory.
    
    Args:
        instances_dir: Path to the instances directory
        pattern: File pattern to match (default: "*.csv")
    
    Returns:
        List of absolute paths to instance files
    """
    instance_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(instances_dir):
        for file in files:
            if file.endswith('.csv'):
                instance_files.append(os.path.join(root, file))
    
    return sorted(instance_files)


def get_instances_by_size(instances_dir: str, customer_count: int) -> List[str]:
    """
    Get all instance files for a specific customer count.
    
    Args:
        instances_dir: Path to the instances directory
        customer_count: Number of customers (e.g., 100, 400, 800, 1000)
    
    Returns:
        List of absolute paths to instance files
    """
    subdir = f"{customer_count}_customers"
    target_dir = os.path.join(instances_dir, subdir)
    
    if not os.path.isdir(target_dir):
        raise ValueError(f"Instance directory not found: {target_dir}")
    
    return sorted(glob.glob(os.path.join(target_dir, "*.csv")))


