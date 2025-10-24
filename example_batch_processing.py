#!/usr/bin/env python3

"""
Example script showing how to process multiple instances.
"""

import os
from typing import Optional, List, Dict
from utils import load_instance_and_fleet, get_instances_by_size


def process_all_instances_of_type(customer_count: int, instance_type: Optional[str] = None):
    """
    Process all instances of a given size and optionally filter by type.
    
    Args:
        customer_count: Number of customers (100, 400, 800, 1000)
        instance_type: Optional filter for instance type (C1, R1, RC1, etc.)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(script_dir, 'data', 'instances')
    
    # Get all instances for this size
    instances = get_instances_by_size(instances_dir, customer_count)
    
    # Filter by type if specified
    if instance_type:
        instances = [i for i in instances if instance_type in os.path.basename(i)]
    
    print(f"Processing {len(instances)} instances...")
    
    results = []
    for instance_path in instances:
        instance_name = os.path.basename(instance_path)
        print(f"\nProcessing: {instance_name}")
        
        try:
            # Load instance and fleet
            customers, fleet, fleet_type = load_instance_and_fleet(instance_path)
            
            # Display basic info
            num_customers = len(customers) - 1
            total_demand = sum(c['DEMAND'] for c in customers[1:])
            total_vehicles = sum(v['count'] for v in fleet)
            
            print(f"  Fleet Type: {fleet_type}")
            print(f"  Customers: {num_customers}")
            print(f"  Total Demand: {total_demand}")
            print(f"  Available Vehicles: {total_vehicles}")
            
            # Here you would call your solver
            # solution = solve_with_alns(customers, fleet)
            # or
            # solution = solve_with_tabu_search(customers, fleet)
            
            results.append({
                'instance': instance_name,
                'fleet_type': fleet_type,
                'customers': num_customers,
                'demand': total_demand,
                'vehicles': total_vehicles
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    return results


def main():
    """Example usage."""
    
    print("="*60)
    print("Batch Processing Example")
    print("="*60)
    
    # Example 1: Process all 100-customer C1 instances
    print("\n--- Processing all C1 instances with 100 customers ---")
    results = process_all_instances_of_type(100, 'C1')
    print(f"\nProcessed {len(results)} instances")
    
    # Example 2: Process all R-type instances
    print("\n--- Processing all R1 instances with 100 customers ---")
    results = process_all_instances_of_type(100, 'R1')
    print(f"\nProcessed {len(results)} instances")


if __name__ == "__main__":
    main()
