#!/usr/bin/env python3

"""
Main script for the Route Orchestrator.

This script loads customer instances and their corresponding fleet configurations,
then solves the Vehicle Routing Problem with Heterogeneous Fleet and Time Windows (VRPHETW).
"""

import os
import sys
from typing import List, Dict, Tuple

from utils import (
    load_instance_and_fleet,
    get_all_instances,
    get_instances_by_size
)


def calculate_distance(customer1: Dict, customer2: Dict) -> float:
    """
    Calculate Euclidean distance between two customers.
    
    Args:
        customer1: First customer with XCOORD. and YCOORD.
        customer2: Second customer with XCOORD. and YCOORD.
    
    Returns:
        Euclidean distance
    """
    dx = customer1['XCOORD.'] - customer2['XCOORD.']
    dy = customer1['YCOORD.'] - customer2['YCOORD.']
    return (dx**2 + dy**2) ** 0.5


def print_instance_info(customers: List[Dict], fleet: List[Dict], fleet_type: str):
    """
    Print summary information about the loaded instance.
    
    Args:
        customers: List of customer dictionaries
        fleet: List of vehicle type dictionaries
        fleet_type: Fleet type identifier
    """
    print(f"\n{'='*60}")
    print(f"Instance Information")
    print(f"{'='*60}")
    
    print(f"\nFleet Type: {fleet_type}")
    print(f"Number of Customers: {len(customers) - 1}")  # -1 for depot
    print(f"Depot: Customer {customers[0]['CUST NO.']}")
    
    print(f"\nFleet Configuration:")
    total_vehicles = 0
    total_capacity = 0
    for vehicle_type in fleet:
        print(f"  Type {vehicle_type['type']}:")
        print(f"    Count: {vehicle_type['count']}")
        print(f"    Capacity: {vehicle_type['capacity']}")
        print(f"    Latest Return Time: {vehicle_type['latest_return_time']}")
        print(f"    Fixed Cost: {vehicle_type['fixed_cost']}")
        print(f"    Variable Cost: {vehicle_type['variable_cost']}")
        total_vehicles += vehicle_type['count']
        total_capacity += vehicle_type['count'] * vehicle_type['capacity']
    
    print(f"\n  Total Vehicles: {total_vehicles}")
    print(f"  Total Fleet Capacity: {total_capacity}")
    
    # Calculate total demand
    total_demand = sum(c['DEMAND'] for c in customers[1:])  # Skip depot
    print(f"\n  Total Customer Demand: {total_demand}")
    print(f"  Capacity Utilization: {total_demand / total_capacity * 100:.2f}%")
    
    print(f"\nCustomer Statistics:")
    if len(customers) > 1:
        earliest_ready = min(c['READY TIME'] for c in customers[1:])
        latest_due = max(c['DUE DATE'] for c in customers[1:])
        avg_demand = total_demand / (len(customers) - 1)
        avg_service = sum(c['SERVICE TIME'] for c in customers[1:]) / (len(customers) - 1)
        
        print(f"  Time Window: [{earliest_ready}, {latest_due}]")
        print(f"  Average Demand: {avg_demand:.2f}")
        print(f"  Average Service Time: {avg_service:.2f}")
    
    print(f"{'='*60}\n")


def solve_instance(
    customers: List[Dict],
    fleet: List[Dict],
    fleet_type: str
) -> Dict:
    """
    Solve a single VRPHETW instance.
    
    This is a placeholder for your actual solver (ALNS, Tabu Search, etc.)
    
    Args:
        customers: List of customer dictionaries
        fleet: List of vehicle type dictionaries
        fleet_type: Fleet type identifier
    
    Returns:
        Solution dictionary with routes and cost information
    """
    # TODO: Implement your actual solver here
    # This could be ALNS, Tabu Search, or any other metaheuristic
    
    print(f"[TODO] Solving instance with {len(customers)-1} customers and fleet type {fleet_type}")
    print(f"[TODO] This is where your ALNS or Tabu Search algorithm would run")
    
    # Placeholder solution
    solution = {
        'fleet_type': fleet_type,
        'num_customers': len(customers) - 1,
        'routes': [],
        'total_cost': 0.0,
        'num_vehicles_used': 0,
        'status': 'unsolved'
    }
    
    return solution


def solve_single_instance(instance_path: str):
    """
    Load and solve a single instance.
    
    Args:
        instance_path: Path to the instance CSV file
    """
    print(f"Loading instance: {os.path.basename(instance_path)}")
    
    try:
        # Load both customer data and fleet configuration
        customers, fleet, fleet_type = load_instance_and_fleet(instance_path)
        
        # Print instance information
        print_instance_info(customers, fleet, fleet_type)
        
        # Solve the instance
        solution = solve_instance(customers, fleet, fleet_type)
        
        # Print results (placeholder)
        print(f"Solution Status: {solution['status']}")
        print(f"Total Cost: {solution['total_cost']}")
        
    except Exception as e:
        print(f"Error processing instance: {e}")
        import traceback
        traceback.print_exc()


def solve_all_instances_by_size(customer_count: int):
    """
    Load and solve all instances for a specific customer count.
    
    Args:
        customer_count: Number of customers (100, 400, 800, or 1000)
    """
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(script_dir, 'data', 'instances')
    
    print(f"\nFinding all instances with {customer_count} customers...")
    
    try:
        instances = get_instances_by_size(instances_dir, customer_count)
        print(f"Found {len(instances)} instances")
        
        for instance_path in instances:
            solve_single_instance(instance_path)
            print("\n" + "-"*60 + "\n")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    """
    Main entry point for the route orchestrator.
    """
    print("="*60)
    print("Route Orchestrator - VRPHETW Solver")
    print("="*60)
    
    # Example 1: Solve a single instance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_instance = os.path.join(
        script_dir, 
        'data', 
        'instances', 
        '100_customers', 
        'C1_1_01.csv'
    )
    
    if os.path.exists(example_instance):
        print("\n--- Example 1: Solving a single instance ---")
        solve_single_instance(example_instance)
    
    # Example 2: Solve all 100-customer instances
    # Uncomment to run:
    # print("\n--- Example 2: Solving all 100-customer instances ---")
    # solve_all_instances_by_size(100)
    
    # Example 3: Solve all instances of all sizes
    # Uncomment to run:
    # instances_dir = os.path.join(script_dir, 'data', 'instances')
    # all_instances = get_all_instances(instances_dir)
    # for instance_path in all_instances:
    #     solve_single_instance(instance_path)


if __name__ == "__main__":
    main()
