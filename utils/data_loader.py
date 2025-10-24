#!/usr/bin/env python3

"""
Parses customer data from .txt files (Solomon CVRPTW format)
and converts them to .csv files.

This script dynamically finds the 'CUST NO.' header, skips the
next blank line, and processes all subsequent lines as data.

Usage:
    python data_loader.py <input_directory> <output_directory>
"""

import sys
import os
import glob
import csv
import re
import json
import pandas as pd
from typing import Tuple, List, Dict, Optional

def load_solomon_file(txt_file_path, csv_file_path):
    """
    Parses a single .txt file and saves its customer data as a .csv file.
    
    - Dynamically finds the header line starting with "CUST NO.".
    - Skips the blank line immediately after the header.
    - Parses all subsequent lines as data rows.
    """
    all_lines = []
    try:        
        with open(txt_file_path, 'r', encoding='utf-8') as f_in:
            all_lines = f_in.readlines()
    except UnicodeDecodeError:
        try:    
            print(f"    [INFO] UTF-8 failed for {txt_file_path}, retrying with 'latin-1'.")
            with open(txt_file_path, 'r', encoding='latin-1') as f_in:
                all_lines = f_in.readlines()
        except Exception as e:
            print(f"    [FAIL] Could not read file {txt_file_path}: {e}")
            return
    except FileNotFoundError:
        print(f"    [FAIL] Error: File not found {txt_file_path}")
        return
    except Exception as e:
        print(f"    [FAIL] Error processing {txt_file_path}: {e}")
        return
    header_index = -1
    for i, line in enumerate(all_lines):
        if line.strip().startswith("CUST NO."):
            header_index = i
            break
    if header_index == -1:
        print(f"    [SKIP] Could not find header 'CUST NO.' in {txt_file_path}. Skipping file.")
        return

    header_line = all_lines[header_index].strip()
    
    header_line = re.sub(r'SERVICE\s+TIME', 'SERVICE_TIME', header_line)

    if header_index + 2 >= len(all_lines):
        print(f"    [SKIP] File {txt_file_path} has header but no data lines. Skipping file.")
        return
        
    data_lines = all_lines[header_index + 2:]
    
    header = re.split(r'\s{2,}', header_line)
    
    header = [h.replace('SERVICE_TIME', 'SERVICE TIME') for h in header]
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
            
        writer.writerow(header)

        rows_written = 0
        for line in data_lines:
            line = line.strip()
            if line:                  
                row = line.split()
                
                if len(row) == len(header):
                    writer.writerow(row)
                    rows_written += 1
                else:                    
                    print(f"    [WARN] Skipping malformed line in {txt_file_path} (columns mismatch):")
                    print(f"             Header has {len(header)} cols: {header}")
                    print(f"             Data has {len(row)} cols: {row}")
        
        if rows_written > 0:
            print(f"    [OK] Converted {txt_file_path} -> {csv_file_path} ({rows_written} rows)")
        else:
            print(f"    [WARN] Converted {txt_file_path} but wrote 0 data rows (check for [WARN] messages).")

def main():
    """
    Main function to handle command-line arguments, find files,
    and orchestrate the parsing.
    """
    
    if len(sys.argv) != 3:
        print("Usage: python data_loader.py <input_directory> <output_directory>")
        print("Example: python data_loader.py ./my_txt_files ./my_csv_files")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
        
    txt_files = glob.glob(os.path.join(input_dir, "*.txt")) or glob.glob(os.path.join(input_dir, "*.TXT"))

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} .txt files. Starting conversion...")

    for txt_file in txt_files:
        
        base_name = os.path.basename(txt_file)
            
        file_name_no_ext = os.path.splitext(base_name)[0]
        
        csv_name = f"{file_name_no_ext}.csv"
        
        csv_file_path = os.path.join(output_dir, csv_name)
        
        load_solomon_file(txt_file, csv_file_path)

    print("\nConversion complete.")

if __name__ == "__main__":
    main()


def get_fleet_type_from_instance_name(instance_name: str) -> str:
    """
    Extract the fleet type from an instance filename.
    
    Examples:
        'C1_1_01.csv' -> 'C1'
        'R2_4_05.csv' -> 'R2'
        'RC1_10_3.csv' -> 'RC1'
    
    Args:
        instance_name: The instance filename (e.g., 'C1_1_01.csv')
    
    Returns:
        The fleet type (e.g., 'C1', 'R2', 'RC1')
    """
    # Remove file extension
    base_name = os.path.splitext(instance_name)[0]
    
    # Split by underscore and take the first part (e.g., 'C1', 'R2', 'RC1')
    parts = base_name.split('_')
    
    if len(parts) >= 1:
        return parts[0]
    
    raise ValueError(f"Cannot extract fleet type from instance name: {instance_name}")


def load_fleet_data(fleet_file_path: str) -> List[Dict]:
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


def load_customer_data(instance_file_path: str) -> List[Dict]:
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


def load_instance_and_fleet(
    instance_path: str,
    fleet_dir: Optional[str] = None
) -> Tuple[List[Dict], List[Dict], str]:
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
    customers = load_customer_data(instance_path)
    
    # Extract fleet type from instance filename
    instance_name = os.path.basename(instance_path)
    fleet_type = get_fleet_type_from_instance_name(instance_name)
    
    # Determine fleet directory
    if fleet_dir is None:
        # Default: ../fleets relative to instance directory
        # If instance is in data/instances/100_customers/file.csv
        # Fleet should be in data/fleets/
        instance_dir = os.path.dirname(instance_path)  # data/instances/100_customers
        data_dir = os.path.dirname(instance_dir)  # data/instances
        data_dir = os.path.dirname(data_dir)  # data
        fleet_dir = os.path.join(data_dir, 'fleets')
    
    # Construct fleet file path
    fleet_file_path = os.path.join(fleet_dir, f"{fleet_type}.json")
    
    # Load fleet data
    fleet = load_fleet_data(fleet_file_path)
    
    return customers, fleet, fleet_type


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

