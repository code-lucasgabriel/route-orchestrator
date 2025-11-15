"""
parser.py

A script to parse the raw experiment logs into a single, consolidated
analysis-ready Parquet file.

Assumes a log directory structure like:
logs/
    <algorithm_name>/
        results/
            <instance_name>.txt

Where <instance_name> is in the format:
    <category>_<cust_size>_<id>.txt
    e.g., "C1_1_01.txt" or "RC2_10_5.txt"

And <algorithm_name> is one of:
    "ts_tenure0", "ts_tenure5", "alns_adaptive_sa", "alns_greedy_lns"
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
LOGS_ROOT_DIR = 'logs'
OUTPUT_FILE = 'results.parquet'

# Regex to parse instance names like "C1_1_01" or "RC2_10_5"
# Group 1: Category (R1, R2, C1, C2, RC1, RC2)
# Group 2: Customer Size scale (1, 4, 8, 10) which maps to (100, 400, 800, 1000)
# Group 3: Instance ID (e.g., "01", "1", "05")
INSTANCE_REGEX = re.compile(r"^(R1|R2|C1|C2|RC1|RC2)_(\d+)_(.*)$")

# Map from scale to actual customer size
SCALE_TO_SIZE = {
    '1': 100,
    '4': 400,
    '8': 800,
    '10': 1000
}

def parse_solution_line(solution_str: str) -> dict:
    """
    Parses the solution line which has format like:
    A: [[0, 1, 2, 0], [0, 3, 4, 0]]
    B: [[0, 5, 6, 0]]
    
    Returns a dictionary mapping fleet type to routes.
    """
    solution_dict = {}
    
    # Split by lines and process each fleet type
    lines = solution_str.strip().split('\n')
    for line in lines:
        if ':' not in line:
            continue
        
        # Split fleet type and routes
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
            
        fleet_type = parts[0].strip()
        routes_str = parts[1].strip()
        
        # Parse the routes using eval (safe here as we control the log format)
        try:
            routes = eval(routes_str)
            solution_dict[fleet_type] = routes
        except:
            # If eval fails, skip this fleet type
            continue
    
    return solution_dict


def parse_log_file(log_path: Path) -> dict | None:
    """
    Parses a single log file and extracts all relevant data.
    """
    try:
        # 1. Extract metadata from the file path
        # e.g., path = "logs/ts_tenure0/results/C1_1_01.txt"
        # algorithm = "ts_tenure0"
        algorithm = log_path.parent.parent.name
        # instance = "C1_1_01"
        instance = log_path.stem
        
        # 2. Parse instance name with regex
        match = INSTANCE_REGEX.match(instance)
        if not match:
            print(f"Warning: Skipping file, could not parse instance name: {instance}")
            return None
            
        category = match.group(1)
        scale = match.group(2)
        
        # Map scale to customer size
        cust_size = SCALE_TO_SIZE.get(scale)
        if cust_size is None:
            print(f"Warning: Skipping file, unknown customer size scale '{scale}' in: {instance}")
            return None
        
        # 3. Read log content
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Split into lines
        lines = content.strip().split('\n')
        
        if len(lines) < 2:
            print(f"Warning: Skipping file, not enough lines: {log_path}")
            return None
        
        # First line: [best_loss, time]
        first_line = lines[0].strip()
        try:
            final_data = eval(first_line)
            if not isinstance(final_data, list) or len(final_data) != 2:
                raise ValueError("First line is not a [loss, time] pair")
            final_loss = float(final_data[0])
            final_time = float(final_data[1])
        except Exception as e:
            print(f"Warning: Skipping file, could not parse first line in {log_path}: {e}")
            return None
        
        # Remaining lines: solution dictionary (fleet_type: routes)
        solution_str = '\n'.join(lines[1:])
        final_solution = parse_solution_line(solution_str)
        
        # 4. Parse execution log to get convergence history
        exec_log_path = log_path.parent.parent / 'execution' / log_path.name
        
        loss_history = []
        time_history = []
        
        if exec_log_path.exists():
            with open(exec_log_path, 'r') as f:
                exec_lines = f.readlines()
            
            for line in exec_lines:
                line = line.strip()
                if line.startswith('[') and ',' in line and ']' in line:
                    try:
                        data = eval(line)
                        if isinstance(data, list) and len(data) == 2:
                            loss_history.append(float(data[0]))
                            time_history.append(float(data[1]))
                    except:
                        # Skip lines that don't parse
                        continue
        
        # If no history found, use final values
        if len(loss_history) == 0:
            loss_history = [final_loss]
            time_history = [final_time]
        
        # Convert to numpy arrays for efficient storage
        loss_history = np.array(loss_history, dtype=np.float64)
        time_history = np.array(time_history, dtype=np.float64)
        
        # 5. Assemble the record
        record = {
            'algorithm': algorithm,
            'instance': instance,
            'category': category,
            'cust_size': cust_size,
            'final_loss': final_loss,
            'final_time': final_time,
            'loss_history': loss_history,
            'time_history': time_history,
            'final_solution': final_solution
        }
        return record

    except Exception as e:
        print(f"Error parsing file {log_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function to find all logs, parse them, and save to Parquet.
    """
    print(f"Starting log parsing from root directory: {LOGS_ROOT_DIR}")
    
    # Use pathlib.Path.rglob to recursively find all .txt files in results/ directories
    results_dirs = list(Path(LOGS_ROOT_DIR).rglob('results'))
    
    log_files = []
    for results_dir in results_dirs:
        log_files.extend(list(results_dir.glob('*.txt')))
    
    print(f"Found {len(log_files)} log files to parse.")
    
    all_results = []
    
    # Use tqdm for a progress bar
    for log_path in tqdm(log_files, desc="Parsing logs"):
        record = parse_log_file(log_path)
        if record:
            all_results.append(record)
            
    if not all_results:
        print("Error: No results were successfully parsed. Exiting.")
        return
        
    # Convert the list of dicts to a DataFrame (most efficient method)
    df = pd.DataFrame(all_results)
    
    # Sort by instance and algorithm for easier analysis
    df = df.sort_values(['instance', 'algorithm']).reset_index(drop=True)
    
    # Save to Parquet for efficient storage and I/O
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"\n{'='*80}")
    print(f"Successfully parsed {len(df)} records.")
    print(f"Master analysis file saved to: {OUTPUT_FILE}")
    print(f"{'='*80}\n")
    
    print("Data Schema:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.groupby(['algorithm', 'cust_size'])['final_loss'].describe())
    print("\nExample records:")
    print(df.head(10)[['algorithm', 'instance', 'category', 'cust_size', 'final_loss', 'final_time']])


if __name__ == "__main__":
    main()
