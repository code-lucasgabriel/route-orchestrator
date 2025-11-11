"""
test_parser.py

Quick test to verify the parser works correctly on a sample of log files.
"""

import sys
from pathlib import Path

# Test if required libraries are available
try:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    print("✓ All required libraries are installed")
except ImportError as e:
    print(f"✗ Missing library: {e}")
    print("\nPlease install requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Test if log files exist
logs_dir = Path('logs')
if not logs_dir.exists():
    print(f"✗ Logs directory not found: {logs_dir}")
    sys.exit(1)

# Count log files
result_files = list(logs_dir.rglob('results/*.txt'))
execution_files = list(logs_dir.rglob('execution/*.txt'))

print(f"\n✓ Found {len(result_files)} result files")
print(f"✓ Found {len(execution_files)} execution files")

if len(result_files) == 0:
    print("\n✗ No result files found. Have you run the experiments yet?")
    sys.exit(1)

# Test parsing a sample file
print("\nTesting parser on a sample file...")
sample_file = result_files[0]
print(f"  File: {sample_file}")

try:
    with open(sample_file, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    # Parse first line
    first_line = eval(lines[0])
    print(f"  ✓ Best solution: {first_line[0]:.2f} (found at {first_line[1]:.2f}s)")
    
    # Parse solution
    if len(lines) > 1:
        print(f"  ✓ Solution has {len(lines)-1} fleet types")
    
    print("\n✓ Parser test successful!")
    print("\nYou can now run:")
    print("  python parser.py     # Parse all logs")
    print("  python plotter.py    # Generate plots")
    
except Exception as e:
    print(f"\n✗ Error parsing file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
