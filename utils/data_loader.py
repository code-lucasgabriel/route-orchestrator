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

