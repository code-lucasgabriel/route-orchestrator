import csv
import os

def log_experiment_data(results_dir, filename, data_row):
    """
    Appends a dictionary of data as a new row to a CSV file.

    This function handles directory creation, file creation, and
    CSV header writing automatically.

    Args:
        results_dir (str): The path to the directory (e.g., 'logs/').
        filename (str): The name of the file (e.g., 'experiment_1.csv').
        data_row (dict): The data to log (e.g., {'epoch': 1, 'loss': 0.5}).
    """
    
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    file_exists = os.path.exists(filepath)

    is_empty = os.path.getsize(filepath) == 0 if file_exists else True
    write_header = not file_exists or is_empty
    fieldnames = data_row.keys()

    with open(filepath, 'a', newline='') as f:        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()

        writer.writerow(data_row)
