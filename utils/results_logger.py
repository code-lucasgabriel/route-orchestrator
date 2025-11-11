import os
import json
from typing import List
from np_solver.core import BaseSolution
import shutil
import tempfile

def save_solution_json(results_dir: str, filename: str, solution_data: BaseSolution):
    """
    Saves a final solution as a compact, single-line JSON file,
    wrapped in a dictionary.

    Args:
        results_dir: The path to the directory (e.g., 'results/').
        filename: The name of the file (e.g., 'solution_1.json').
        solution_data: The solution data to save.
    """
    
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)

    data = {}
    for i, route in enumerate(solution_data):
        data[f"Vehicle_{i}"] = route

    data_to_save = {
        "solution": {"Routes": data, "Total Cost": solution_data.cost}
    }

    with open(filepath, 'w') as f:
        json.dump(data_to_save, f)
