import os

_base_path = os.getcwd()
# path constants
FLEETS_PATH = os.path.join(_base_path, "data/fleets")
INSTANCES_PATH = os.path.join(_base_path, "data/instances")
RESULTS_PATH = os.path.join(_base_path, "data/results")

# Solver configuration
TIME_LIMIT = 10  # Time limit in seconds for both ALNS and TS solvers

# List of instances to process (first 5 from each customer size)
INSTANCES = [
    # 100_customers - first 5
    "100_customers/C1_1_01.csv",
    "100_customers/C1_1_02.csv",
    "100_customers/C1_1_03.csv",
    "100_customers/C1_1_04.csv",
    "100_customers/C1_1_05.csv",
    # 400_customers - first 5
    "400_customers/C1_4_1.csv",
    "400_customers/C1_4_2.csv",
    "400_customers/C1_4_3.csv",
    "400_customers/C1_4_4.csv",
    "400_customers/C1_4_5.csv",
    # 800_customers - first 5
    "800_customers/C1_8_1.csv",
    "800_customers/C1_8_2.csv",
    "800_customers/C1_8_3.csv",
    "800_customers/C1_8_4.csv",
    "800_customers/C1_8_5.csv",
    # 1000_customers - first 5
    "1000_customers/C1_10_1.csv",
    "1000_customers/C1_10_2.csv",
    "1000_customers/C1_10_3.csv",
    "1000_customers/C1_10_4.csv",
    "1000_customers/C1_10_5.csv",
]
