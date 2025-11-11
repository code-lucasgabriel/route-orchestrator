# Performance Analysis Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     STEP 1: Run Experiments                      │
│                                                                   │
│  $ python main.py                                                │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ alns_adaptive│  │ alns_greedy_ │  │  ts_tenure5  │  ...     │
│  │     _sa      │  │     lns      │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
│         v                 v                  v                   │
│  logs/alns_adaptive_sa/  logs/alns_greedy_lns/  logs/ts_tenure5/│
│      ├── execution/          ├── execution/     ├── execution/  │
│      │   ├── C1_1_01.txt     │   └── ...        │   └── ...     │
│      │   └── ...             │                  │                │
│      └── results/            └── results/       └── results/     │
│          ├── C1_1_01.txt         └── ...            └── ...      │
│          └── ...                                                 │
└───────────────────────────────────────────────────────────────────┘

                              ▼

┌─────────────────────────────────────────────────────────────────┐
│                     STEP 2: Parse Logs                           │
│                                                                   │
│  $ python parser.py                                              │
│                                                                   │
│  Scans all log files:                                            │
│  • logs/*/results/*.txt   (final solutions)                      │
│  • logs/*/execution/*.txt (convergence history)                  │
│                                                                   │
│  Extracts:                                                       │
│  • Algorithm name (from path)                                    │
│  • Instance name, category, customer size (from filename)        │
│  • Final loss, final time (from results file line 1)            │
│  • Fleet routes (from results file lines 2+)                     │
│  • Loss history, time history (from execution file)              │
│                                                                   │
│  Output: results.parquet                                         │
│  ┌──────────────────────────────────────────────────┐            │
│  │ algorithm │ instance │ category │ cust_size │... │            │
│  ├───────────┼──────────┼──────────┼───────────┼────┤            │
│  │ alns_sa   │ C1_1_01  │ C1       │ 100       │... │            │
│  │ alns_lns  │ C1_1_01  │ C1       │ 100       │... │            │
│  │ ts_5      │ C1_1_01  │ C1       │ 100       │... │            │
│  │ ts_0      │ C1_1_01  │ C1       │ 100       │... │            │
│  │ ...       │ ...      │ ...      │ ...       │... │            │
│  └──────────────────────────────────────────────────┘            │
│  944 rows × 9 columns                                            │
└───────────────────────────────────────────────────────────────────┘

                              ▼

┌─────────────────────────────────────────────────────────────────┐
│                   STEP 3: Generate Plots                         │
│                                                                   │
│  $ python plotter.py                                             │
│                                                                   │
│  Creates plots/ directory with:                                  │
│                                                                   │
│  📊 TTT Plots (24 files)                                         │
│     ttt_C1_100.png   ttt_C2_100.png   ttt_R1_100.png   ...      │
│     ttt_C1_400.png   ttt_C2_400.png   ttt_R1_400.png   ...      │
│     ...                                                          │
│                                                                   │
│  📈 Performance Profiles (11 files)                              │
│     performance_profile.png          (overall, 236 instances)    │
│     performance_profile_100.png      (by size)                   │
│     performance_profile_400.png                                  │
│     performance_profile_800.png                                  │
│     performance_profile_1000.png                                 │
│     performance_profile_C1.png       (by category)               │
│     performance_profile_C2.png                                   │
│     performance_profile_R1.png                                   │
│     performance_profile_R2.png                                   │
│     performance_profile_RC1.png                                  │
│     performance_profile_RC2.png                                  │
│                                                                   │
│  📉 Convergence Examples (3 files)                               │
│     convergence_C1_1_01.png                                      │
│     convergence_C1_1_02.png                                      │
│     convergence_C1_1_03.png                                      │
│                                                                   │
│  📋 Summary Statistics (3 files)                                 │
│     summary_overall.csv                                          │
│     summary_by_size.csv                                          │
│     summary_by_category.csv                                      │
└───────────────────────────────────────────────────────────────────┘

                              ▼

┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: Analysis & Report Writing                   │
│                                                                   │
│  Use generated plots in your report:                             │
│  • TTT plots → show convergence speed                            │
│  • Performance profiles → show solution quality                  │
│  • Convergence curves → show detailed behavior                   │
│  • Summary tables → statistical comparisons                      │
│                                                                   │
│  Custom analysis with results.parquet:                           │
│  ```python                                                       │
│  import pandas as pd                                             │
│  df = pd.read_parquet('results.parquet')                         │
│                                                                   │
│  # Statistical tests                                             │
│  # Gap analysis                                                  │
│  # Custom visualizations                                         │
│  ```                                                             │
└───────────────────────────────────────────────────────────────────┘
```

## File Organization

```
route-orchestrator/
├── main.py                    # Your existing experiment runner
├── parser.py                  # NEW: Log parsing script
├── plotter.py                 # NEW: Plot generation script
├── test_parser.py             # NEW: Validation script
├── requirements.txt           # NEW: Python dependencies
├── README.md                  # UPDATED: Added plotting section
├── PLOTTING_README.md         # NEW: Detailed plotting guide
├── PLOTTING_SUMMARY.md        # NEW: This summary document
│
├── logs/                      # Your existing logs
│   ├── alns_adaptive_sa/
│   │   ├── execution/
│   │   └── results/
│   ├── alns_greedy_lns/
│   ├── ts_tenure5/
│   └── ts_tenure0/
│
├── results.parquet            # GENERATED: Parsed data
│
└── plots/                     # GENERATED: All visualizations
    ├── ttt_*.png
    ├── performance_profile*.png
    ├── convergence_*.png
    └── summary_*.csv
```

## Quick Reference Commands

```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Verify setup
python test_parser.py

# Parse all logs
python parser.py

# Generate all plots
python plotter.py

# Custom analysis (Python)
python
>>> import pandas as pd
>>> df = pd.read_parquet('results.parquet')
>>> df.head()
```

## Data Flow

```
Raw Logs (TXT)
    ↓ [parser.py]
Structured Data (Parquet)
    ↓ [plotter.py]
Visualizations (PNG) + Statistics (CSV)
    ↓ [your analysis]
Report Figures & Tables
```

## What Each Tool Does

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `main.py` | Problem instances | Log files | Run experiments |
| `parser.py` | Log files | `results.parquet` | Consolidate data |
| `plotter.py` | `results.parquet` | Plots & CSVs | Visualize results |
| Your analysis | `results.parquet` | Custom insights | Statistical tests |
