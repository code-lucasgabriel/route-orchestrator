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
│                   STEP 3: Generate Plots (v7)                    │
│                                                                   │
│  $ python plotter.py                                             │
│                                                                   │
│  Creates plots/ directory with publication-defining visuals:     │
│                                                                   │
│  📊 TTT Plots (4 files: 2 PNG + 2 PDF)                           │
│     ttt_combined_by_size.png/.pdf                                │
│     ttt_combined_by_category.png/.pdf                            │
│     • Step-function ECDFs (statistical precision)                │
│     • 4-color palette (perceptual clarity)                       │
│     • Automatic label collision prevention ⭐ NEW               │
│     • Success rates annotated (e.g., "25.6%", "0.0%")            │
│     • Legend in right margin (collision-free) ⭐ NEW            │
│                                                                   │
│  📈 Performance Profiles (4 files: 2 PNG + 2 PDF)                │
│     performance_profile_combined_by_size.png/.pdf                │
│     performance_profile_overall.png/.pdf                         │
│     • 4 distinct colors (no overlapping blue lines) ⭐ NEW      │
│     • Elegant leader-line annotations ⭐ NEW                    │
│     • Win rate annotated (e.g., "77.5%")                         │
│     • Legend in right margin (collision-free) ⭐ NEW            │
│                                                                   │
│  📉 Convergence Examples (6 files: 3 PNG + 3 PDF)                │
│     convergence_C1_10_1.png/.pdf                                 │
│     convergence_C1_10_10.png/.pdf                                │
│     convergence_C1_10_2.png/.pdf                                 │
│     • Target benchmark line (from TTT) ⭐ CRITICAL NEW          │
│     • Shaded phase regions (background) ⭐ NEW                  │
│     • 4-color consistency ⭐ NEW                                │
│     • Even marker spacing (markevery=0.1)                        │
│                                                                   │
│  📄 Summary Statistics (3 CSV files)                             │
│     summary_overall.csv                                          │
│     summary_by_size.csv                                          │
│     summary_by_category.csv                                      │
│                                                                   │
│  Publication-Defining Features (v7):                             │
│  ━━━ PILLAR I: NARRATIVE INTEGRATION ━━━                         │
│  ✓ Target benchmark on convergence (explains TTT results)        │
│  ━━━ PILLAR II: PERCEPTUAL-FIRST ENCODING ━━━                    │
│  ✓ 4-color distinct palette (eliminates ambiguity)               │
│  ✓ Colorblind-safe (IBM palette)                                 │
│  ━━━ PILLAR III: AUTOMATED AESTHETIC REFINEMENT ━━━              │
│  ✓ Automatic label collision prevention                          │
│  ✓ Elegant leader-line annotations                               │
│  ✓ Collision-free legend placement                               │
│  ✓ Non-intrusive phase regions                                   │
│                                                                   │
│  Plus all v6 foundations:                                        │
│  ✓ Statistical honesty (TS 0% success shown)                     │
│  ✓ Statistical precision (Step-function ECDFs)                   │
│  ✓ Professional aesthetics (Serif fonts, minimal grids)          │
│  ✓ Vector output (PDF files for manuscript)                      │
│                                                                   │
│  Total: 17 files (7 PNG + 7 PDF + 3 CSV)                         │
└───────────────────────────────────────────────────────────────────┘
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
