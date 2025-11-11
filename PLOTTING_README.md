# Performance Analysis & Plotting Guide

This guide explains how to analyze and visualize the results from your HFFVRPTW solver experiments.

## Overview

The plotting system consists of two main scripts:

1. **`parser.py`** - Converts raw text logs into an efficient Parquet file for analysis
2. **`plotter.py`** - Generates comprehensive performance visualizations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Parse the Logs

First, convert all your raw log files into a consolidated analysis file:

```bash
python parser.py
```

This will:
- Scan all log files in `logs/*/results/*.txt`
- Extract convergence histories from `logs/*/execution/*.txt`
- Parse instance metadata (category, customer size)
- Save everything to `results.parquet`

**Expected output:**
```
Starting log parsing from root directory: logs
Found 944 log files to parse.
Parsing logs: 100%|████████████| 944/944
Successfully parsed 944 records.
Master analysis file saved to: results.parquet
```

### 3. Generate Plots

Now create all the performance visualizations:

```bash
python plotter.py
```

This will generate:
- **TTT Plots** - Time-to-target curves for each category/size combination
- **Performance Profiles** - Algorithm comparison across all instances
- **Convergence Examples** - Solution quality over time for sample instances
- **Summary Statistics** - CSV tables with detailed performance metrics

**All plots are saved to the `plots/` directory.**

---

## Understanding the Data Schema

The parsed data (`results.parquet`) uses an efficient "tidy data" format:

| Column | Type | Description |
|--------|------|-------------|
| `algorithm` | str | Solver name (`alns_adaptive_sa`, `alns_greedy_lns`, `ts_tenure5`, `ts_tenure0`) |
| `instance` | str | Instance identifier (e.g., `C1_1_01`) |
| `category` | str | Problem category (`C1`, `C2`, `R1`, `R2`, `RC1`, `RC2`) |
| `cust_size` | int | Number of customers (100, 400, 800, 1000) |
| `final_loss` | float | Best objective value found |
| `final_time` | float | Time (seconds) when best solution was found |
| `loss_history` | np.array | All recorded objective values during search |
| `time_history` | np.array | Timestamps for each `loss_history` entry |
| `final_solution` | dict | Fleet routes (`{'A': [[0,1,2,0]], 'B': [[0,3,4,0]]}`) |

**Why Parquet?**
- 10-100x faster to read than CSV
- Column-oriented storage is perfect for analytical queries
- Native support for complex types (numpy arrays, nested dictionaries)
- Compressed format saves disk space

**Why numpy arrays for history?**
- Enables vectorized operations (e.g., `np.where(loss_history <= target)`)
- Avoids slow `pandas.apply()` loops
- Minimal memory footprint

---

## Understanding the Plots

### 1. Time-To-Target (TTT) Plots

**Filename:** `plots/ttt_<category>_<size>.png`

**What it shows:**
- X-axis: Time in seconds (log scale)
- Y-axis: Proportion of instances where the algorithm reached the target
- One curve per algorithm

**How to interpret:**
- **Higher/more left** curve = better (reaches target faster on more instances)
- Target is set at the **95th percentile** of best solutions (aggressive target)
- Aggregates multiple instances in the same category/size

**Example:**
```
TTT Plot: C1 (100 customers)
Target = 95th percentile (9 instances)
```
If ALNS-SA reaches y=0.8 at x=100s, it means:
- On 80% of C1_100 instances, ALNS-SA found a solution ≤ target within 100 seconds

**Use case:** Show which algorithm converges fastest to high-quality solutions.

---

### 2. Performance Profiles

**Filename:** `plots/performance_profile*.png`

**What it shows:**
- X-axis: Performance ratio τ (best_solution / algorithm_solution)
- Y-axis: Probability that algorithm achieves ratio ≥ τ
- One curve per algorithm

**How to interpret:**
- **Higher curve** = better overall performance
- τ = 1.0 means the algorithm found the best-known solution
- τ = 0.98 means the algorithm's solution is 98% of the best (2% worse)
- The curve shows how often each algorithm is within a given quality threshold

**Example:**
If ALNS-SA has P(τ ≥ 0.99) = 0.6, it means:
- On 60% of instances, ALNS-SA found a solution within 1% of the best

**Generated variants:**
1. `performance_profile.png` - All 236 instances combined
2. `performance_profile_<size>.png` - By customer size (4 plots)
3. `performance_profile_<category>.png` - By problem category (6 plots)

**Use case:** Show solution quality consistency across diverse problem instances.

---

### 3. Convergence Plots (Examples)

**Filename:** `plots/convergence_<instance>.png`

**What it shows:**
- X-axis: Time (log scale)
- Y-axis: Objective value (lower is better)
- One curve per algorithm showing improvement over time

**How to interpret:**
- Steeper initial drop = good initial heuristic
- Early plateau = premature convergence
- Late improvements = effective diversification

**Use case:** Detailed analysis of algorithm behavior on specific instances.

---

### 4. Summary Statistics

**Files:** `plots/summary_*.csv`

Three CSV files with detailed performance metrics:

1. **`summary_overall.csv`** - Mean, std, min, max per algorithm
2. **`summary_by_size.csv`** - Performance broken down by customer size
3. **`summary_by_category.csv`** - Performance broken down by category

**Use case:** Generate tables for your report/paper.

---

## Customization Guide

### Change Target Percentile for TTT Plots

In `plotter.py`, modify the `generate_ttt_plot_by_category()` call:

```python
# Use 90th percentile (easier target)
generate_ttt_plot_by_category(df, category, cust_size, output_dir, target_percentile=90)

# Use 100th percentile (best-known solution only)
generate_ttt_plot_by_category(df, category, cust_size, output_dir, target_percentile=100)
```

### Generate Plots for Specific Subsets

```python
# Only C1 instances
df_c1 = df[df['category'] == 'C1']
generate_all_performance_profiles(df_c1, output_dir)

# Only 1000-customer instances
df_large = df[df['cust_size'] == 1000]
generate_all_ttt_plots(df_large, output_dir)
```

### Change Plot Styling

At the top of `plotter.py`:

```python
# Use different matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')  # or 'ggplot', 'bmh', 'dark_background'

# Change color palette
sns.set_palette("Set2")  # or "tab10", "colorblind", "pastel"

# Customize algorithm colors
ALGORITHM_COLORS = {
    'alns_adaptive_sa': '#FF0000',  # Pure red
    'alns_greedy_lns': '#0000FF',   # Pure blue
    ...
}
```

### Add More Plot Types

Example: Add a box plot comparing final costs

```python
def generate_cost_boxplot(df: pd.DataFrame, output_dir: Path):
    """Box plot of final costs by algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_plot = df.copy()
    df_plot['algorithm'] = df_plot['algorithm'].map(ALGORITHM_LABELS)
    
    sns.boxplot(data=df_plot, x='algorithm', y='final_loss', ax=ax)
    ax.set_ylabel('Final Objective Value')
    ax.set_title('Solution Quality Distribution by Algorithm')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_cost.png', dpi=300)
    plt.close()

# Call it in main()
generate_cost_boxplot(df, output_dir)
```

---

## Log File Format Reference

### Results File (`logs/*/results/<instance>.txt`)

```
[<best_cost>, <time_found>]
<fleet_A>: [[route1], [route2], ...]
<fleet_B>: [[route3], ...]
```

**Example:**
```
[2251.21, 593.58]
A: [[0, 20, 42, 41, 0], [0, 13, 17, 15, 0]]
B: [[0, 5, 3, 94, 92, 93, 0]]
```

### Execution File (`logs/*/execution/<instance>.txt`)

```
[<cost>, <time>]
<fleet_A>: [[...]]
...
[<cost>, <time>]
<fleet_A>: [[...]]
...
```

Each improvement is logged as `[cost, time]` followed by the solution.

---

## Typical Workflow for Your Report

### 1. Run Experiments
```bash
python main.py  # Run all solvers on all instances
```

### 2. Parse Results
```bash
python parser.py
```

### 3. Generate Plots
```bash
python plotter.py
```

### 4. Select Plots for Report

**Recommended plots:**

1. **Performance Profile (Overall)** - `performance_profile.png`
   - Shows which algorithm is most robust across all problem types
   
2. **Performance Profile by Size** - `performance_profile_<size>.png`
   - Shows how algorithms scale with problem size
   
3. **TTT Plots (Selected Categories)** - e.g., `ttt_C1_100.png`, `ttt_R1_400.png`
   - Shows convergence speed for representative problem types
   
4. **Convergence Examples** - `convergence_<instance>.png`
   - Detailed behavior on specific challenging instances

5. **Summary Tables** - Import CSV files into your LaTeX/Word document

### 5. Statistical Analysis

Use the Parquet file for custom analysis:

```python
import pandas as pd

df = pd.read_parquet('results.parquet')

# Wilcoxon signed-rank test for paired comparisons
from scipy.stats import wilcoxon

alns_sa = df[df['algorithm'] == 'alns_adaptive_sa'].sort_values('instance')
ts5 = df[df['algorithm'] == 'ts_tenure5'].sort_values('instance')

statistic, pvalue = wilcoxon(alns_sa['final_loss'], ts5['final_loss'])
print(f"p-value: {pvalue}")

# Average gap to best-known
for instance in df['instance'].unique():
    instance_df = df[df['instance'] == instance]
    best = instance_df['final_loss'].min()
    
    for alg in df['algorithm'].unique():
        alg_cost = instance_df[instance_df['algorithm'] == alg]['final_loss'].values[0]
        gap = 100 * (alg_cost - best) / best
        print(f"{instance} - {alg}: {gap:.2f}% gap")
```

---

## Troubleshooting

### "No module named 'pandas'"
```bash
pip install -r requirements.txt
```

### "File 'results.parquet' not found"
Run `parser.py` first:
```bash
python parser.py
```

### "No data for <category> with <size> customers"
Your logs might be incomplete. Check:
```bash
ls logs/*/results/*.txt | wc -l  # Should be ~944 files (236 instances × 4 algorithms)
```

### Plots look cluttered
- Reduce number of instances per plot by filtering
- Increase figure size in `plt.subplots(figsize=(width, height))`
- Use separate plots for each algorithm

### Memory issues with large datasets
The Parquet format is already very efficient. If needed:
- Process one customer size at a time
- Use `df = pd.read_parquet(file, columns=['algorithm', 'instance', 'final_loss'])` to load only needed columns

---

## Advanced: Custom Analysis Examples

### 1. Find instances where ALNS beats TS

```python
df = pd.read_parquet('results.parquet')

for instance in df['instance'].unique():
    instance_df = df[df['instance'] == instance]
    
    alns_cost = instance_df[instance_df['algorithm'] == 'alns_adaptive_sa']['final_loss'].values[0]
    ts_cost = instance_df[instance_df['algorithm'] == 'ts_tenure5']['final_loss'].values[0]
    
    if alns_cost < ts_cost:
        print(f"{instance}: ALNS wins ({alns_cost:.2f} vs {ts_cost:.2f})")
```

### 2. Average convergence time to 105% of final solution

```python
for alg in df['algorithm'].unique():
    alg_data = df[df['algorithm'] == alg]
    
    times = []
    for _, row in alg_data.iterrows():
        target = 1.05 * row['final_loss']
        reached = np.where(row['loss_history'] <= target)[0]
        if len(reached) > 0:
            times.append(row['time_history'][reached[0]])
    
    print(f"{alg}: Avg time to 105% of final = {np.mean(times):.2f}s")
```

### 3. Instances where all algorithms struggle (high variance)

```python
instance_variance = df.groupby('instance')['final_loss'].var()
hard_instances = instance_variance.nlargest(10)
print("Most challenging instances:")
print(hard_instances)
```

---

## Questions?

Check the inline comments in `parser.py` and `plotter.py` for implementation details.

For algorithm-specific questions, see the main `README.md`.
