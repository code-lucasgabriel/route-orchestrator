# Summary: Performance Analysis System Created

## What Was Created

I've built a comprehensive performance analysis and plotting system for your HFFVRPTW solver experiments. Here's what you now have:

### 📁 New Files Created

1. **`parser.py`** (215 lines)
   - Converts raw text logs into efficient Parquet format
   - Extracts convergence histories from execution logs
   - Parses instance metadata (category, customer size)
   - Outputs: `results.parquet` (ready for analysis)

2. **`plotter.py`** (414 lines)
   - Generates Time-To-Target (TTT) plots
   - Creates Performance Profile plots
   - Produces convergence curve examples
   - Exports summary statistics tables
   - Outputs: `plots/` directory with publication-quality figures

3. **`PLOTTING_README.md`** (Comprehensive guide)
   - Installation instructions
   - Plot interpretation guide
   - Customization examples
   - Advanced analysis recipes
   - Troubleshooting tips

4. **`test_parser.py`** (Quick verification script)
   - Tests if dependencies are installed
   - Verifies log file structure
   - Validates parser on sample file

5. **`requirements.txt`** (Dependencies)
   ```
   pandas>=2.0.0
   numpy>=1.24.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   pyarrow>=12.0.0
   tqdm>=4.65.0
   ```

6. **Updated `README.md`**
   - Added "Performance Analysis & Plotting" section
   - Integrated with existing documentation

---

## 🚀 How to Use

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Parse Your Logs

```bash
python parser.py
```

**What it does:**
- Scans `logs/*/results/*.txt` (final solutions)
- Scans `logs/*/execution/*.txt` (convergence histories)
- Extracts: algorithm, instance, category, size, costs, times, routes
- Saves: `results.parquet` (efficient columnar format)

**Expected output:**
```
Starting log parsing from root directory: logs
Found 944 log files to parse.
Parsing logs: 100%|████████████████| 944/944
Successfully parsed 944 records.
Master analysis file saved to: results.parquet
```

### Step 3: Generate Plots

```bash
python plotter.py
```

**What it creates:**
- **24 TTT plots** - One per category/size combination
  - Example: `plots/ttt_C1_100.png`
- **11 Performance profiles**
  - Overall: `plots/performance_profile.png`
  - By size (4): `plots/performance_profile_<100|400|800|1000>.png`
  - By category (6): `plots/performance_profile_<C1|C2|R1|R2|RC1|RC2>.png`
- **3 Convergence examples** - `plots/convergence_<instance>.png`
- **3 Summary CSVs** - `plots/summary_*.csv`

---

## 📊 Understanding the Plots

### Time-To-Target (TTT) Plots

**File:** `plots/ttt_C1_100.png`

```
Y-axis: Proportion of instances reaching target
X-axis: Time (seconds, log scale)
Curves: One per algorithm
```

**Interpretation:**
- **Higher/left curve = better** (faster convergence)
- Target = 95th percentile of best solutions (aggressive)
- Shows: "On X% of instances, algorithm Y reached target in Z seconds"

**Use case:** Which algorithm converges fastest to high-quality solutions?

---

### Performance Profiles

**File:** `plots/performance_profile.png`

```
Y-axis: P(ratio ≥ τ)
X-axis: Performance ratio τ = best_solution / algorithm_solution
Curves: One per algorithm
```

**Interpretation:**
- **Higher curve = better** (more consistent)
- τ = 1.0 means algorithm found best solution
- τ = 0.98 means solution is 98% of best (2% worse)
- Shows: "Algorithm X found solutions within Y% of best on Z% of instances"

**Use case:** Which algorithm is most robust across diverse problems?

---

## 🎯 Key Design Decisions

### Why Parquet Instead of CSV?

- **10-100x faster** to read/write
- **Column-oriented** storage (perfect for analytics)
- **Native support** for complex types (numpy arrays, nested dicts)
- **Compressed** (saves disk space)

### Why NumPy Arrays for History?

Your original idea was brilliant. Instead of storing convergence history as strings or Python lists:

```python
# ❌ BAD: String in DataFrame
loss_history: "[[3763.29, 0.0], [2967.23, 51.92], ...]"

# ❌ BAD: Python list in DataFrame  
loss_history: [[3763.29, 0.0], [2967.23, 51.92], ...]

# ✅ GOOD: Split into numpy arrays
loss_history: np.array([3763.29, 2967.23, ...])
time_history: np.array([0.0, 51.92, ...])
```

**Benefits:**
- Vectorized operations: `np.where(loss_history <= target)[0]` (10-100x faster)
- No slow `pandas.apply()` loops
- Minimal memory footprint
- Efficient I/O with Parquet

---

## 📈 Data Schema

The `results.parquet` file contains:

```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 944 entries, 0 to 943
Data columns (total 9 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   algorithm        944 non-null    object 
 1   instance         944 non-null    object 
 2   category         944 non-null    object 
 3   cust_size        944 non-null    int64  
 4   final_loss       944 non-null    float64
 5   final_time       944 non-null    float64
 6   loss_history     944 non-null    object  # numpy.ndarray
 7   time_history     944 non-null    object  # numpy.ndarray
 8   final_solution   944 non-null    object  # dict
```

**Example row:**
```python
row = df.iloc[0]
print(row['algorithm'])      # 'alns_adaptive_sa'
print(row['instance'])       # 'C1_1_01'
print(row['category'])       # 'C1'
print(row['cust_size'])      # 100
print(row['final_loss'])     # 2251.21
print(row['final_time'])     # 593.58
print(row['loss_history'])   # array([3763.29, 2967.23, ..., 2251.21])
print(row['time_history'])   # array([0.0, 51.92, ..., 593.58])
print(row['final_solution']) # {'A': [[0, 20, 42, ...], ...], 'B': [...]}
```

---

## 🔧 Customization Examples

### Change TTT Target Percentile

In `plotter.py`, line ~95:

```python
# Current: 95th percentile (aggressive target)
generate_ttt_plot_by_category(df, category, cust_size, output_dir, target_percentile=95)

# Option 1: 90th percentile (easier target)
generate_ttt_plot_by_category(df, category, cust_size, output_dir, target_percentile=90)

# Option 2: 100th percentile (best-known only)
generate_ttt_plot_by_category(df, category, cust_size, output_dir, target_percentile=100)
```

### Generate Plots for Specific Subset

```python
# Only large instances
df_large = df[df['cust_size'] >= 400]
generate_all_ttt_plots(df_large, output_dir)

# Only C-type instances
df_c = df[df['category'].str.startswith('C')]
generate_all_performance_profiles(df_c, output_dir)
```

### Custom Statistical Analysis

```python
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

df = pd.read_parquet('results.parquet')

# Pairwise Wilcoxon tests
for alg1 in ['alns_adaptive_sa', 'ts_tenure5']:
    for alg2 in ['alns_greedy_lns', 'ts_tenure0']:
        data1 = df[df['algorithm'] == alg1].sort_values('instance')
        data2 = df[df['algorithm'] == alg2].sort_values('instance')
        stat, pval = wilcoxon(data1['final_loss'], data2['final_loss'])
        print(f"{alg1} vs {alg2}: p = {pval:.4f}")

# Friedman test (4 algorithms, paired by instance)
pivot = df.pivot(index='instance', columns='algorithm', values='final_loss')
stat, pval = friedmanchisquare(*[pivot[alg] for alg in pivot.columns])
print(f"Friedman test: χ² = {stat:.2f}, p = {pval:.4f}")
```

---

## 📝 For Your Report

### Recommended Figures

**Figure 1: Overall Performance**
- `performance_profile.png` - Shows robustness across all 236 instances

**Figure 2: Scalability**
- `performance_profile_100.png` vs `performance_profile_1000.png`
- Shows how algorithms scale with problem size

**Figure 3: Convergence Speed**
- `ttt_C1_100.png` and `ttt_R1_400.png`
- Shows different convergence behaviors on clustered vs random instances

**Figure 4: Detailed Analysis**
- `convergence_<interesting_instance>.png`
- Shows specific algorithm behavior

**Table 1: Summary Statistics**
- Import `summary_overall.csv` into LaTeX/Word

**Table 2: Win Counts**
```python
best_per_instance = df.loc[df.groupby('instance')['final_loss'].idxmin()]
win_counts = best_per_instance['algorithm'].value_counts()
```

---

## 🎨 Plot Styling

All plots use:
- **Seaborn darkgrid** style
- **Consistent colors:**
  - ALNS-SA: Red (#E74C3C)
  - ALNS-Greedy: Blue (#3498DB)
  - TS(5): Green (#2ECC71)
  - TS(0): Orange (#F39C12)
- **300 DPI** resolution (publication-quality)
- **Clear labels** and legends

To change styling, edit top of `plotter.py`:

```python
plt.style.use('seaborn-v0_8-whitegrid')  # Different style
sns.set_palette("colorblind")            # Color-blind friendly
```

---

## ✅ Validation Checklist

Before running analysis:

- [ ] All experiments completed (944 result files)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Logs in correct format (verified with `python test_parser.py`)

After running analysis:

- [ ] `results.parquet` created (~2-5 MB)
- [ ] `plots/` directory contains 38 files
- [ ] No error messages during parsing
- [ ] Plots look reasonable (curves are smooth, no outliers)

---

## 🐛 Troubleshooting

### "No module named 'pandas'"

```bash
pip install -r requirements.txt
```

### "File 'results.parquet' not found"

Run parser first:
```bash
python parser.py
```

### "Warning: Skipping file, could not parse instance name"

Check your instance naming convention. Parser expects:
```
<CATEGORY>_<SCALE>_<ID>.txt
Examples: C1_1_01.txt, RC2_10_5.txt
```

Where SCALE maps to customer size:
- 1 → 100 customers
- 4 → 400 customers
- 8 → 800 customers
- 10 → 1000 customers

### "No data for <category> with <size> customers"

Your logs might be incomplete. Check:
```bash
ls logs/*/results/*.txt | wc -l  # Should be 944
```

---

## 🎓 Next Steps

1. **Run the analysis:**
   ```bash
   python parser.py
   python plotter.py
   ```

2. **Review plots** in `plots/` directory

3. **Select best plots** for your report

4. **Generate summary tables:**
   ```python
   import pandas as pd
   df = pd.read_parquet('results.parquet')
   
   # Best algorithm per instance
   best = df.loc[df.groupby('instance')['final_loss'].idxmin()]
   print(best['algorithm'].value_counts())
   
   # Average gaps
   for instance in df['instance'].unique():
       instance_df = df[df['instance'] == instance]
       best_val = instance_df['final_loss'].min()
       for alg in df['algorithm'].unique():
           alg_val = instance_df[instance_df['algorithm'] == alg]['final_loss'].values[0]
           gap = 100 * (alg_val - best_val) / best_val
           print(f"{instance},{alg},{gap:.2f}")
   ```

5. **Statistical tests** (see customization examples above)

---

## 📚 Additional Resources

- **PLOTTING_README.md** - Detailed guide with more examples
- **main.py** - Your existing experiment runner
- **parser.py** - Comments explain each parsing step
- **plotter.py** - Inline documentation for each plot type

---

## Questions?

The code is well-commented. If you need to:
- **Add new plot types** → See `plotter.py` functions
- **Modify data schema** → See `parser.py` `parse_log_file()`
- **Custom analysis** → Load `results.parquet` and use pandas/numpy

Good luck with your report! 🚀
