# Plotting System Improvements

## Summary of Changes

I've improved the plotting system with **combined visualizations** and **fixed CSV formatting** to make your analysis more efficient and report-ready.

---

## ✨ New Combined Plots

### 1. **Combined TTT Plot by Size** (`ttt_combined_by_size.png`)
- **Format**: 2×2 grid showing all 4 customer sizes (100, 400, 800, 1000)
- **Each subplot**: Aggregates all categories for that size
- **Benefit**: See scalability trends at a glance - compare algorithm performance across problem sizes in one figure
- **Use in reports**: Perfect for showing how algorithms scale with problem complexity

### 2. **Combined TTT Plot by Category** (`ttt_combined_by_category.png`)
- **Format**: 2×3 grid showing all 6 problem categories (C1, C2, R1, R2, RC1, RC2)
- **Each subplot**: Aggregates all sizes for that category
- **Benefit**: Compare algorithm behavior across different problem characteristics
- **Use in reports**: Demonstrates robustness across diverse problem types

### 3. **Combined Performance Profile by Size** (`performance_profile_combined_by_size.png`)
- **Format**: 2×2 grid showing all 4 customer sizes
- **Each subplot**: Performance profile for that size
- **Benefit**: Compare solution quality consistency across scales
- **Use in reports**: Shows which algorithms maintain quality as problems grow

---

## 🔧 Fixed CSV Summaries

### Before (Misaligned)
```csv
,final_loss,final_loss,final_loss,final_loss,final_time,final_time,final_time,final_time,instance
,mean,std,min,max,mean,std,min,max,count
algorithm,,,,,,,,,
alns_adaptive_sa,108411.26,93826.03,1163.85,325099.86,506.36,133.47,0.0,621.13,236
```

### After (Clean & Aligned)
```csv
Algorithm,Instances,Avg_Loss,Std_Loss,Min_Loss,Max_Loss,Avg_Time,Std_Time
ALNS-SA,236,108411.26,93826.03,1163.85,325099.86,506.36,133.47
ALNS-Greedy,236,94222.45,82125.68,1086.32,302194.14,527.18,129.53
```

**Improvements**:
- ✅ Clear column headers with descriptive names
- ✅ Proper alignment (no multi-index confusion)
- ✅ Human-readable algorithm labels (ALNS-SA instead of alns_adaptive_sa)
- ✅ Consistent 2-decimal precision
- ✅ Ready to import into Excel/LaTeX/reports

---

## 📊 Complete Plot Inventory

### Individual Plots (38 files)
- **24 TTT plots**: One per category×size combination (e.g., `ttt_C1_100.png`)
- **11 Performance profiles**: Overall + 4 by size + 6 by category
- **3 Convergence examples**: Sample solution quality over time

### Combined Plots (3 files) - NEW!
- **`ttt_combined_by_size.png`**: All sizes in one 2×2 figure
- **`ttt_combined_by_category.png`**: All categories in one 2×3 figure
- **`performance_profile_combined_by_size.png`**: All sizes in one 2×2 figure

### Summary Data (3 CSV files)
- **`summary_overall.csv`**: Aggregate statistics across all 236 instances
- **`summary_by_size.csv`**: Performance broken down by customer size
- **`summary_by_category.csv`**: Performance broken down by problem category

---

## 📝 Recommended Figures for Your Report

### Minimal Set (4 figures)
1. **`ttt_combined_by_size.png`** - Shows convergence speed across all problem scales
2. **`performance_profile_combined_by_size.png`** - Shows solution quality consistency
3. **`performance_profile.png`** - Overall comparison (all 236 instances)
4. **`summary_overall.csv`** - Statistical summary table

### Extended Set (add 2-3 more)
5. **`ttt_combined_by_category.png`** - Shows algorithm behavior across problem types
6. **`convergence_C1_10_1.png`** - Example of how solutions improve over time
7. **Individual TTT plots** - For detailed analysis of specific problem classes

---

## 🎯 Key Insights from the Data

Based on the summary statistics:

### Overall Winner: **ALNS-Greedy**
- Found best solution on **183 out of 236 instances** (77.5%)
- Average loss: 94,222 (lowest among all algorithms)
- Slightly slower than ALNS-SA (527s vs 506s average)

### Second Best: **ALNS-SA**
- Found best solution on **47 instances** (19.9%)
- Faster convergence (506s average)
- Good balance of speed and quality

### Tabu Search Performance
- **TS algorithms** (both tenure=0 and tenure=5) show similar results
- Higher average loss (129,000+) compared to ALNS variants
- Found best solution on only 6 instances total

### Scalability Insights
- **100 customers**: All algorithms perform similarly (~3,300-3,500 avg loss)
- **1000 customers**: Performance gap widens significantly
  - ALNS-Greedy: 198,165
  - ALNS-SA: 230,965
  - TS: 272,000+

---

## 🚀 Quick Usage

### One-Time Setup
```bash
./setup_and_plot.sh
```

### Re-generate Plots Later
```bash
source venv/bin/activate
python plotter.py
```

### Custom Analysis
```python
import pandas as pd

# Load parsed data
df = pd.read_parquet('results.parquet')

# Example: Find instances where TS outperformed ALNS
ts_best = df[df['algorithm'] == 'ts_tenure5']
alns_best = df[df['algorithm'] == 'alns_greedy_lns']
merged = pd.merge(ts_best, alns_best, on='instance', suffixes=('_ts', '_alns'))
ts_wins = merged[merged['final_loss_ts'] < merged['final_loss_alns']]
print(f"TS won on {len(ts_wins)} instances:")
print(ts_wins[['instance', 'final_loss_ts', 'final_loss_alns']])
```

---

## 📁 File Structure

```
plots/
├── ttt_combined_by_size.png              ← NEW: All sizes in one figure
├── ttt_combined_by_category.png          ← NEW: All categories in one figure
├── performance_profile_combined_by_size.png  ← NEW: All sizes in one figure
├── performance_profile.png               ← Overall comparison
├── ttt_C1_100.png ... ttt_RC2_1000.png  ← 24 individual TTT plots
├── performance_profile_*.png             ← 11 individual profiles
├── convergence_*.png                     ← 3 example convergence curves
├── summary_overall.csv                   ← FIXED: Clean CSV format
├── summary_by_size.csv                   ← FIXED: Clean CSV format
└── summary_by_category.csv               ← FIXED: Clean CSV format
```

---

## 🔄 Python Environment

The system now uses a **local Python 3.14 virtual environment**:

- **Virtual environment**: `venv/` directory
- **Python version**: 3.14.0
- **Dependencies**: Installed via `pip install -r requirements.txt`
- **Isolation**: No conflicts with system Python packages

### Benefits
- ✅ Reproducible environment
- ✅ No system package conflicts
- ✅ Easy to share (just include requirements.txt)
- ✅ Consistent across different machines

---

## 📖 Additional Resources

- **PLOTTING_README.md**: Comprehensive guide with interpretation tips
- **PLOTTING_SUMMARY.md**: Quick reference and feature overview
- **WORKFLOW.md**: Visual workflow diagram
- **quick_analysis.sh**: Automated pipeline using venv

---

**Total Files Generated**: 41 plots + 3 CSVs = 44 files
**Storage**: ~15-20 MB (high-resolution PNG at 300 DPI)
**Recommended for Reports**: 4-7 key figures (see above)
