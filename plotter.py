"""
plotter.py

A comprehensive plotting suite for analyzing HFFVRPTW solver performance.

Generates:
1. TTT (Time-To-Target) plots - showing probability of reaching target solutions over time
2. Performance Profile plots - comparing algorithm efficiency across all instances

The plots are organized by:
- Customer size (100, 400, 800, 1000)
- Problem category (C1, C2, R1, R2, RC1, RC2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = 'results.parquet'
OUTPUT_DIR = 'plots'

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Algorithm names for better labels
ALGORITHM_LABELS = {
    'alns_adaptive_sa': 'ALNS-SA',
    'alns_greedy_lns': 'ALNS-Greedy',
    'ts_tenure5': 'TS (tenure=5)',
    'ts_tenure0': 'TS (tenure=0)'
}

# Colors for consistency across plots
ALGORITHM_COLORS = {
    'alns_adaptive_sa': '#E74C3C',  # Red
    'alns_greedy_lns': '#3498DB',   # Blue
    'ts_tenure5': '#2ECC71',        # Green
    'ts_tenure0': '#F39C12'         # Orange
}


def load_data() -> pd.DataFrame:
    """Load the parsed results from Parquet file."""
    try:
        df = pd.read_parquet(DATA_FILE)
        print(f"Loaded {len(df)} records from {DATA_FILE}")
        print(f"Algorithms: {df['algorithm'].unique()}")
        print(f"Customer sizes: {sorted(df['cust_size'].unique())}")
        print(f"Categories: {sorted(df['category'].unique())}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File '{DATA_FILE}' not found.")
        print("Please run parser.py first to generate the data file.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        exit(1)


def compute_target_value(df: pd.DataFrame, instance: str, percentile: float = 95) -> float:
    """
    Compute target value for TTT plot.
    Uses the 95th percentile of best solutions across all algorithms for this instance.
    
    Args:
        df: DataFrame with results
        instance: Instance name
        percentile: Percentile to use for target (default 95, can use 90 or 100 for best-known)
    
    Returns:
        Target objective value
    """
    instance_data = df[df['instance'] == instance]
    if len(instance_data) == 0:
        return None
    
    # Use percentile of final losses
    target = np.percentile(instance_data['final_loss'].values, percentile)
    return target


def generate_ttt_plot_by_category(df: pd.DataFrame, category: str, cust_size: int, 
                                   output_dir: Path, target_percentile: float = 95):
    """
    Generate a single TTT plot for all instances in a category/size combination.
    
    This aggregates multiple instances to show overall algorithm performance.
    """
    # Filter data
    subset = df[(df['category'] == category) & (df['cust_size'] == cust_size)]
    
    if len(subset) == 0:
        print(f"  No data for {category} with {cust_size} customers")
        return
    
    instances = subset['instance'].unique()
    print(f"  Generating TTT plot for {category}_{cust_size} ({len(instances)} instances)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    algorithms = subset['algorithm'].unique()
    
    # For each algorithm, collect all time-to-target values across instances
    for alg in sorted(algorithms):
        alg_data = subset[subset['algorithm'] == alg]
        
        all_times_to_target = []
        
        for instance in instances:
            instance_df = df[df['instance'] == instance]
            target = compute_target_value(instance_df, instance, percentile=target_percentile)
            
            if target is None:
                continue
            
            # Get this algorithm's run for this instance
            run_data = alg_data[alg_data['instance'] == instance]
            
            if len(run_data) == 0:
                continue
            
            row = run_data.iloc[0]
            loss_hist = row['loss_history']
            time_hist = row['time_history']
            
            # Find when target was reached
            reached_indices = np.where(loss_hist <= target)[0]
            
            if len(reached_indices) > 0:
                time_to_target = time_hist[reached_indices[0]]
                all_times_to_target.append(time_to_target)
        
        if len(all_times_to_target) == 0:
            print(f"    Warning: {alg} never reached target for any instance in {category}_{cust_size}")
            continue
        
        # Sort times and compute empirical CDF
        sorted_times = np.sort(all_times_to_target)
        n = len(sorted_times)
        probabilities = (np.arange(1, n + 1)) / len(instances)  # Normalize by total instances
        
        # Add starting point at (0, 0)
        plot_times = np.insert(sorted_times, 0, 0)
        plot_probs = np.insert(probabilities, 0, 0)
        
        # Plot
        label = ALGORITHM_LABELS.get(alg, alg)
        color = ALGORITHM_COLORS.get(alg, None)
        ax.plot(plot_times, plot_probs, marker='o', markersize=4, 
                linestyle='-', linewidth=2, label=label, color=color, alpha=0.8)
    
    # Formatting
    ax.set_title(f'Time-To-Target Plot: {category} ({cust_size} customers)\n'
                 f'Target = {target_percentile}th percentile of best solutions ({len(instances)} instances)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Proportion of Instances Reaching Target', fontsize=12)
    ax.set_xscale('log')
    ax.set_xlim(left=0.1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", linestyle=':', linewidth=0.6, alpha=0.7)
    ax.legend(loc='lower right', fontsize=11)
    
    # Save
    filename = f"ttt_{category}_{cust_size}.png"
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {filename}")


def generate_all_ttt_plots(df: pd.DataFrame, output_dir: Path):
    """Generate TTT plots for all category/size combinations."""
    print("\n" + "="*80)
    print("Generating Time-To-Target (TTT) Plots")
    print("="*80)
    
    categories = sorted(df['category'].unique())
    cust_sizes = sorted(df['cust_size'].unique())
    
    for cust_size in cust_sizes:
        print(f"\nCustomer size: {cust_size}")
        for category in categories:
            generate_ttt_plot_by_category(df, category, cust_size, output_dir)


def generate_performance_profile(df: pd.DataFrame, output_dir: Path, 
                                 filter_category: str = None, 
                                 filter_cust_size: int = None):
    """
    Generate performance profile comparing all algorithms.
    
    Performance profile shows P(ratio <= tau) where ratio = best_overall / algorithm_result.
    A higher curve is better (algorithm is more often close to best).
    
    Args:
        df: Results dataframe
        output_dir: Output directory
        filter_category: If provided, only use this category
        filter_cust_size: If provided, only use this customer size
    """
    # Filter data if requested
    subset = df.copy()
    title_suffix = ""
    file_suffix = ""
    
    if filter_category:
        subset = subset[subset['category'] == filter_category]
        title_suffix += f" - {filter_category}"
        file_suffix += f"_{filter_category}"
    
    if filter_cust_size:
        subset = subset[subset['cust_size'] == filter_cust_size]
        title_suffix += f" ({filter_cust_size} customers)"
        file_suffix += f"_{filter_cust_size}"
    
    if len(subset) == 0:
        print(f"  No data for performance profile{title_suffix}")
        return
    
    print(f"  Generating performance profile{title_suffix} ({len(subset)} runs)")
    
    # For each instance, find the best solution across all algorithms
    best_per_instance = subset.groupby('instance')['final_loss'].min().reset_index()
    best_per_instance = best_per_instance.rename(columns={'final_loss': 'best_overall'})
    
    # Merge with original data
    merged = pd.merge(subset[['instance', 'algorithm', 'final_loss']], 
                     best_per_instance, on='instance')
    
    # Compute performance ratio (best / algorithm_result)
    # Higher is better, 1.0 means algorithm found the best solution
    merged['ratio'] = merged['best_overall'] / merged['final_loss']
    
    # Handle edge cases
    merged['ratio'] = merged['ratio'].replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=['ratio'])
    
    # Compute tau range
    max_ratio = merged['ratio'].max()
    tau_range = np.linspace(0.95, max(1.0, min(max_ratio, 1.05)), 200)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    algorithms = sorted(merged['algorithm'].unique())
    
    for alg in algorithms:
        alg_ratios = merged[merged['algorithm'] == alg]['ratio'].values
        
        # Compute P(ratio >= tau) for each tau
        # Note: We want ratio >= tau because higher ratio is better
        probabilities = [(alg_ratios >= tau).mean() for tau in tau_range]
        
        label = ALGORITHM_LABELS.get(alg, alg)
        color = ALGORITHM_COLORS.get(alg, None)
        ax.plot(tau_range, probabilities, linewidth=2.5, label=label, color=color, alpha=0.9)
    
    # Formatting
    ax.set_title(f'Performance Profile{title_suffix}\n'
                 f'({len(merged["instance"].unique())} instances)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Performance Ratio (τ = best_solution / algorithm_solution)', fontsize=12)
    ax.set_ylabel('P(performance ratio ≥ τ)', fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(tau_range.min(), tau_range.max())
    ax.set_ylim(0, 1.05)
    
    # Add vertical line at ratio=1.0 (optimal performance)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Optimal (τ=1.0)')
    
    # Save
    filename = f"performance_profile{file_suffix}.png"
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {filename}")


def generate_all_performance_profiles(df: pd.DataFrame, output_dir: Path):
    """Generate performance profiles at different aggregation levels."""
    print("\n" + "="*80)
    print("Generating Performance Profile Plots")
    print("="*80)
    
    # 1. Overall performance profile (all instances)
    print("\nOverall performance profile:")
    generate_performance_profile(df, output_dir)
    
    # 2. Performance profiles by customer size
    print("\nPerformance profiles by customer size:")
    for cust_size in sorted(df['cust_size'].unique()):
        generate_performance_profile(df, output_dir, filter_cust_size=cust_size)
    
    # 3. Performance profiles by category
    print("\nPerformance profiles by category:")
    for category in sorted(df['category'].unique()):
        generate_performance_profile(df, output_dir, filter_category=category)


def generate_combined_ttt_by_size(df: pd.DataFrame, output_dir: Path, target_percentile: float = 95):
    """
    Generate a 2x2 combined TTT plot showing all 4 customer sizes in one figure.
    Each subplot aggregates all categories for that size.
    """
    print("\n" + "="*80)
    print("GENERATING COMBINED TTT PLOT BY SIZE")
    print("="*80)
    
    sizes = sorted(df['cust_size'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, cust_size in enumerate(sizes):
        ax = axes[idx]
        subset = df[df['cust_size'] == cust_size]
        instances = subset['instance'].unique()
        
        print(f"  Processing {cust_size} customers ({len(instances)} instances)")
        
        algorithms = subset['algorithm'].unique()
        
        for alg in sorted(algorithms):
            alg_subset = subset[subset['algorithm'] == alg]
            all_ttt = []
            
            for instance in instances:
                inst_data = alg_subset[alg_subset['instance'] == instance].iloc[0]
                target = compute_target_value(df, instance, target_percentile)
                
                loss_hist = inst_data['loss_history']
                time_hist = inst_data['time_history']
                
                if len(loss_hist) > 0 and len(time_hist) > 0:
                    target_idx = np.where(loss_hist <= target)[0]
                    if len(target_idx) > 0:
                        ttt = time_hist[target_idx[0]]
                        all_ttt.append(ttt)
            
            if len(all_ttt) > 0:
                all_ttt_sorted = np.sort(all_ttt)
                n = len(all_ttt_sorted)
                proportions = np.arange(1, n + 1) / len(instances)
                
                ax.plot(all_ttt_sorted, proportions, 
                       label=ALGORITHM_LABELS.get(alg, alg),
                       color=ALGORITHM_COLORS.get(alg),
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        
        ax.set_title(f'{cust_size} Customers ({len(instances)} instances)', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Proportion Reaching Target', fontsize=11)
        ax.set_xscale('log')
        ax.set_xlim(left=0.1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", linestyle=':', linewidth=0.6, alpha=0.7)
        ax.legend(loc='lower right', fontsize=10)
    
    fig.suptitle(f'Time-To-Target Plots by Customer Size\n(Target = {target_percentile}th percentile)',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    filename = "ttt_combined_by_size.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def generate_combined_ttt_by_category(df: pd.DataFrame, output_dir: Path, target_percentile: float = 95):
    """
    Generate a 2x3 combined TTT plot showing all 6 categories in one figure.
    Each subplot aggregates all sizes for that category.
    """
    print("\n" + "="*80)
    print("GENERATING COMBINED TTT PLOT BY CATEGORY")
    print("="*80)
    
    categories = sorted(df['category'].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        subset = df[df['category'] == category]
        instances = subset['instance'].unique()
        
        print(f"  Processing category {category} ({len(instances)} instances)")
        
        algorithms = subset['algorithm'].unique()
        
        for alg in sorted(algorithms):
            alg_subset = subset[subset['algorithm'] == alg]
            all_ttt = []
            
            for instance in instances:
                inst_data = alg_subset[alg_subset['instance'] == instance].iloc[0]
                target = compute_target_value(df, instance, target_percentile)
                
                loss_hist = inst_data['loss_history']
                time_hist = inst_data['time_history']
                
                if len(loss_hist) > 0 and len(time_hist) > 0:
                    target_idx = np.where(loss_hist <= target)[0]
                    if len(target_idx) > 0:
                        ttt = time_hist[target_idx[0]]
                        all_ttt.append(ttt)
            
            if len(all_ttt) > 0:
                all_ttt_sorted = np.sort(all_ttt)
                n = len(all_ttt_sorted)
                proportions = np.arange(1, n + 1) / len(instances)
                
                ax.plot(all_ttt_sorted, proportions, 
                       label=ALGORITHM_LABELS.get(alg, alg),
                       color=ALGORITHM_COLORS.get(alg),
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        
        ax.set_title(f'Category {category} ({len(instances)} instances)', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Proportion Reaching Target', fontsize=11)
        ax.set_xscale('log')
        ax.set_xlim(left=0.1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", linestyle=':', linewidth=0.6, alpha=0.7)
        ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(f'Time-To-Target Plots by Problem Category\n(Target = {target_percentile}th percentile)',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    filename = "ttt_combined_by_category.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def generate_combined_performance_profiles(df: pd.DataFrame, output_dir: Path):
    """
    Generate a 2x2 combined performance profile showing all sizes in one figure.
    """
    print("\n" + "="*80)
    print("GENERATING COMBINED PERFORMANCE PROFILE BY SIZE")
    print("="*80)
    
    sizes = sorted(df['cust_size'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, cust_size in enumerate(sizes):
        ax = axes[idx]
        subset = df[df['cust_size'] == cust_size]
        
        print(f"  Processing {cust_size} customers")
        
        instances = subset['instance'].unique()
        algorithms = subset['algorithm'].unique()
        
        # For each instance, find the best solution
        best_solutions = {}
        for instance in instances:
            inst_data = subset[subset['instance'] == instance]
            best_solutions[instance] = inst_data['final_loss'].min()
        
        # For each algorithm, compute performance ratios
        for alg in sorted(algorithms):
            alg_data = subset[subset['algorithm'] == alg]
            ratios = []
            
            for _, row in alg_data.iterrows():
                instance = row['instance']
                best = best_solutions[instance]
                ratio = best / row['final_loss'] if row['final_loss'] > 0 else 1.0
                ratios.append(ratio)
            
            # Sort ratios for cumulative distribution
            ratios_sorted = np.sort(ratios)
            n = len(ratios_sorted)
            probabilities = np.arange(1, n + 1) / n
            
            ax.plot(ratios_sorted, probabilities,
                   label=ALGORITHM_LABELS.get(alg, alg),
                   color=ALGORITHM_COLORS.get(alg),
                   linewidth=2.5, marker='o', markersize=3, alpha=0.8)
        
        ax.set_title(f'{cust_size} Customers ({len(instances)} instances)',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Performance Ratio (τ = best/algorithm)', fontsize=11)
        ax.set_ylabel('P(ratio ≥ τ)', fontsize=11)
        ax.set_xlim(0.85, 1.01)
        ax.set_ylim(0, 1.05)
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
        ax.legend(loc='lower left', fontsize=10)
    
    fig.suptitle('Performance Profiles by Customer Size',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    filename = "performance_profile_combined_by_size.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def generate_convergence_plot(df: pd.DataFrame, instance: str, output_dir: Path):
    """
    Generate convergence plot for a specific instance showing solution quality over time.
    """
    subset = df[df['instance'] == instance]
    
    if len(subset) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for _, row in subset.iterrows():
        alg = row['algorithm']
        loss_hist = row['loss_history']
        time_hist = row['time_history']
        
        label = ALGORITHM_LABELS.get(alg, alg)
        color = ALGORITHM_COLORS.get(alg, None)
        
        ax.plot(time_hist, loss_hist, linewidth=2, label=label, 
                color=color, alpha=0.8, marker='o', markersize=3)
    
    ax.set_title(f'Convergence Plot: {instance}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Objective Value (lower is better)', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, which="both", linestyle=':', linewidth=0.6, alpha=0.7)
    ax.legend(loc='upper right', fontsize=11)
    
    filename = f"convergence_{instance}.png"
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics tables with clean CSV formatting."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall summary with clean formatting
    print("\n  Overall Algorithm Performance")
    overall_stats = []
    for alg in sorted(df['algorithm'].unique()):
        alg_data = df[df['algorithm'] == alg]
        overall_stats.append({
            'Algorithm': ALGORITHM_LABELS.get(alg, alg),
            'Instances': len(alg_data),
            'Avg_Loss': alg_data['final_loss'].mean(),
            'Std_Loss': alg_data['final_loss'].std(),
            'Min_Loss': alg_data['final_loss'].min(),
            'Max_Loss': alg_data['final_loss'].max(),
            'Avg_Time': alg_data['final_time'].mean(),
            'Std_Time': alg_data['final_time'].std()
        })
    overall_df = pd.DataFrame(overall_stats)
    print(overall_df.to_string(index=False))
    overall_df.to_csv(output_dir / 'summary_overall.csv', index=False, float_format='%.2f')
    
    # By customer size with clean formatting
    print("\n  Performance by Customer Size")
    size_stats = []
    for size in sorted(df['cust_size'].unique()):
        for alg in sorted(df['algorithm'].unique()):
            subset = df[(df['cust_size'] == size) & (df['algorithm'] == alg)]
            if len(subset) > 0:
                size_stats.append({
                    'Customer_Size': size,
                    'Algorithm': ALGORITHM_LABELS.get(alg, alg),
                    'Instances': len(subset),
                    'Avg_Loss': subset['final_loss'].mean(),
                    'Std_Loss': subset['final_loss'].std()
                })
    size_df = pd.DataFrame(size_stats)
    print(size_df.to_string(index=False))
    size_df.to_csv(output_dir / 'summary_by_size.csv', index=False, float_format='%.2f')
    
    # By category with clean formatting
    print("\n  Performance by Category")
    category_stats = []
    for cat in sorted(df['category'].unique()):
        for alg in sorted(df['algorithm'].unique()):
            subset = df[(df['category'] == cat) & (df['algorithm'] == alg)]
            if len(subset) > 0:
                category_stats.append({
                    'Category': cat,
                    'Algorithm': ALGORITHM_LABELS.get(alg, alg),
                    'Instances': len(subset),
                    'Avg_Loss': subset['final_loss'].mean(),
                    'Std_Loss': subset['final_loss'].std()
                })
    category_df = pd.DataFrame(category_stats)
    print(category_df.to_string(index=False))
    category_df.to_csv(output_dir / 'summary_by_category.csv', index=False, float_format='%.2f')
    
    # Best algorithm per instance
    print("\n  Best Algorithm Count (instances where each algorithm found best solution):")
    best_per_instance = df.loc[df.groupby('instance')['final_loss'].idxmin()]
    best_counts = best_per_instance['algorithm'].value_counts()
    for alg, count in best_counts.items():
        print(f"    {ALGORITHM_LABELS.get(alg, alg)}: {count}")
    
    print(f"\n  Summary statistics saved to {output_dir}/summary_*.csv")


def main():
    """Main plotting pipeline."""
    print("="*80)
    print("HFFVRPTW Solver Performance Analysis")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Generate individual TTT plots (24 plots: 6 categories × 4 sizes)
    generate_all_ttt_plots(df, output_dir)
    
    # Generate combined TTT plots (2 plots: by size and by category)
    generate_combined_ttt_by_size(df, output_dir)
    generate_combined_ttt_by_category(df, output_dir)
    
    # Generate individual performance profiles (11 plots: 1 overall + 4 sizes + 6 categories)
    generate_all_performance_profiles(df, output_dir)
    
    # Generate combined performance profile (1 plot: 2x2 grid by size)
    generate_combined_performance_profiles(df, output_dir)
    
    # Generate summary statistics
    generate_summary_statistics(df, output_dir)
    
    # Generate a few example convergence plots (first 3 instances)
    print("\n" + "="*80)
    print("GENERATING EXAMPLE CONVERGENCE PLOTS")
    print("="*80)
    sample_instances = df['instance'].unique()[:3]
    for instance in sample_instances:
        print(f"  Generating convergence plot for {instance}")
        generate_convergence_plot(df, instance, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {output_dir.absolute()}:")
    print("  • 24 individual TTT plots (by category and size)")
    print("  • 2 combined TTT plots (by size and by category)")
    print("  • 11 individual performance profiles (overall + by size + by category)")
    print("  • 1 combined performance profile (by size)")
    print("  • 3 example convergence plots")
    print("  • 3 summary CSV files")
    print("\nRecommended for reports:")
    print("  → ttt_combined_by_size.png (shows all 4 sizes)")
    print("  → ttt_combined_by_category.png (shows all 6 categories)")
    print("  → performance_profile_combined_by_size.png (shows all 4 sizes)")
    print("  → performance_profile.png (overall comparison)")
    print("="*80)


if __name__ == "__main__":
    main()
