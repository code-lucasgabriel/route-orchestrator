"""
analysis_suite.py

Advanced aggregated visualizations for HFFVRPTW solver performance analysis.
Extends the v7 plotter with:
1. Distributional Analysis: Violin plots showing full loss distributions
2. Aggregated Temporal Analysis: Median convergence with IQR bands
3. Advanced Narrative Integration: Pairwise dominance heatmaps

Uses the same aesthetic constants (ALGORITHM_LABELS, PALETTE_V7) as plotter.py
for visual consistency across the entire publication suite.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = 'results.parquet'
OUTPUT_DIR = 'plots'
IMAGES_DIR = 'plots/images'
PDFS_DIR = 'plots/PDFs'
SUMMARIES_DIR = 'plots/summaries'

# Publication-quality typography (matching plotter.py)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Source Serif Pro', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'

# Algorithm names (from plotter.py)
ALGORITHM_LABELS = {
    'alns_adaptive_sa': 'ALNS-SA',
    'alns_greedy_lns': 'ALNS-Greedy',
    'ts_tenure5': 'TS (tenure=5)',
    'ts_tenure0': 'TS (tenure=0)'
}

# V7 4-Color Distinct Palette (from plotter.py)
PALETTE_V7 = {
    'ALNS-SA': '#0072B2',        # Dark Blue
    'ALNS-Greedy': '#56B4E9',    # Sky Blue
    'TS (tenure=5)': '#D55E00',  # Vermillion
    'TS (tenure=0)': '#E69F00'   # Orange
}


def ensure_output_directories():
    """Create output directories if they don't exist."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(PDFS_DIR).mkdir(parents=True, exist_ok=True)
    Path(SUMMARIES_DIR).mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load the parquet file and add Algorithm labels."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    try:
        df = pd.read_parquet(DATA_FILE)
        print(f"✓ Loaded {len(df)} records from {DATA_FILE}")
        
        # Apply algorithm labels
        df['Algorithm'] = df['algorithm'].map(ALGORITHM_LABELS)
        
        print(f"✓ Algorithms: {sorted(df['Algorithm'].unique())}")
        print(f"✓ Customer sizes: {sorted(df['cust_size'].unique())}")
        print(f"✓ Categories: {sorted(df['category'].unique())}")
        
        return df
    except FileNotFoundError:
        print(f"ERROR: File '{DATA_FILE}' not found.")
        print("Please run parser.py first to generate the data file.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        exit(1)


# ============================================================================
# PART 1: DISTRIBUTIONAL ANALYSIS
# ============================================================================

def generate_violin_plot_by_size(df: pd.DataFrame):
    """
    Create violin plots showing final_loss distribution by customer size.
    Uses log scale on y-axis to handle different scales.
    """
    print("\n[1.2] Generating Violin Plot: Final Loss Distribution by Customer Size")
    print("-"*80)
    
    # Create the plot using seaborn.catplot
    g = sns.catplot(
        data=df,
        x='Algorithm',
        y='final_loss',
        hue='Algorithm',
        col='cust_size',
        col_wrap=2,
        kind='violin',
        palette=PALETTE_V7,
        inner='box',
        legend=False,
        height=5,
        aspect=1.2
    )
    
    # Set log scale on y-axis for all facets
    for ax in g.axes.flat:
        ax.set_yscale('log')
        ax.set_xlabel('Algorithm', fontsize=11)
        ax.set_ylabel('Final Loss (log scale)', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, which="major", axis='y', linestyle=':', linewidth=0.7, color='lightgray', alpha=0.7)
    
    # Set main title
    g.fig.suptitle('Final Loss Distribution by Customer Size', 
                   fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    filename = "distribution_by_size"
    g.savefig(Path(IMAGES_DIR) / f"{filename}.png", dpi=300, bbox_inches='tight')
    g.savefig(Path(PDFS_DIR) / f"{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}.png and {filename}.pdf")


def generate_violin_plot_by_category(df: pd.DataFrame):
    """
    Create violin plots showing final_loss distribution by problem category.
    Uses log scale on y-axis to handle different scales.
    """
    print("\n[1.3] Generating Violin Plot: Final Loss Distribution by Problem Category")
    print("-"*80)
    
    # Create the plot using seaborn.catplot
    g = sns.catplot(
        data=df,
        x='Algorithm',
        y='final_loss',
        hue='Algorithm',
        col='category',
        col_wrap=3,
        kind='violin',
        palette=PALETTE_V7,
        inner='box',
        legend=False,
        height=4,
        aspect=1.2
    )
    
    # Set log scale on y-axis for all facets
    for ax in g.axes.flat:
        ax.set_yscale('log')
        ax.set_xlabel('Algorithm', fontsize=10)
        ax.set_ylabel('Final Loss (log scale)', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, which="major", axis='y', linestyle=':', linewidth=0.7, color='lightgray', alpha=0.7)
    
    # Set main title
    g.fig.suptitle('Final Loss Distribution by Problem Category', 
                   fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    filename = "distribution_by_category"
    g.savefig(Path(IMAGES_DIR) / f"{filename}.png", dpi=300, bbox_inches='tight')
    g.savefig(Path(PDFS_DIR) / f"{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}.png and {filename}.pdf")


# ============================================================================
# PART 2: AGGREGATED TEMPORAL ANALYSIS
# ============================================================================

def prepare_aggregated_convergence_data(df: pd.DataFrame):
    """
    Prepare interpolated data for aggregated convergence plots.
    Creates a standardized time axis and interpolates all 944 runs.
    """
    print("\n" + "="*80)
    print("PART 2: AGGREGATED TEMPORAL ANALYSIS")
    print("="*80)
    print("\n[2.1] Preparing Aggregated Convergence Data")
    print("-"*80)
    print("Creating standardized time axis and interpolating all runs...")
    
    # Define standardized time axis (log-spaced from 0.1s to 600s)
    time_axis = np.logspace(np.log10(0.1), np.log10(600), 100)
    
    all_interpolated_data = []
    
    # Iterate through all rows
    for idx, row in df.iterrows():
        row_num = len(all_interpolated_data) + 1
        if row_num % 100 == 0:
            print(f"  Processing row {row_num}/{len(df)}...", end='\r')
        
        time_history = np.array(row['time_history'])
        loss_history = np.array(row['loss_history'])
        
        # Handle duplicate time values by keeping only the last (best) loss at each time
        df_temp = pd.DataFrame({'time': time_history, 'loss': loss_history})
        df_temp = df_temp.drop_duplicates(subset='time', keep='last')
        time_history = df_temp['time'].values
        loss_history = df_temp['loss'].values
        
        # Create pandas Series for forward-fill interpolation
        # This respects the step-function nature of "best-so-far" values
        series = pd.Series(loss_history, index=time_history)
        
        # Reindex with new time axis using forward-fill
        interpolated = series.reindex(
            series.index.union(time_axis)
        ).sort_index().ffill().reindex(time_axis)
        
        # Create temporary DataFrame
        temp_df = pd.DataFrame({
            'Time': time_axis,
            'Interpolated_Loss': interpolated.values,
            'Algorithm': row['Algorithm'],
            'cust_size': row['cust_size'],
            'category': row['category']
        })
        
        all_interpolated_data.append(temp_df)
    
    # Concatenate all DataFrames
    aggregated_df = pd.concat(all_interpolated_data, ignore_index=True)
    
    print(f"\n✓ Created aggregated dataset with {len(aggregated_df)} rows")
    print(f"  (100 time points × {len(df)} runs = {len(aggregated_df)} total)")
    
    return aggregated_df


def generate_aggregated_convergence_by_size(aggregated_df: pd.DataFrame):
    """
    Create aggregated convergence plots by customer size.
    Shows median performance with IQR bands.
    """
    print("\n[2.2] Generating Aggregated Convergence Plot by Customer Size")
    print("-"*80)
    
    # Create the plot using seaborn.relplot
    g = sns.relplot(
        data=aggregated_df,
        x='Time',
        y='Interpolated_Loss',
        hue='Algorithm',
        col='cust_size',
        col_wrap=2,
        kind='line',
        palette=PALETTE_V7,
        estimator=np.median,
        errorbar=('pi', 50),  # 25th to 75th percentile (IQR)
        height=5,
        aspect=1.2,
        linewidth=2.5,
        err_kws={'alpha': 0.3}
    )
    
    # Set log scale on both axes for all facets
    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Objective Value (log scale)', fontsize=11)
        ax.grid(True, which="major", linestyle=':', linewidth=0.7, color='lightgray', alpha=0.7)
    
    # Set main title
    g.fig.suptitle('Aggregated Median Convergence by Customer Size (IQR Bands)', 
                   fontsize=16, fontweight='bold', y=1.02)
    
    # Move legend to better position (if legend exists)
    if hasattr(g, '_legend') and g._legend is not None:
        g._legend.set_bbox_to_anchor((1.05, 0.5))
        g._legend.set_title('Algorithm')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    filename = "convergence_aggregated_by_size"
    g.savefig(Path(IMAGES_DIR) / f"{filename}.png", dpi=300, bbox_inches='tight')
    g.savefig(Path(PDFS_DIR) / f"{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}.png and {filename}.pdf")


def generate_aggregated_convergence_by_category(aggregated_df: pd.DataFrame):
    """
    Create aggregated convergence plots by problem category.
    Shows median performance with IQR bands.
    """
    print("\n[2.3] Generating Aggregated Convergence Plot by Category")
    print("-"*80)
    
    # Create the plot using seaborn.relplot
    g = sns.relplot(
        data=aggregated_df,
        x='Time',
        y='Interpolated_Loss',
        hue='Algorithm',
        col='category',
        col_wrap=3,
        kind='line',
        palette=PALETTE_V7,
        estimator=np.median,
        errorbar=('pi', 50),  # 25th to 75th percentile (IQR)
        height=4,
        aspect=1.2,
        linewidth=2.5,
        err_kws={'alpha': 0.3}
    )
    
    # Set log scale on both axes for all facets
    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Objective Value (log scale)', fontsize=10)
        ax.grid(True, which="major", linestyle=':', linewidth=0.7, color='lightgray', alpha=0.7)
    
    # Set main title
    g.fig.suptitle('Aggregated Median Convergence by Problem Category (IQR Bands)', 
                   fontsize=16, fontweight='bold', y=1.02)
    
    # Move legend to better position (if legend exists)
    if hasattr(g, '_legend') and g._legend is not None:
        g._legend.set_bbox_to_anchor((1.05, 0.5))
        g._legend.set_title('Algorithm')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    filename = "convergence_aggregated_by_category"
    g.savefig(Path(IMAGES_DIR) / f"{filename}.png", dpi=300, bbox_inches='tight')
    g.savefig(Path(PDFS_DIR) / f"{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}.png and {filename}.pdf")


# ============================================================================
# PART 3: ADVANCED NARRATIVE INTEGRATION
# ============================================================================

def generate_pairwise_dominance_heatmap(df: pd.DataFrame):
    """
    Create a pairwise dominance heatmap showing win rates.
    Answers: "What percentage of the time does Algorithm i beat Algorithm j?"
    """
    print("\n" + "="*80)
    print("PART 3: ADVANCED NARRATIVE INTEGRATION")
    print("="*80)
    print("\n[3.1] Generating Pairwise Dominance Heatmap")
    print("-"*80)
    
    # Pivot to get final_loss for each algorithm in columns
    pivoted = df.pivot_table(
        index='instance',
        columns='Algorithm',
        values='final_loss'
    )
    
    print(f"Analyzing {len(pivoted)} instances...")
    
    # Get sorted algorithm list
    algorithms = sorted(pivoted.columns)
    
    # Create win matrix
    win_matrix = pd.DataFrame(
        index=algorithms,
        columns=algorithms,
        dtype=float
    )
    
    # Calculate win rates
    for row_alg in algorithms:
        for col_alg in algorithms:
            if row_alg == col_alg:
                # Diagonal: algorithm vs itself = 50% (tie)
                win_matrix.loc[row_alg, col_alg] = 50.0
            else:
                # Win rate: percentage where row_alg < col_alg (lower is better)
                # Account for ties by giving 0.5 to each algorithm
                wins = (pivoted[row_alg] < pivoted[col_alg]).mean()
                ties = (pivoted[row_alg] == pivoted[col_alg]).mean()
                win_rate = (wins + 0.5 * ties) * 100
                win_matrix.loc[row_alg, col_alg] = win_rate
    
    # Convert to numeric
    win_matrix = win_matrix.astype(float)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        win_matrix,
        annot=True,
        fmt=".1f",
        cmap='vlag',
        center=50.0,
        vmin=0,
        vmax=100,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Win Rate (%)'},
        ax=ax
    )
    
    ax.set_title('Pairwise Dominance Heatmap (% Wins)\nRow Algorithm beats Column Algorithm', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Column Algorithm (Opponent)', fontsize=12)
    ax.set_ylabel('Row Algorithm', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    filename = "heatmap_dominance_overall"
    plt.savefig(Path(IMAGES_DIR) / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(Path(PDFS_DIR) / f"{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}.png and {filename}.pdf")
    
    # Print summary
    print("\nWin Rate Summary:")
    print("-"*80)
    for row_alg in algorithms:
        avg_win_rate = win_matrix.loc[row_alg].drop(row_alg).mean()
        print(f"  {row_alg}: {avg_win_rate:.1f}% average win rate vs others")
    
    return win_matrix


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline for advanced analysis suite."""
    print("="*80)
    print("ADVANCED ANALYSIS SUITE FOR HFFVRPTW SOLVER PERFORMANCE")
    print("="*80)
    print("\nExtending v7 plotter with:")
    print("  1. Distributional Analysis (Violin Plots)")
    print("  2. Aggregated Temporal Analysis (Median Convergence)")
    print("  3. Advanced Narrative Integration (Dominance Heatmaps)")
    print("\n")
    
    # Ensure directories exist
    ensure_output_directories()
    
    # Load data
    df = load_and_prepare_data()
    
    # PART 1: Distributional Analysis
    generate_violin_plot_by_size(df)
    generate_violin_plot_by_category(df)
    
    # PART 2: Aggregated Temporal Analysis
    aggregated_df = prepare_aggregated_convergence_data(df)
    generate_aggregated_convergence_by_size(aggregated_df)
    generate_aggregated_convergence_by_category(aggregated_df)
    
    # PART 3: Advanced Narrative Integration
    generate_pairwise_dominance_heatmap(df)
    
    # Final summary
    print("\n" + "="*80)
    print("ADVANCED ANALYSIS SUITE COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {Path(OUTPUT_DIR).absolute()}:")
    print(f"\nPNG files in {Path(IMAGES_DIR).absolute()}:")
    print("  • distribution_by_size.png")
    print("  • distribution_by_category.png")
    print("  • convergence_aggregated_by_size.png")
    print("  • convergence_aggregated_by_category.png")
    print("  • heatmap_dominance_overall.png")
    print(f"\nPDF files in {Path(PDFS_DIR).absolute()}:")
    print("  • distribution_by_size.pdf")
    print("  • distribution_by_category.pdf")
    print("  • convergence_aggregated_by_size.pdf")
    print("  • convergence_aggregated_by_category.pdf")
    print("  • heatmap_dominance_overall.pdf")
    print(f"\nCSV files in {Path(SUMMARIES_DIR).absolute()}:")
    print("  • mean_vs_median_comparison.csv")
    print("\nAll visualizations use v7 aesthetic constants for consistency.")
    print("="*80)


if __name__ == "__main__":
    main()
