#!/usr/bin/env python3
"""
Generate Publication-Quality Plots for CPU vs GPU Comparison
=============================================================
Creates figures suitable for academic term papers
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Publication-quality settings
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6

def plot_execution_time_comparison(df, output_dir, metadata):
    """Plot 1: Execution time vs problem size (log-log)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.loglog(df['N'], df['CPU_Time_s'], 'o-', label='CPU (OpenMP)', color='#1f77b4')
    ax.loglog(df['N'], df['GPU_Time_s'], 's-', label='GPU (CUDA)', color='#ff7f0e')
    
    # Add O(N²) reference line
    n_ref = np.array([df['N'].min(), df['N'].max()])
    # Scale to match GPU at midpoint
    scale = df['GPU_Time_s'].iloc[len(df)//2] / (df['N'].iloc[len(df)//2]**2)
    t_ref = scale * n_ref**2
    ax.loglog(n_ref, t_ref, '--', color='gray', linewidth=1, label='O(N²) reference')
    
    ax.set_xlabel('Number of Particles (N)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('CPU vs GPU Performance Scaling\n' + 
                 f"CPU: {metadata['cpu']['model'][:40]}..., GPU: {metadata['gpu']['model']}")
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_time_comparison.svg', bbox_inches='tight')
    print("  ✓ fig1_time_comparison.png/svg")
    plt.close()

def plot_speedup(df, output_dir, metadata):
    """Plot 2: GPU speedup vs problem size"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(df['N'], df['Speedup'], 'o-', color='#2ca02c', linewidth=2.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='No speedup')
    
    ax.set_xlabel('Number of Particles (N)')
    ax.set_ylabel('Speedup Factor (CPU time / GPU time)')
    ax.set_title('GPU Speedup Over CPU\n' +
                 f"GPU: {metadata['gpu']['model']}, CPU: {metadata['cpu']['threads_used']} threads")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annotate max speedup
    max_idx = df['Speedup'].idxmax()
    max_n = df.loc[max_idx, 'N']
    max_speedup = df.loc[max_idx, 'Speedup']
    ax.annotate(f'{max_speedup:.1f}x', 
                xy=(max_n, max_speedup), 
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_speedup.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_speedup.svg', bbox_inches='tight')
    print("  ✓ fig2_speedup.png/svg")
    plt.close()

def plot_gflops_comparison(df, output_dir, metadata):
    """Plot 3: Computational throughput (GFlops)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    width = 0.35
    x = np.arange(len(df))
    
    bars1 = ax.bar(x - width/2, df['CPU_GFlops'], width, label='CPU', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['GPU_GFlops'], width, label='GPU', color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Problem Size (N particles)')
    ax.set_ylabel('Computational Throughput (GFlops)')
    ax.set_title('Computational Performance Comparison\n' +
                 'Direct N-body simulation (all-pairs)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['N'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels if needed
    if len(df) > 6:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_gflops.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_gflops.svg', bbox_inches='tight')
    print("  ✓ fig3_gflops.png/svg")
    plt.close()

def plot_combined_analysis(df, output_dir, metadata):
    """Plot 4: Combined 2x2 panel figure"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Time comparison (log-log)
    ax1.loglog(df['N'], df['CPU_Time_s'], 'o-', label='CPU', color='#1f77b4')
    ax1.loglog(df['N'], df['GPU_Time_s'], 's-', label='GPU', color='#ff7f0e')
    ax1.set_xlabel('Number of Particles (N)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('(a) Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Panel 2: Speedup
    ax2.semilogx(df['N'], df['Speedup'], 'o-', color='#2ca02c', linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Number of Particles (N)')
    ax2.set_ylabel('Speedup (CPU/GPU)')
    ax2.set_title('(b) GPU Speedup Factor')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: GFlops
    ax3.semilogx(df['N'], df['CPU_GFlops'], 'o-', label='CPU', color='#1f77b4')
    ax3.semilogx(df['N'], df['GPU_GFlops'], 's-', label='GPU', color='#ff7f0e')
    ax3.set_xlabel('Number of Particles (N)')
    ax3.set_ylabel('Throughput (GFlops)')
    ax3.set_title('(c) Computational Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Efficiency (GFlops per watt - placeholder, or use time per interaction)
    time_per_interaction_cpu = df['CPU_Time_s'] / (df['N']**2 * 10) * 1e9  # nanoseconds
    time_per_interaction_gpu = df['GPU_Time_s'] / (df['N']**2 * 10) * 1e9
    ax4.loglog(df['N'], time_per_interaction_cpu, 'o-', label='CPU', color='#1f77b4')
    ax4.loglog(df['N'], time_per_interaction_gpu, 's-', label='GPU', color='#ff7f0e')
    ax4.set_xlabel('Number of Particles (N)')
    ax4.set_ylabel('Time per Interaction (ns)')
    ax4.set_title('(d) Per-Interaction Cost')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(f'N-Body Performance Analysis: {metadata["gpu"]["model"]} vs {metadata["cpu"]["model"][:30]}...',
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_combined_analysis.svg', bbox_inches='tight')
    print("  ✓ fig4_combined_analysis.png/svg")
    plt.close()

def plot_performance_table(df, output_dir, metadata):
    """Generate a publication-ready table figure"""
    fig, ax = plt.subplots(figsize=(10, len(df)*0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['N', 'CPU Time (s)', 'GPU Time (s)', 'Speedup', 'CPU GFlops', 'GPU GFlops'])
    for _, row in df.iterrows():
        table_data.append([
            f"{int(row['N']):,}",
            f"{row['CPU_Time_s']:.4f}",
            f"{row['GPU_Time_s']:.4f}",
            f"{row['Speedup']:.2f}x",
            f"{row['CPU_GFlops']:.2f}",
            f"{row['GPU_GFlops']:.2f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.18, 0.18, 0.15, 0.17, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(6):
            table[(i, j)].set_facecolor(color)
    
    plt.title(f'Performance Comparison Table\nCPU: {metadata["cpu"]["model"]}\nGPU: {metadata["gpu"]["model"]}',
              fontsize=11, pad=20)
    plt.savefig(output_dir / 'fig5_table.png', dpi=300, bbox_inches='tight')
    print("  ✓ fig5_table.png")
    plt.close()

def main(results_dir):
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load CSV data
    csv_file = results_path / 'comparison_scaling.csv'
    if not csv_file.exists():
        print(f"Error: CSV file not found. Run parse_comparison.py first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    
    # Load metadata
    with open(results_path / 'metadata.json') as f:
        metadata = json.load(f)
    
    # Create plots directory
    plots_dir = results_path / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("Generating plots...")
    print(f"Output directory: {plots_dir}/")
    print()
    
    plot_execution_time_comparison(df, plots_dir, metadata)
    plot_speedup(df, plots_dir, metadata)
    plot_gflops_comparison(df, plots_dir, metadata)
    plot_combined_analysis(df, plots_dir, metadata)
    plot_performance_table(df, plots_dir, metadata)
    
    print()
    print("=" * 60)
    print("✓ All plots generated successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  - fig1_time_comparison.png/svg  : Execution time scaling")
    print("  - fig2_speedup.png/svg          : GPU speedup factor")
    print("  - fig3_gflops.png/svg           : Computational throughput")
    print("  - fig4_combined_analysis.png/svg: 4-panel comprehensive view")
    print("  - fig5_table.png                : Performance data table")
    print()
    print(f"Location: {plots_dir}/")
    print()
    print("These figures are publication-ready for your term paper!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 plot_comparison.py <results_dir>")
        sys.exit(1)
    
    main(sys.argv[1])
