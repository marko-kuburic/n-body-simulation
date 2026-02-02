#!/usr/bin/env python3
"""
Compare CPU and GPU benchmark results.
Merges results from separate CPU and GPU benchmark runs.
"""

import argparse
import json
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.use('Agg')


def load_csv_data(csv_file: Path):
    """Load CSV data into a dictionary."""
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key in ['N', 'steps', 'threads', 'runs']:
                    row[key] = int(row[key]) if row[key] else 0
                elif key in ['median_s', 'std_s', 'speedup', 'efficiency']:
                    row[key] = float(row[key]) if row[key] and row[key] != '' else None
            data.append(row)
    return data


def merge_cpu_gpu_results(cpu_dir: Path, gpu_dir: Path, output_dir: Path):
    """Merge CPU and GPU results for comparison."""
    
    print(f"Loading CPU results from: {cpu_dir}")
    print(f"Loading GPU results from: {gpu_dir}")
    
    # Load CPU selection
    cpu_selection_file = cpu_dir / "selected_cpu_variants.json"
    if not cpu_selection_file.exists():
        print("ERROR: No selected_cpu_variants.json in CPU results")
        sys.exit(1)
    
    with open(cpu_selection_file, 'r') as f:
        cpu_selection = json.load(f)
    
    # Load CPU final results or summary
    cpu_data_file = cpu_dir / "final_summary_median.csv"
    if not cpu_data_file.exists():
        cpu_data_file = cpu_dir / "summary_median.csv"
    
    if not cpu_data_file.exists():
        print("ERROR: No CPU summary CSV found")
        sys.exit(1)
    
    cpu_data = load_csv_data(cpu_data_file)
    
    # Load GPU summary
    gpu_data_file = gpu_dir / "cuda_summary_median.csv"
    if not gpu_data_file.exists():
        # Try parsing GPU results first
        print("Parsing GPU results...")
        import subprocess
        result = subprocess.run(
            ['python3', 'bench/parse_and_plot.py', str(gpu_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"ERROR: Failed to parse GPU results: {result.stderr}")
            sys.exit(1)
    
    if not gpu_data_file.exists():
        print("ERROR: No GPU summary CSV found")
        sys.exit(1)
    
    gpu_data = load_csv_data(gpu_data_file)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to matching configurations
    print("\nCreating comparison dataset...")
    
    # Get best CPU variants for N=6000, steps=10
    best_seq_name = cpu_selection.get('best_sequential', {}).get('variant', 'seq_sym')
    best_omp_name = cpu_selection.get('best_openmp', {}).get('variant', 'omp_nonsym')
    
    # Prepare comparison data
    comparison_data = []
    
    # Find common N values
    cpu_Ns = set(row['N'] for row in cpu_data)
    gpu_Ns = set(row['N'] for row in gpu_data)
    common_Ns = sorted(cpu_Ns & gpu_Ns)
    
    print(f"Common problem sizes: {common_Ns}")
    
    # For each common N, extract best CPU and GPU times
    for N in common_Ns:
        # Prefer size_sweep rows to keep experiments consistent across Ns
        seq_rows_pref = [r for r in cpu_data if r['N'] == N and r.get('experiment') == 'size_sweep' and best_seq_name in r.get('variant', r.get('implementation', ''))]
        seq_rows_any = [r for r in cpu_data if r['N'] == N and best_seq_name in r.get('variant', r.get('implementation', ''))]
        if seq_rows_pref or seq_rows_any:
            seq_row = (seq_rows_pref or seq_rows_any)[0]
            comparison_data.append({
                'implementation': 'cpu_' + best_seq_name,
                'N': N,
                'steps': seq_row.get('steps', 10),
                'median_s': seq_row.get('median_s', 0),
                'std_s': seq_row.get('std_s', 0),
                'runs': seq_row.get('runs', 0),
                'speedup': 1.0,
                'efficiency': None
            })
        
        # Find best OpenMP time at this N (prefer size_sweep)
        omp_rows_pref = [r for r in cpu_data if r['N'] == N and r.get('experiment') == 'size_sweep' and best_omp_name in r.get('variant', r.get('implementation', ''))]
        omp_rows_any = [r for r in cpu_data if r['N'] == N and best_omp_name in r.get('variant', r.get('implementation', ''))]
        if omp_rows_pref or omp_rows_any:
            omp_row = (omp_rows_pref or omp_rows_any)[0]
            baseline = seq_row['median_s'] if 'seq_row' in locals() and seq_row else None
            speedup = baseline / omp_row['median_s'] if baseline and omp_row['median_s'] > 0 else None
            
            comparison_data.append({
                'implementation': 'cpu_' + best_omp_name,
                'N': N,
                'steps': omp_row.get('steps', 10),
                'median_s': omp_row.get('median_s', 0),
                'std_s': omp_row.get('std_s', 0),
                'runs': omp_row.get('runs', 0),
                'speedup': speedup,
                'efficiency': speedup / 32 if speedup else None
            })
        
        # Find GPU time at this N (use 'total' type)
        gpu_rows = [r for r in gpu_data if r['N'] == N and r.get('type', 'total') == 'total']
        if gpu_rows:
            gpu_row = gpu_rows[0]
            baseline = seq_row['median_s'] if 'seq_row' in locals() and seq_row else None
            speedup = baseline / gpu_row['median_s'] if baseline and gpu_row['median_s'] > 0 else None
            
            comparison_data.append({
                'implementation': 'gpu_cuda',
                'N': N,
                'steps': gpu_row.get('steps', 10),
                'median_s': gpu_row.get('median_s', 0),
                'std_s': gpu_row.get('std_s', 0),
                'runs': gpu_row.get('runs', 0),
                'speedup': speedup,
                'efficiency': None
            })
    
    # Save comparison CSV
    comparison_file = output_dir / "cpu_gpu_comparison.csv"
    with open(comparison_file, 'w', newline='') as f:
        fieldnames = ['implementation', 'N', 'steps', 'median_s', 'std_s', 'runs', 'speedup', 'efficiency']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_data)
    
    print(f"Saved: {comparison_file}")
    
    # Create comparison plots
    create_comparison_plots(comparison_data, output_dir, cpu_selection)
    
    # Create summary text
    create_comparison_summary(comparison_data, output_dir, cpu_selection)
    
    print(f"\n✓ Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {output_dir / 'comparison_summary.txt'}")
    print(f"Plots: {output_dir / 'plots/'}")


def create_comparison_plots(data, output_dir: Path, cpu_selection):
    """Create comparison plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("\nCreating comparison plots...")
    
    # Collect unique Ns for consistent x-axis ticks
    unique_Ns = sorted(set(row['N'] for row in data))
    
    # Group by implementation
    implementations = {}
    for row in data:
        impl = row['implementation']
        if impl not in implementations:
            implementations[impl] = []
        implementations[impl].append(row)
    
    # Sort each implementation by N
    for impl in implementations:
        implementations[impl] = sorted(implementations[impl], key=lambda x: x['N'])
    
    # Plot 1: Time vs N
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'cpu_seq_sym': 'C0', 'cpu_seq_nonsym': 'C0', 
              'cpu_omp_sym': 'C1', 'cpu_omp_nonsym': 'C1',
              'gpu_cuda': 'C2'}
    
    for impl, impl_data in implementations.items():
        Ns = [r['N'] for r in impl_data]
        times = [r['median_s'] for r in impl_data]
        stds = [r['std_s'] for r in impl_data]
        
        label = impl.replace('cpu_', 'CPU ').replace('gpu_cuda', 'GPU CUDA').replace('_', ' ').title()
        color = colors.get(impl, 'C3')
        
        ax.errorbar(Ns, times, yerr=stds, marker='o', label=label,
                   color=color, capsize=5, linewidth=2, markersize=8)
    
    ax.set_xlabel('Problem Size (N)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('CPU vs GPU Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Set exact N values on x-axis
    ax.set_xticks(unique_Ns)
    ax.set_xticklabels([str(n) for n in unique_Ns])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "cpu_gpu_time_comparison.png", dpi=300)
    plt.savefig(plots_dir / "cpu_gpu_time_comparison.svg")
    plt.close()
    print(f"  Saved: {plots_dir / 'cpu_gpu_time_comparison.png'}")
    
    # Plot 2: Speedup
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for impl, impl_data in implementations.items():
        impl_data_with_speedup = [r for r in impl_data if r['speedup']]
        if not impl_data_with_speedup:
            continue
        
        Ns = [r['N'] for r in impl_data_with_speedup]
        speedups = [r['speedup'] for r in impl_data_with_speedup]
        
        label = impl.replace('cpu_', 'CPU ').replace('gpu_cuda', 'GPU CUDA').replace('_', ' ').title()
        color = colors.get(impl, 'C3')
        
        ax.plot(Ns, speedups, marker='o', label=label,
               color=color, linewidth=2, markersize=8)
    
    ax.set_xlabel('Problem Size (N)', fontsize=12)
    ax.set_ylabel('Speedup vs Sequential', fontsize=12)
    ax.set_title('Speedup Comparison: CPU vs GPU', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    # Set exact N values on x-axis
    ax.set_xticks(unique_Ns)
    ax.set_xticklabels([str(n) for n in unique_Ns])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "cpu_gpu_speedup.png", dpi=300)
    plt.savefig(plots_dir / "cpu_gpu_speedup.svg")
    plt.close()
    print(f"  Saved: {plots_dir / 'cpu_gpu_speedup.png'}")


def create_comparison_summary(data, output_dir: Path, cpu_selection):
    """Create human-readable comparison summary."""
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("CPU vs GPU COMPARISON SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    summary_lines.append("IMPLEMENTATIONS COMPARED:")
    summary_lines.append("-" * 80)
    
    if cpu_selection.get('best_sequential'):
        seq = cpu_selection['best_sequential']
        summary_lines.append(f"CPU Sequential:   {seq['variant']} ({seq['median_s']:.4f} ± {seq['std_s']:.4f} s)")
    
    if cpu_selection.get('best_openmp'):
        omp = cpu_selection['best_openmp']
        summary_lines.append(f"CPU OpenMP:       {omp['variant']} ({omp['median_s']:.4f} ± {omp['std_s']:.4f} s)")
    
    summary_lines.append(f"GPU CUDA:         Docker container")
    
    summary_lines.append("")
    summary_lines.append("PERFORMANCE COMPARISON:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'N':<8} {'Implementation':<25} {'Time (s)':<12} {'Speedup':<10} {'Winner':<10}")
    summary_lines.append("-" * 80)
    
    # Group by N
    by_N = {}
    for row in data:
        N = row['N']
        if N not in by_N:
            by_N[N] = []
        by_N[N].append(row)
    
    for N in sorted(by_N.keys()):
        rows = sorted(by_N[N], key=lambda x: x['median_s'])
        winner = rows[0] if rows else None
        
        for row in rows:
            impl = row['implementation'].replace('cpu_', '').replace('gpu_cuda', 'GPU').replace('_', ' ')
            time_str = f"{row['median_s']:.4f}"
            speedup_str = f"{row['speedup']:.2f}x" if row['speedup'] else "-"
            winner_str = "***" if row == winner else ""
            
            summary_lines.append(f"{N:<8} {impl:<25} {time_str:<12} {speedup_str:<10} {winner_str:<10}")
        
        summary_lines.append("")
    
    summary_lines.append("")
    summary_lines.append("KEY FINDINGS:")
    summary_lines.append("-" * 80)
    
    # Find crossover point
    for N in sorted(by_N.keys()):
        rows = by_N[N]
        gpu_row = next((r for r in rows if 'gpu' in r['implementation']), None)
        cpu_rows = [r for r in rows if 'gpu' not in r['implementation']]
        
        if gpu_row and cpu_rows:
            best_cpu = min(cpu_rows, key=lambda x: x['median_s'])
            if gpu_row['median_s'] < best_cpu['median_s']:
                speedup = best_cpu['median_s'] / gpu_row['median_s']
                summary_lines.append(f"• At N={N}, GPU is {speedup:.1f}x faster than best CPU")
                break
    
    # OpenMP efficiency
    omp_rows = [r for r in data if 'omp' in r['implementation'] and r['efficiency']]
    if omp_rows:
        avg_eff = np.mean([r['efficiency'] for r in omp_rows])
        summary_lines.append(f"• OpenMP achieves {avg_eff*100:.1f}% parallel efficiency on average")
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    # Write summary
    summary_file = output_dir / "comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"  Saved: {summary_file}")
    
    # Also print to console
    print("")
    print('\n'.join(summary_lines))


def main():
    parser = argparse.ArgumentParser(
        description='Compare CPU and GPU benchmark results'
    )
    parser.add_argument('cpu_dir', type=Path,
                       help='CPU results directory')
    parser.add_argument('gpu_dir', type=Path,
                       help='GPU results directory')
    parser.add_argument('--output', '-o', type=Path,
                       default=Path('results/comparison'),
                       help='Output directory for comparison')
    
    args = parser.parse_args()
    
    if not args.cpu_dir.exists():
        print(f"ERROR: CPU results directory not found: {args.cpu_dir}")
        sys.exit(1)
    
    if not args.gpu_dir.exists():
        print(f"ERROR: GPU results directory not found: {args.gpu_dir}")
        sys.exit(1)
    
    merge_cpu_gpu_results(args.cpu_dir, args.gpu_dir, args.output)


if __name__ == '__main__':
    main()
