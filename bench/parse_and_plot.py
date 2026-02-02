#!/usr/bin/env python3
"""
Parse N-body benchmark logs, compute statistics, select best variants,
and generate publication-ready plots.
"""

import re
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set matplotlib to not use display
mpl.use('Agg')

# ============================================================================
# Parsing Functions
# ============================================================================

def parse_cpu_log(logfile: Path) -> Dict[str, float]:
    """Parse CPU log file and extract timing for all four variants."""
    times = {}
    
    with open(logfile, 'r') as f:
        content = f.read()
    
    # Patterns for CPU variants
    patterns = {
        'seq_nonsym': r'Sequential Non-Symmetric:\s+([\d.]+)\s+seconds',
        'seq_sym': r'Sequential Symmetric:\s+([\d.]+)\s+seconds',
        'omp_nonsym': r'OpenMP Non-Symmetric:\s+([\d.]+)\s+seconds',
        'omp_sym': r'OpenMP Symmetric:\s+([\d.]+)\s+seconds',
    }
    
    for variant, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            times[variant] = float(match.group(1))
    
    return times


def parse_cuda_log(logfile: Path) -> Dict[str, float]:
    """Parse CUDA log file and extract kernel and total times."""
    times = {}
    
    with open(logfile, 'r') as f:
        content = f.read()
    
    # Patterns for CUDA variants (flexible)
    # Look for lines containing "CUDA" and "(kernel):" or "(total):"
    kernel_pattern = r'CUDA.*\(kernel\):\s+([\d.]+)\s+seconds'
    total_pattern = r'CUDA.*\(total\):\s+([\d.]+)\s+seconds'
    
    kernel_match = re.search(kernel_pattern, content, re.IGNORECASE)
    total_match = re.search(total_pattern, content, re.IGNORECASE)
    
    if kernel_match:
        times['kernel'] = float(kernel_match.group(1))
    if total_match:
        times['total'] = float(total_match.group(1))
    
    return times


def extract_params_from_filename(filename: str) -> Dict:
    """Extract N, steps, threads, run number from filename."""
    params = {}
    
    # Thread count
    match = re.search(r'_T(\d+)_', filename)
    if match:
        params['threads'] = int(match.group(1))
    
    # N (problem size)
    match = re.search(r'_N(\d+)', filename)
    if match:
        params['N'] = int(match.group(1))
    
    # Steps
    match = re.search(r'_S(\d+)', filename)
    if match:
        params['steps'] = int(match.group(1))
    
    # Run number
    match = re.search(r'_run(\d+)\.log', filename)
    if match:
        params['run'] = int(match.group(1))
    
    # Experiment type
    if 'cpu_strong' in filename:
        params['experiment'] = 'strong_scaling'
    elif 'cpu_size' in filename:
        params['experiment'] = 'size_sweep'
    elif 'cpu_steps' in filename:
        params['experiment'] = 'timestep_sweep'
    elif 'cuda_' in filename:
        params['experiment'] = 'cuda'
        # Extract CUDA variant name
        match = re.search(r'cuda_([^_]+)_', filename)
        if match:
            params['cuda_variant'] = match.group(1)
    elif 'final_cpu' in filename:
        params['experiment'] = 'final_cpu'
    elif 'final_cuda' in filename:
        params['experiment'] = 'final_cuda'
        match = re.search(r'final_cuda_([^_]+)_', filename)
        if match:
            params['cuda_variant'] = match.group(1)
    
    return params


# ============================================================================
# Statistics & Analysis
# ============================================================================

def compute_stats(times: List[float]) -> Tuple[float, float]:
    """Compute median and standard deviation."""
    if not times:
        return 0.0, 0.0
    times = np.array(times)
    return float(np.median(times)), float(np.std(times))


def compute_interactions(N: int, symmetric: bool) -> int:
    """Compute number of interactions."""
    if symmetric:
        return N * (N - 1) // 2
    else:
        return N * (N - 1)


# ============================================================================
# Main Processing Functions
# ============================================================================

def process_cpu_baseline(results_dir: Path):
    """Process CPU baseline benchmark results."""
    print("Processing CPU baseline results...")
    
    # Collect all CPU log files
    log_files = list(results_dir.glob("cpu_*.log"))
    
    if not log_files:
        print("  No CPU log files found.")
        return
    
    print(f"  Found {len(log_files)} CPU log files")
    
    # Data structure: {(experiment, variant, N, steps, threads): [times]}
    data = {}
    raw_data = []
    
    for logfile in log_files:
        params = extract_params_from_filename(logfile.name)
        times = parse_cpu_log(logfile)
        
        for variant, time_s in times.items():
            key = (
                params.get('experiment', 'unknown'),
                variant,
                params.get('N', 0),
                params.get('steps', 0),
                params.get('threads', 0)
            )
            
            if key not in data:
                data[key] = []
            data[key].append(time_s)
            
            raw_data.append({
                'logfile': logfile.name,
                'experiment': params.get('experiment', 'unknown'),
                'variant': variant,
                'threads': params.get('threads', 0),
                'N': params.get('N', 0),
                'steps': params.get('steps', 0),
                'run_index': params.get('run', 0),
                'time_s': time_s
            })
    
    # Compute statistics
    summary_data = []
    
    for key, times in data.items():
        experiment, variant, N, steps, threads = key
        median, std = compute_stats(times)
        
        # Determine if symmetric
        symmetric = 'sym' in variant and 'nonsym' not in variant
        interactions = compute_interactions(N, symmetric)
        time_per_interaction = median / interactions if interactions > 0 else 0
        
        # Parallel/formulation
        parallel = 'omp' in variant
        formulation = 'symmetric' if symmetric else 'non-symmetric'
        
        summary_data.append({
            'experiment': experiment,
            'variant': variant,
            'formulation': formulation,
            'parallel': parallel,
            'threads': threads,
            'N': N,
            'steps': steps,
            'median_s': median,
            'std_s': std,
            'runs': len(times),
            'interactions': interactions,
            'time_per_interaction': time_per_interaction
        })
    
    # Save summary CSV
    import csv
    
    summary_file = results_dir / "summary_median.csv"
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['experiment', 'variant', 'formulation', 'parallel', 'threads', 
                     'N', 'steps', 'median_s', 'std_s', 'runs', 'interactions', 
                     'time_per_interaction']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"  Saved: {summary_file}")
    
    # Save raw times CSV
    raw_file = results_dir / "raw_times.csv"
    with open(raw_file, 'w', newline='') as f:
        fieldnames = ['logfile', 'experiment', 'variant', 'threads', 'N', 
                     'steps', 'run_index', 'time_s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_data)
    
    print(f"  Saved: {raw_file}")
    
    # Save CPU summary JSON
    cpu_summary = {}
    for row in summary_data:
        key = f"{row['variant']}_T{row['threads']}_N{row['N']}_S{row['steps']}"
        cpu_summary[key] = {
            'variant': row['variant'],
            'threads': row['threads'],
            'N': row['N'],
            'steps': row['steps'],
            'median_s': row['median_s'],
            'std_s': row['std_s']
        }
    
    summary_json_file = results_dir / "cpu_summary.json"
    with open(summary_json_file, 'w') as f:
        json.dump(cpu_summary, f, indent=2)
    
    print(f"  Saved: {summary_json_file}")
    
    # Select best CPU variants (canonical: N=6000, steps=10, threads=32)
    select_best_cpu_variants(summary_data, results_dir)


def select_best_cpu_variants(summary_data: List[Dict], results_dir: Path):
    """Select best sequential and OpenMP variants at canonical config."""
    print("  Selecting best CPU variants...")
    
    # Filter to canonical configuration
    canonical = [row for row in summary_data 
                 if row['N'] == 6000 and row['steps'] == 10 and row['threads'] == 32]
    
    if not canonical:
        print("    WARNING: No data for canonical config (N=6000, steps=10, threads=32)")
        return
    
    # Find best sequential
    sequential = [row for row in canonical if not row['parallel']]
    best_seq = min(sequential, key=lambda x: x['median_s']) if sequential else None
    
    # Find best OpenMP
    openmp = [row for row in canonical if row['parallel']]
    best_omp = min(openmp, key=lambda x: x['median_s']) if openmp else None
    
    selection = {
        'canonical_config': {'N': 6000, 'steps': 10, 'threads': 32},
        'best_sequential': {
            'variant': best_seq['variant'],
            'median_s': best_seq['median_s'],
            'std_s': best_seq['std_s']
        } if best_seq else None,
        'best_openmp': {
            'variant': best_omp['variant'],
            'median_s': best_omp['median_s'],
            'std_s': best_omp['std_s']
        } if best_omp else None
    }
    
    selection_file = results_dir / "selected_cpu_variants.json"
    with open(selection_file, 'w') as f:
        json.dump(selection, f, indent=2)
    
    print(f"    Best sequential: {best_seq['variant']} ({best_seq['median_s']:.4f} s)" if best_seq else "    No sequential variant found")
    print(f"    Best OpenMP: {best_omp['variant']} ({best_omp['median_s']:.4f} s)" if best_omp else "    No OpenMP variant found")
    print(f"  Saved: {selection_file}")


def process_cuda_results(results_dir: Path):
    """Process CUDA benchmark results."""
    print("Processing CUDA results...")
    
    # Collect all CUDA log files
    log_files = list(results_dir.glob("cuda_*.log"))
    
    if not log_files:
        print("  No CUDA log files found.")
        return
    
    print(f"  Found {len(log_files)} CUDA log files")
    
    # Data structure: {(cuda_variant, N, steps): {'kernel': [times], 'total': [times]}}
    data = {}
    raw_data = []
    
    for logfile in log_files:
        params = extract_params_from_filename(logfile.name)
        times = parse_cuda_log(logfile)
        
        cuda_variant = params.get('cuda_variant', 'unknown')
        N = params.get('N', 0)
        steps = params.get('steps', 0)
        run = params.get('run', 0)
        
        key = (cuda_variant, N, steps)
        
        if key not in data:
            data[key] = {'kernel': [], 'total': []}
        
        if 'kernel' in times:
            data[key]['kernel'].append(times['kernel'])
            raw_data.append({
                'logfile': logfile.name,
                'cuda_variant': cuda_variant,
                'N': N,
                'steps': steps,
                'run_index': run,
                'type': 'kernel',
                'time_s': times['kernel']
            })
        
        if 'total' in times:
            data[key]['total'].append(times['total'])
            raw_data.append({
                'logfile': logfile.name,
                'cuda_variant': cuda_variant,
                'N': N,
                'steps': steps,
                'run_index': run,
                'type': 'total',
                'time_s': times['total']
            })
    
    # Compute statistics
    summary_data = []
    
    for key, times_dict in data.items():
        cuda_variant, N, steps = key
        
        for time_type in ['kernel', 'total']:
            times = times_dict[time_type]
            if times:
                median, std = compute_stats(times)
                
                summary_data.append({
                    'cuda_variant': cuda_variant,
                    'N': N,
                    'steps': steps,
                    'type': time_type,
                    'median_s': median,
                    'std_s': std,
                    'runs': len(times)
                })
    
    # Save CUDA summary CSV
    import csv
    
    summary_file = results_dir / "cuda_summary_median.csv"
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['cuda_variant', 'N', 'steps', 'type', 'median_s', 'std_s', 'runs']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"  Saved: {summary_file}")
    
    # Save CUDA summary JSON
    cuda_summary = {}
    for row in summary_data:
        key = f"{row['cuda_variant']}_N{row['N']}_S{row['steps']}_{row['type']}"
        cuda_summary[key] = {
            'cuda_variant': row['cuda_variant'],
            'N': row['N'],
            'steps': row['steps'],
            'type': row['type'],
            'median_s': row['median_s'],
            'std_s': row['std_s']
        }
    
    summary_json_file = results_dir / "cuda_summary.json"
    with open(summary_json_file, 'w') as f:
        json.dump(cuda_summary, f, indent=2)
    
    print(f"  Saved: {summary_json_file}")
    
    # Select best CUDA variant
    select_best_cuda_variant(summary_data, results_dir)


def select_best_cuda_variant(summary_data: List[Dict], results_dir: Path):
    """Select best CUDA variant at canonical config (N=6000, steps=10)."""
    print("  Selecting best CUDA variant...")
    
    # Filter to canonical configuration and total times only
    canonical = [row for row in summary_data 
                 if row['N'] == 6000 and row['steps'] == 10 and row['type'] == 'total']
    
    if not canonical:
        print("    WARNING: No CUDA data for canonical config (N=6000, steps=10)")
        return
    
    # Find best CUDA variant (lowest total time)
    best_cuda = min(canonical, key=lambda x: x['median_s'])
    
    # Also get kernel time for best variant
    kernel_row = next((row for row in summary_data 
                      if row['cuda_variant'] == best_cuda['cuda_variant'] 
                      and row['N'] == 6000 and row['steps'] == 10 
                      and row['type'] == 'kernel'), None)
    
    selection = {
        'canonical_config': {'N': 6000, 'steps': 10},
        'best_cuda': {
            'variant': best_cuda['cuda_variant'],
            'total_median_s': best_cuda['median_s'],
            'total_std_s': best_cuda['std_s'],
            'kernel_median_s': kernel_row['median_s'] if kernel_row else None,
            'kernel_std_s': kernel_row['std_s'] if kernel_row else None,
            'kernel_to_total_ratio': kernel_row['median_s'] / best_cuda['median_s'] if kernel_row else None
        }
    }
    
    selection_file = results_dir / "selected_cuda_variant.json"
    with open(selection_file, 'w') as f:
        json.dump(selection, f, indent=2)
    
    print(f"    Best CUDA: {best_cuda['cuda_variant']} (total: {best_cuda['median_s']:.4f} s)")
    if kernel_row:
        print(f"      Kernel: {kernel_row['median_s']:.4f} s ({selection['best_cuda']['kernel_to_total_ratio']*100:.1f}% of total)")
    print(f"  Saved: {selection_file}")


def process_final_comparison(results_dir: Path):
    """Process final comparison results."""
    print("Processing final comparison results...")
    
    # Collect all final log files
    cpu_logs = list(results_dir.glob("final_cpu_*.log"))
    cuda_logs = list(results_dir.glob("final_cuda_*.log"))
    
    if not cpu_logs:
        print("  No final CPU log files found.")
        return
    
    print(f"  Found {len(cpu_logs)} CPU logs, {len(cuda_logs)} CUDA logs")
    
    # Load selections
    cpu_selection_file = results_dir / "selected_cpu_variants.json"
    cuda_selection_file = results_dir / "selected_cuda_variant.json"
    
    selected_cpu = {}
    selected_cuda = {}
    
    if cpu_selection_file.exists():
        with open(cpu_selection_file, 'r') as f:
            selected_cpu = json.load(f)
    
    if cuda_selection_file.exists():
        with open(cuda_selection_file, 'r') as f:
            selected_cuda = json.load(f)
    
    # Process CPU logs
    cpu_data = {}
    raw_data = []
    
    for logfile in cpu_logs:
        params = extract_params_from_filename(logfile.name)
        times = parse_cpu_log(logfile)
        
        for variant, time_s in times.items():
            key = (variant, params.get('N', 0), params.get('steps', 0))
            
            if key not in cpu_data:
                cpu_data[key] = []
            cpu_data[key].append(time_s)
            
            raw_data.append({
                'logfile': logfile.name,
                'implementation': f"cpu_{variant}",
                'N': params.get('N', 0),
                'steps': params.get('steps', 0),
                'run_index': params.get('run', 0),
                'time_s': time_s
            })
    
    # Process CUDA logs
    cuda_data = {}
    
    for logfile in cuda_logs:
        params = extract_params_from_filename(logfile.name)
        times = parse_cuda_log(logfile)
        
        cuda_variant = params.get('cuda_variant', 'unknown')
        
        if 'total' in times:
            key = (f"cuda_{cuda_variant}", params.get('N', 0), params.get('steps', 0))
            
            if key not in cuda_data:
                cuda_data[key] = []
            cuda_data[key].append(times['total'])
            
            raw_data.append({
                'logfile': logfile.name,
                'implementation': f"cuda_{cuda_variant}",
                'N': params.get('N', 0),
                'steps': params.get('steps', 0),
                'run_index': params.get('run', 0),
                'time_s': times['total']
            })
    
    # Combine data
    all_data = {**cpu_data, **cuda_data}
    
    # Compute statistics
    summary_data = []
    
    for key, times in all_data.items():
        impl, N, steps = key
        median, std = compute_stats(times)
        
        summary_data.append({
            'implementation': impl,
            'N': N,
            'steps': steps,
            'median_s': median,
            'std_s': std,
            'runs': len(times)
        })
    
    # Calculate speedups (baseline: sequential non-symmetric)
    baseline_times = {}
    for row in summary_data:
        if 'seq_nonsym' in row['implementation']:
            baseline_times[(row['N'], row['steps'])] = row['median_s']
    
    for row in summary_data:
        baseline = baseline_times.get((row['N'], row['steps']), None)
        if baseline and baseline > 0:
            row['speedup'] = baseline / row['median_s']
        else:
            row['speedup'] = None
        
        # Parallel efficiency for OpenMP (assuming 32 threads)
        if 'omp' in row['implementation'] and row['speedup']:
            row['efficiency'] = row['speedup'] / 32
        else:
            row['efficiency'] = None
    
    # Save final summary CSV
    import csv
    
    summary_file = results_dir / "final_summary_median.csv"
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['implementation', 'N', 'steps', 'median_s', 'std_s', 
                     'runs', 'speedup', 'efficiency']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"  Saved: {summary_file}")
    
    # Save raw times CSV
    raw_file = results_dir / "final_raw_times.csv"
    with open(raw_file, 'w', newline='') as f:
        fieldnames = ['logfile', 'implementation', 'N', 'steps', 'run_index', 'time_s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_data)
    
    print(f"  Saved: {raw_file}")
    
    # Generate summary text
    generate_summary_text(summary_data, selected_cpu, selected_cuda, results_dir)


def generate_summary_text(summary_data: List[Dict], selected_cpu: Dict, 
                         selected_cuda: Dict, results_dir: Path):
    """Generate human-readable summary text."""
    print("  Generating summary text...")
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("FINAL COMPARISON SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Selected implementations
    summary_lines.append("SELECTED IMPLEMENTATIONS:")
    summary_lines.append("-" * 80)
    
    if selected_cpu.get('best_sequential'):
        seq = selected_cpu['best_sequential']
        summary_lines.append(f"Best Sequential:  {seq['variant']} ({seq['median_s']:.4f} ± {seq['std_s']:.4f} s)")
    
    if selected_cpu.get('best_openmp'):
        omp = selected_cpu['best_openmp']
        summary_lines.append(f"Best OpenMP:      {omp['variant']} ({omp['median_s']:.4f} ± {omp['std_s']:.4f} s)")
    
    if selected_cuda.get('best_cuda'):
        cuda = selected_cuda['best_cuda']
        summary_lines.append(f"Best CUDA:        {cuda['variant']} ({cuda['total_median_s']:.4f} ± {cuda['total_std_s']:.4f} s)")
        if cuda.get('kernel_median_s'):
            summary_lines.append(f"  Kernel time:    {cuda['kernel_median_s']:.4f} s ({cuda.get('kernel_to_total_ratio', 0)*100:.1f}% of total)")
    
    summary_lines.append("")
    summary_lines.append("PERFORMANCE RESULTS (median times in seconds):")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'N':<8} {'Implementation':<25} {'Time (s)':<12} {'Speedup':<10} {'Winner':<10}")
    summary_lines.append("-" * 80)
    
    # Group by N
    sizes = sorted(set(row['N'] for row in summary_data))
    
    for N in sizes:
        rows = [row for row in summary_data if row['N'] == N]
        rows_sorted = sorted(rows, key=lambda x: x['median_s'])
        
        winner = rows_sorted[0] if rows_sorted else None
        
        for row in rows_sorted:
            impl = row['implementation'].replace('cpu_', '').replace('cuda_', 'CUDA ')
            time_str = f"{row['median_s']:.4f}"
            speedup_str = f"{row['speedup']:.2f}x" if row['speedup'] else "-"
            winner_str = "***" if row == winner else ""
            
            summary_lines.append(f"{N:<8} {impl:<25} {time_str:<12} {speedup_str:<10} {winner_str:<10}")
        
        summary_lines.append("")
    
    summary_lines.append("")
    summary_lines.append("INTERPRETATION:")
    summary_lines.append("-" * 80)
    
    # Find crossover points and general trends
    interpretations = []
    
    # GPU advantage
    gpu_rows = [row for row in summary_data if 'cuda' in row['implementation'].lower()]
    cpu_rows = [row for row in summary_data if 'cuda' not in row['implementation'].lower()]
    
    if gpu_rows and cpu_rows:
        for N in sizes:
            gpu_time = next((r['median_s'] for r in gpu_rows if r['N'] == N), None)
            cpu_times = [r['median_s'] for r in cpu_rows if r['N'] == N]
            best_cpu_time = min(cpu_times) if cpu_times else None
            
            if gpu_time and best_cpu_time:
                if gpu_time < best_cpu_time:
                    interpretations.append(f"At N={N}, GPU is {best_cpu_time/gpu_time:.1f}x faster than best CPU variant.")
                    break
    
    # OpenMP efficiency
    omp_rows = [row for row in summary_data if 'omp' in row['implementation'].lower() and row['efficiency']]
    if omp_rows:
        avg_efficiency = np.mean([r['efficiency'] for r in omp_rows if r['efficiency']])
        interpretations.append(f"OpenMP achieves {avg_efficiency*100:.1f}% parallel efficiency on average (32 threads).")
    
    # Symmetric vs non-symmetric
    seq_sym = [row for row in summary_data if 'seq_sym' in row['implementation']]
    seq_nonsym = [row for row in summary_data if 'seq_nonsym' in row['implementation']]
    
    if seq_sym and seq_nonsym:
        avg_speedup = np.mean([s['median_s'] / ns['median_s'] 
                              for s in seq_sym for ns in seq_nonsym 
                              if s['N'] == ns['N']])
        if avg_speedup < 1:
            interpretations.append(f"Sequential symmetric formulation is {1/avg_speedup:.2f}x faster than non-symmetric (fewer operations).")
    
    if not interpretations:
        interpretations.append("Results show expected performance characteristics for N-body simulations.")
    
    for interp in interpretations:
        summary_lines.append(f"• {interp}")
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    # Write summary
    summary_file = results_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"  Saved: {summary_file}")
    
    # Also print to console
    print("")
    print('\n'.join(summary_lines))


# ============================================================================
# Plotting Functions
# ============================================================================

def create_plots(results_dir: Path, final_mode: bool = False):
    """Create all plots."""
    print("Creating plots...")
    
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    if final_mode:
        create_final_plots(results_dir, plots_dir)
    else:
        create_baseline_plots(results_dir, plots_dir)


def create_baseline_plots(results_dir: Path, plots_dir: Path):
    """Create plots for baseline CPU/CUDA experiments."""
    import csv
    
    # Load summary data
    summary_file = results_dir / "summary_median.csv"
    if not summary_file.exists():
        print("  No summary data found for plotting")
        return
    
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Convert numeric fields
    for row in data:
        row['threads'] = int(row['threads']) if row['threads'] else 0
        row['N'] = int(row['N']) if row['N'] else 0
        row['steps'] = int(row['steps']) if row['steps'] else 0
        row['median_s'] = float(row['median_s']) if row['median_s'] else 0
        row['std_s'] = float(row['std_s']) if row['std_s'] else 0
    
    # Plot 1: Strong scaling (N=6000, steps=10)
    print("  Creating strong scaling plot...")
    strong_data = [row for row in data 
                  if row['N'] == 6000 and row['steps'] == 10 and row['threads'] > 0]
    
    if strong_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        variants = set(row['variant'] for row in strong_data)
        colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
        
        for i, variant in enumerate(sorted(variants)):
            variant_data = [row for row in strong_data if row['variant'] == variant]
            variant_data = sorted(variant_data, key=lambda x: x['threads'])
            
            threads = [row['threads'] for row in variant_data]
            times = [row['median_s'] for row in variant_data]
            stds = [row['std_s'] for row in variant_data]
            
            ax.errorbar(threads, times, yerr=stds, marker='o', label=variant, 
                       color=colors[i], capsize=5, linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Threads', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Strong Scaling (N=6000, steps=10)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "strong_scaling_N6000.png", dpi=300)
        plt.savefig(plots_dir / "strong_scaling_N6000.svg")
        plt.close()
        print(f"    Saved: {plots_dir / 'strong_scaling_N6000.png'}")
    
    # Plot 2: Size sweep
    print("  Creating size sweep plot...")
    size_data = [row for row in data 
                if row['threads'] == 32 and row['steps'] == 10]
    
    if size_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        variants = set(row['variant'] for row in size_data)
        colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
        
        for i, variant in enumerate(sorted(variants)):
            variant_data = [row for row in size_data if row['variant'] == variant]
            variant_data = sorted(variant_data, key=lambda x: x['N'])
            
            Ns = [row['N'] for row in variant_data]
            times = [row['median_s'] for row in variant_data]
            stds = [row['std_s'] for row in variant_data]
            
            ax.errorbar(Ns, times, yerr=stds, marker='o', label=variant, 
                       color=colors[i], capsize=5, linewidth=2, markersize=8)
        
        ax.set_xlabel('Problem Size (N)', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Problem Size Scaling (threads=32, steps=10)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "time_vs_N.png", dpi=300)
        plt.savefig(plots_dir / "time_vs_N.svg")
        plt.close()
        print(f"    Saved: {plots_dir / 'time_vs_N.png'}")


def create_final_plots(results_dir: Path, plots_dir: Path):
    """Create plots for final comparison."""
    import csv
    
    # Load final summary data
    summary_file = results_dir / "final_summary_median.csv"
    if not summary_file.exists():
        print("  No final summary data found for plotting")
        return
    
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Convert numeric fields
    for row in data:
        row['N'] = int(row['N']) if row['N'] else 0
        row['steps'] = int(row['steps']) if row['steps'] else 0
        row['median_s'] = float(row['median_s']) if row['median_s'] else 0
        row['std_s'] = float(row['std_s']) if row['std_s'] else 0
        row['speedup'] = float(row['speedup']) if row['speedup'] and row['speedup'] != '' else None
        row['efficiency'] = float(row['efficiency']) if row['efficiency'] and row['efficiency'] != '' else None
    
    # Plot 1: Time vs N (final comparison)
    print("  Creating final time vs N plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    implementations = set(row['implementation'] for row in data)
    colors = plt.cm.tab10(np.linspace(0, 1, len(implementations)))
    
    for i, impl in enumerate(sorted(implementations)):
        impl_data = [row for row in data if row['implementation'] == impl]
        impl_data = sorted(impl_data, key=lambda x: x['N'])
        
        Ns = [row['N'] for row in impl_data]
        times = [row['median_s'] for row in impl_data]
        stds = [row['std_s'] for row in impl_data]
        
        label = impl.replace('cpu_', '').replace('cuda_', 'CUDA ')
        
        ax.errorbar(Ns, times, yerr=stds, marker='o', label=label, 
                   color=colors[i], capsize=5, linewidth=2, markersize=8)
    
    ax.set_xlabel('Problem Size (N)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "final_time_vs_N.png", dpi=300)
    plt.savefig(plots_dir / "final_time_vs_N.svg")
    plt.close()
    print(f"    Saved: {plots_dir / 'final_time_vs_N.png'}")
    
    # Plot 2: Speedup
    print("  Creating speedup plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, impl in enumerate(sorted(implementations)):
        impl_data = [row for row in data if row['implementation'] == impl and row['speedup']]
        impl_data = sorted(impl_data, key=lambda x: x['N'])
        
        if impl_data:
            Ns = [row['N'] for row in impl_data]
            speedups = [row['speedup'] for row in impl_data]
            
            label = impl.replace('cpu_', '').replace('cuda_', 'CUDA ')
            
            ax.plot(Ns, speedups, marker='o', label=label, 
                   color=colors[i], linewidth=2, markersize=8)
    
    # Ideal speedup line (for OpenMP with 32 threads)
    if any('omp' in impl for impl in implementations):
        ax.axhline(y=32, color='gray', linestyle='--', linewidth=1, label='Ideal (32 threads)')
    
    ax.set_xlabel('Problem Size (N)', fontsize=12)
    ax.set_ylabel('Speedup vs Sequential Non-Symmetric', fontsize=12)
    ax.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "speedup_final.png", dpi=300)
    plt.savefig(plots_dir / "speedup_final.svg")
    plt.close()
    print(f"    Saved: {plots_dir / 'speedup_final.png'}")
    
    # Plot 3: Time per interaction
    print("  Creating time per interaction plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, impl in enumerate(sorted(implementations)):
        impl_data = [row for row in data if row['implementation'] == impl]
        impl_data = sorted(impl_data, key=lambda x: x['N'])
        
        Ns = [row['N'] for row in impl_data]
        
        # Calculate time per interaction
        times_per_int = []
        for row in impl_data:
            # Assume non-symmetric unless 'sym' in name
            symmetric = 'sym' in row['implementation'] and 'nonsym' not in row['implementation']
            interactions = compute_interactions(row['N'], symmetric)
            times_per_int.append(row['median_s'] / interactions if interactions > 0 else 0)
        
        label = impl.replace('cpu_', '').replace('cuda_', 'CUDA ')
        
        ax.plot(Ns, times_per_int, marker='o', label=label, 
               color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Problem Size (N)', fontsize=12)
    ax.set_ylabel('Time per Interaction (seconds)', fontsize=12)
    ax.set_title('Time per Interaction Scaling', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "time_per_interaction_vs_N.png", dpi=300)
    plt.savefig(plots_dir / "time_per_interaction_vs_N.svg")
    plt.close()
    print(f"    Saved: {plots_dir / 'time_per_interaction_vs_N.png'}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Parse N-body benchmark results and generate plots'
    )
    parser.add_argument('results_dir', type=Path, 
                       help='Results directory to process')
    parser.add_argument('--final', action='store_true',
                       help='Process as final comparison (vs baseline)')
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    print(f"\nProcessing results from: {args.results_dir}")
    print("=" * 80)
    
    if args.final:
        # Final comparison mode
        process_final_comparison(args.results_dir)
        create_plots(args.results_dir, final_mode=True)
    else:
        # Baseline mode - process both CPU and CUDA if present
        process_cpu_baseline(args.results_dir)
        
        # Check for CUDA logs
        if list(args.results_dir.glob("cuda_*.log")):
            process_cuda_results(args.results_dir)
        
        create_plots(args.results_dir, final_mode=False)
    
    print("")
    print("=" * 80)
    print("Processing complete!")
    print(f"Results saved to: {args.results_dir}")
    print(f"Plots saved to: {args.results_dir / 'plots'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
