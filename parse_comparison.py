#!/usr/bin/env python3
"""
Parse CPU vs GPU Comparison Benchmark Results
==============================================
Extracts timing data from CPU and GPU logs and generates CSV files
"""

import re
import json
import sys
from pathlib import Path
import statistics

def parse_cpu_log(log_file):
    """Extract CPU timing from nbody_naive output"""
    with open(log_file) as f:
        content = f.read()
    
    # Extract OpenMP non-symmetric time (best CPU performance)
    match = re.search(r'3\.\s+OpenMP Non-Symmetric:\s+([\d.]+)\s+seconds', content)
    if match:
        return float(match.group(1))
    return None

def parse_gpu_log(log_file):
    """Extract GPU timing from nbody_cuda output"""
    with open(log_file) as f:
        content = f.read()
    
    # Extract CUDA tiled kernel time (best GPU performance)
    match = re.search(r'6\.\s+CUDA Tiled \(kernel\):\s+([\d.]+)\s+seconds', content)
    if match:
        return float(match.group(1))
    return None

def main(results_dir):
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load metadata
    with open(results_path / 'metadata.json') as f:
        metadata = json.load(f)
    
    problem_sizes = metadata['problem_sizes']
    repeats = metadata['repeats']
    
    # Parse problem size scaling results
    print("Parsing problem size scaling results...")
    scaling_data = []
    
    for N in problem_sizes:
        cpu_times = []
        gpu_times = []
        
        for rep in range(1, repeats + 1):
            # CPU
            cpu_file = results_path / f"cpu_size_N{N}_rep{rep}.log"
            if cpu_file.exists():
                time = parse_cpu_log(cpu_file)
                if time:
                    cpu_times.append(time)
            
            # GPU
            gpu_file = results_path / f"gpu_size_N{N}_rep{rep}.log"
            if gpu_file.exists():
                time = parse_gpu_log(gpu_file)
                if time:
                    gpu_times.append(time)
        
        if cpu_times and gpu_times:
            cpu_median = statistics.median(cpu_times)
            gpu_median = statistics.median(gpu_times)
            speedup = cpu_median / gpu_median
            
            scaling_data.append({
                'N': N,
                'cpu_time': cpu_median,
                'gpu_time': gpu_median,
                'speedup': speedup,
                'cpu_gflops': (N * N * metadata['timesteps'] * 20) / (cpu_median * 1e9),  # ~20 FLOPs per interaction
                'gpu_gflops': (N * N * metadata['timesteps'] * 20) / (gpu_median * 1e9)
            })
            print(f"  N={N:5d}: CPU={cpu_median:8.4f}s, GPU={gpu_median:8.4f}s, Speedup={speedup:6.2f}x")
    
    # Write CSV
    csv_file = results_path / 'comparison_scaling.csv'
    with open(csv_file, 'w') as f:
        f.write("N,CPU_Time_s,GPU_Time_s,Speedup,CPU_GFlops,GPU_GFlops\n")
        for row in scaling_data:
            f.write(f"{row['N']},{row['cpu_time']:.6f},{row['gpu_time']:.6f},"
                   f"{row['speedup']:.4f},{row['cpu_gflops']:.4f},{row['gpu_gflops']:.4f}\n")
    
    print(f"\n✓ Results saved to {csv_file}")
    
    # Parse large problem results
    print("\nParsing large problem results...")
    cpu_large = []
    gpu_large = []
    
    for rep in range(1, repeats + 1):
        cpu_file = results_path / f"cpu_large_rep{rep}.log"
        if cpu_file.exists():
            time = parse_cpu_log(cpu_file)
            if time:
                cpu_large.append(time)
        
        gpu_file = results_path / f"gpu_large_rep{rep}.log"
        if gpu_file.exists():
            time = parse_gpu_log(gpu_file)
            if time:
                gpu_large.append(time)
    
    if cpu_large and gpu_large:
        cpu_med = statistics.median(cpu_large)
        gpu_med = statistics.median(gpu_large)
        speedup = cpu_med / gpu_med
        
        print(f"  Large problem (N=12000, 20 steps):")
        print(f"    CPU: {cpu_med:.4f}s")
        print(f"    GPU: {gpu_med:.4f}s")
        print(f"    Speedup: {speedup:.2f}x")
        
        # Write summary
        summary_file = results_path / 'comparison_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("CPU vs GPU Performance Comparison\n")
            f.write("==================================\n\n")
            f.write(f"CPU: {metadata['cpu']['model']}\n")
            f.write(f"  Cores/Threads: {metadata['cpu']['cores']}\n\n")
            f.write(f"GPU: {metadata['gpu']['model']}\n")
            f.write(f"  Memory: {metadata['gpu']['memory_gb']} GB\n\n")
            f.write("Problem Size Scaling:\n")
            f.write(f"{'N':>6} {'CPU(s)':>10} {'GPU(s)':>10} {'Speedup':>10}\n")
            f.write("-" * 40 + "\n")
            for row in scaling_data:
                f.write(f"{row['N']:6d} {row['cpu_time']:10.4f} {row['gpu_time']:10.4f} {row['speedup']:10.2f}x\n")
            f.write("\n")
            f.write(f"Large Problem (N=12000):\n")
            f.write(f"  CPU: {cpu_med:.4f}s\n")
            f.write(f"  GPU: {gpu_med:.4f}s\n")
            f.write(f"  GPU Speedup: {speedup:.2f}x\n")
        
        print(f"\n✓ Summary saved to {summary_file}")
    
    print("\n✓ Parsing complete!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 parse_comparison.py <results_dir>")
        sys.exit(1)
    
    main(sys.argv[1])
