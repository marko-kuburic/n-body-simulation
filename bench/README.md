# Benchmarking Suite for N-Body Simulation

This directory contains a comprehensive benchmarking suite for rigorous performance evaluation of N-body simulation implementations (CPU and GPU variants).

## 📋 Overview

The suite performs three experimental phases:

1. **CPU Baseline**: Benchmarks all 4 CPU variants (Sequential Non-Sym, Sequential Sym, OpenMP Non-Sym, OpenMP Sym) with strong scaling, problem size sweep, and timestep sweep
2. **CUDA**: Benchmarks all available CUDA implementations across problem sizes and timesteps
3. **Final Comparison**: Direct comparison of selected best CPU variants vs best CUDA variant

## 🚀 Quick Start

### Run Complete Benchmark Suite

```bash
# Make sure binaries are compiled
make

# Run all experiments (takes 30-60 minutes depending on system)
bash bench/run_all_benchmarks.sh
```

This will:
- Run all CPU experiments (warmup + 5 repeats × multiple configs)
- Run all CUDA experiments (if CUDA binaries exist)
- Select best variants automatically
- Run final comparative benchmark (7 repeats for publication-quality results)
- Generate CSVs and publication-ready plots

### Run Individual Phases

```bash
# Phase 1: CPU baseline only
bash bench/run_cpu_baseline.sh
python3 bench/parse_and_plot.py results/TIMESTAMP/

# Phase 2: CUDA only (requires CUDA binaries)
bash bench/run_cuda_all.sh
python3 bench/parse_and_plot.py results/TIMESTAMP/

# Phase 3: Final comparison (requires phases 1 & 2)
bash bench/run_final_comparison.sh
python3 bench/parse_and_plot.py results/TIMESTAMP/ --final
```

## 📁 Directory Structure

```
bench/
├── run_all_benchmarks.sh     # Master script (runs everything)
├── run_cpu_baseline.sh        # CPU experiments (strong scaling, size sweep, timestep sweep)
├── run_cuda_all.sh            # CUDA experiments (all variants)
├── run_final_comparison.sh    # Final comparative benchmark
├── parse_and_plot.py          # Parse logs, compute stats, generate plots
└── README.md                  # This file

results/
└── YYYYMMDD_HHMMSS/           # Timestamped results directory
    ├── *.log                  # Raw output logs from each run
    ├── summary_median.csv     # CPU summary statistics
    ├── cuda_summary_median.csv # CUDA summary statistics
    ├── final_summary_median.csv # Final comparison statistics
    ├── raw_times.csv          # All individual run times
    ├── selected_cpu_variants.json # Best CPU variants
    ├── selected_cuda_variant.json # Best CUDA variant
    ├── summary.txt            # Human-readable summary
    ├── *_metadata.json        # System metadata
    └── plots/                 # Publication-ready plots
        ├── strong_scaling_N6000.png
        ├── time_vs_N.png
        ├── speedup_final.png
        └── *.svg              # Vector versions
```

## 🔧 Configuration

### Experiment Parameters

Edit the scripts to customize parameters:

**CPU Baseline** (`run_cpu_baseline.sh`):
```bash
STRONG_N=6000           # Fixed N for strong scaling
STRONG_STEPS=10         # Fixed steps for strong scaling
THREADS=(1 2 4 8 16 24 32)  # Thread counts to test
SIZES=(1000 2000 4000 6000 8000 10000 12000)  # Problem sizes
TIMESTEPS=(5 10 20 40 80)   # Timestep counts
REPEATS=5               # Number of repeats per config
```

**CUDA** (`run_cuda_all.sh`):
```bash
SIZES=(1000 2000 4000 6000 8000 10000 12000)
TIMESTEPS=(5 10 20 40 80)
REPEATS=5
```

**Final Comparison** (`run_final_comparison.sh`):
```bash
SIZES=(1000 2000 4000 6000 8000 10000 12000)
STEPS=5
REPEATS=7               # More repeats for final numbers
```

### OpenMP Environment

The scripts automatically set optimal OpenMP parameters:
```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=<varies>
```

## 📊 Output Files

### CSV Files

**`summary_median.csv`** - Aggregated statistics for all configurations:
```csv
experiment,variant,formulation,parallel,threads,N,steps,median_s,std_s,runs,interactions,time_per_interaction
strong_scaling,seq_nonsym,non-symmetric,False,1,6000,10,45.23,0.12,5,35994000,1.26e-06
...
```

**`final_summary_median.csv`** - Final comparison with speedups:
```csv
implementation,N,steps,median_s,std_s,runs,speedup,efficiency
cpu_seq_sym,6000,5,22.34,0.08,7,1.0,None
cpu_omp_nonsym,6000,5,0.98,0.03,7,22.8,0.71
cuda_tiled,6000,5,0.15,0.01,7,148.9,None
...
```

### JSON Files

**`selected_cpu_variants.json`** - Selected best CPU variants:
```json
{
  "canonical_config": {"N": 6000, "steps": 10, "threads": 32},
  "best_sequential": {
    "variant": "seq_sym",
    "median_s": 22.34,
    "std_s": 0.08
  },
  "best_openmp": {
    "variant": "omp_nonsym",
    "median_s": 0.98,
    "std_s": 0.03
  }
}
```

**`selected_cuda_variant.json`** - Selected best CUDA variant:
```json
{
  "canonical_config": {"N": 6000, "steps": 10},
  "best_cuda": {
    "variant": "cuda_tiled",
    "total_median_s": 0.15,
    "kernel_median_s": 0.12,
    "kernel_to_total_ratio": 0.80
  }
}
```

### Plots

All plots saved as both PNG (300 DPI) and SVG (vector) for publication:

1. **`strong_scaling_N6000.png`** - Thread scaling for N=6000
2. **`time_vs_N.png`** - Problem size scaling (log-log)
3. **`final_time_vs_N.png`** - Final comparison across problem sizes
4. **`speedup_final.png`** - Speedup relative to baseline
5. **`time_per_interaction_vs_N.png`** - Efficiency metric

## 🔬 Methodology

### Warmup and Repeats
- **1 warmup run** before timed repeats (discarded)
- **5 repeats** for baseline experiments
- **7 repeats** for final comparison (higher confidence)
- **1 second sleep** between runs (reduce thermal effects)

### Statistics
- **Median** used as central statistic (robust to outliers)
- **Standard deviation** computed for error bars
- Baseline: Sequential Non-Symmetric (original implementation)

### Selection Criteria
- **Canonical config**: N=6000, steps=10, threads=32
- **Best Sequential**: Lowest median time among sequential variants
- **Best OpenMP**: Lowest median time among OpenMP variants
- **Best CUDA**: Lowest **total** time (not just kernel time)

### Metrics Computed
- **Speedup**: `baseline_time / variant_time`
- **Parallel Efficiency**: `speedup / num_threads` (OpenMP only)
- **Time per Interaction**: `total_time / num_interactions`
- **Interactions**: N×(N-1) for non-symmetric, N×(N-1)/2 for symmetric

## 📦 Dependencies

### Required
- Bash shell
- Python 3.6+
- numpy (`pip3 install numpy`)
- matplotlib (`pip3 install matplotlib`)

### Optional
- CUDA toolkit (for GPU benchmarks)
- `nvidia-smi` (for GPU metadata)

Install Python dependencies:
```bash
pip3 install numpy matplotlib
```

Or with conda:
```bash
conda install numpy matplotlib
```

## 🐛 Error Handling

The scripts include robust error handling:
- Missing binaries: Warning + skip that phase
- Failed runs: Logged to `results/TIMESTAMP/errors.log`
- Continue on errors (unless `set -e` prevents it)
- Idempotent design (safe to re-run)

## 📈 Expected Runtime

Approximate times (depends on system):
- CPU Baseline: 15-30 minutes
- CUDA: 5-10 minutes
- Final Comparison: 5-10 minutes
- **Total: 30-60 minutes**

For quick testing, reduce `REPEATS` to 3 and limit `SIZES` array.

## 🎯 Usage Examples

### Minimal Quick Test
```bash
# Edit scripts to use:
REPEATS=3
SIZES=(1000 4000 8000)
THREADS=(1 8 32)

bash bench/run_all_benchmarks.sh
```

### CPU-Only Benchmark
```bash
bash bench/run_cpu_baseline.sh
python3 bench/parse_and_plot.py results/$(ls -t results/ | head -1)/
```

### Re-parse Existing Results
```bash
# If you want to regenerate plots without re-running experiments
python3 bench/parse_and_plot.py results/20260201_214334/ --final
```

### Custom Analysis
```bash
# Use the CSV files for your own analysis
import pandas as pd
df = pd.read_csv('results/TIMESTAMP/final_summary_median.csv')
print(df[df['N'] == 6000])
```

## 📝 Interpreting Results

The `summary.txt` file provides human-readable interpretation:
- **Winner per problem size**: Which implementation was fastest for each N
- **GPU crossover point**: At what N does GPU become faster than CPU
- **OpenMP efficiency**: Actual vs ideal speedup
- **Symmetric advantage**: Speedup from reduced operations

Example interpretation:
```
At N=6000, GPU is 152x faster than best CPU variant.
OpenMP achieves 71% parallel efficiency on average (32 threads).
Sequential symmetric formulation is 2.02x faster than non-symmetric.
```

## 🔄 Reproducibility

System metadata automatically collected:
- `uname -a` (kernel version)
- `lscpu` (CPU info)
- `nvidia-smi` (GPU info)
- `gcc --version`, `nvcc --version`
- Git commit hash (if in repo)
- OpenMP environment settings
- Timestamp (UTC)

Use this metadata to document experimental conditions in publications.

## 📚 Citation

If you use this benchmarking suite in your research, please include:
- System specifications from metadata files
- Methodology description (warmup, repeats, median)
- Plots from `plots/` directory
- Summary statistics from CSV files

## 🤝 Contributing

To extend the suite:
1. Add new experiment scripts following naming convention
2. Update `parse_and_plot.py` to recognize new log formats
3. Add new plotting functions for custom visualizations
4. Document in this README

## 📄 License

See repository LICENSE file.
