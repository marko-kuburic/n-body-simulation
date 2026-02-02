# Quick Start: N-Body Benchmarking Suite

## Prerequisites Check

```bash
# 1. Verify binaries exist
ls -lh nbody nbody_cuda*

# 2. Check Python dependencies
python3 -c "import numpy, matplotlib" && echo "✓ Python packages OK"

# 3. Check system (optional but recommended)
lscpu | grep "Model name"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU"
```

## Run Complete Benchmark (Recommended)

```bash
# This runs everything: CPU → CUDA → Final comparison
bash bench/run_all_benchmarks.sh
```

**Expected time**: 30-60 minutes  
**Output**: `results/YYYYMMDD_HHMMSS/` with CSVs, plots, and summary

## Quick Test (5 minutes)

Edit scripts first to reduce runtime:

```bash
# In bench/run_cpu_baseline.sh, change:
THREADS=(1 8 32)
SIZES=(2000 6000 10000)
REPEATS=3

# Then run:
bash bench/run_all_benchmarks.sh
```

## Run Individual Phases

### Phase 1: CPU Baseline

```bash
bash bench/run_cpu_baseline.sh

# Parse results (finds most recent)
RESULTS=$(ls -td results/*/ | head -1)
python3 bench/parse_and_plot.py $RESULTS
```

**Tests**: Strong scaling, size sweep, timestep sweep  
**Time**: ~15-30 minutes

### Phase 2: CUDA (if available)

```bash
bash bench/run_cuda_all.sh

# Parse results
RESULTS=$(ls -td results/*/ | head -1)
python3 bench/parse_and_plot.py $RESULTS
```

**Tests**: All CUDA variants across sizes and timesteps  
**Time**: ~5-10 minutes

### Phase 3: Final Comparison

```bash
# Requires phases 1 & 2 to be completed first
bash bench/run_final_comparison.sh

# Parse with --final flag
RESULTS=$(ls -td results/*/ | head -1)
python3 bench/parse_and_plot.py $RESULTS --final
```

**Tests**: Best variants head-to-head  
**Time**: ~5-10 minutes

## View Results

```bash
# Most recent results directory
RESULTS=$(ls -td results/*/ | head -1)

# View summary
cat ${RESULTS}summary.txt

# View plots
ls ${RESULTS}plots/

# View CSV data
column -t -s, ${RESULTS}final_summary_median.csv | less -S
```

## Key Output Files

```
results/YYYYMMDD_HHMMSS/
├── summary.txt                    # Human-readable summary
├── final_summary_median.csv       # Final comparison data
├── selected_cpu_variants.json     # Best CPU variants
├── selected_cuda_variant.json     # Best CUDA variant
└── plots/
    ├── final_time_vs_N.png        # Main comparison plot
    ├── speedup_final.png          # Speedup visualization
    └── *.svg                      # Vector formats
```

## Common Issues

### "Binary not found"
```bash
# Compile first
make
make cuda  # if you have CUDA
```

### "Python packages not found"
```bash
pip3 install numpy matplotlib
# or
conda install numpy matplotlib
```

### "Permission denied"
```bash
chmod +x bench/*.sh
```

### Re-parse without re-running
```bash
# If you want to regenerate plots/CSVs without re-running experiments
python3 bench/parse_and_plot.py results/YYYYMMDD_HHMMSS/ --final
```

## Customization

Edit the scripts to change:
- `REPEATS`: Number of timed runs (5 for baseline, 7 for final)
- `THREADS`: Thread counts to test
- `SIZES`: Problem sizes (N values)
- `TIMESTEPS`: Time steps to test

## Next Steps

1. **Run benchmarks**: `bash bench/run_all_benchmarks.sh`
2. **Check summary**: View `results/*/summary.txt`
3. **Analyze plots**: Open `results/*/plots/`
4. **Export data**: Use CSV files for further analysis
5. **Document**: Include metadata files when publishing results

## Example: Extract Best Results

```bash
# Get latest results
RESULTS=$(ls -td results/*/ | head -1)

# Best sequential
jq -r '.best_sequential | "\(.variant): \(.median_s)s"' ${RESULTS}selected_cpu_variants.json

# Best OpenMP
jq -r '.best_openmp | "\(.variant): \(.median_s)s"' ${RESULTS}selected_cpu_variants.json

# Best CUDA
jq -r '.best_cuda | "\(.variant): \(.total_median_s)s"' ${RESULTS}selected_cuda_variant.json

# Speedup at N=6000
awk -F, '$2=="6000" {print $1, $7"x"}' ${RESULTS}final_summary_median.csv
```

## Help & Documentation

- Full documentation: `bench/README.md`
- Script help: Read comments in each `.sh` file
- Parsing help: `python3 bench/parse_and_plot.py --help`

## Performance Tips

For fastest results:
1. Reduce `REPEATS` to 3
2. Limit `SIZES` to 3-4 values
3. Limit `THREADS` to [1, max_cores]
4. Run on idle system (no other heavy processes)
5. Ensure good cooling (thermal throttling affects results)

For publication-quality results:
1. Use `REPEATS=7` or more
2. Full `SIZES` and `THREADS` arrays
3. Run overnight on dedicated machine
4. Include metadata in your paper
