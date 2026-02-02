# 🔬 Rigorous Benchmarking Suite

## NEW: Publication-Ready Benchmark Tools

A comprehensive benchmarking suite has been added to enable rigorous, reproducible performance evaluation of all N-body variants.

### 📦 Location

All benchmarking tools are in the `bench/` directory:

```
bench/
├── run_all_benchmarks.sh      # Master script - run everything
├── run_cpu_baseline.sh         # CPU experiments
├── run_cuda_all.sh             # CUDA experiments  
├── run_final_comparison.sh     # Final comparison
├── parse_and_plot.py           # Analysis & plotting
├── test_parsing.py             # Validation tests
├── README.md                   # Full documentation
├── QUICKSTART.md               # Quick start guide
├── CHECKLIST.md                # Step-by-step checklist
└── IMPLEMENTATION_SUMMARY.md   # Design overview
```

### 🚀 Quick Start

**Run complete benchmark suite** (30-60 minutes):
```bash
bash bench/run_all_benchmarks.sh
```

This will:
1. ✅ Run CPU baseline (strong scaling, size sweep, timestep sweep)
2. ✅ Run CUDA benchmarks (all variants if available)
3. ✅ Automatically select best variants
4. ✅ Run final head-to-head comparison
5. ✅ Generate publication-ready plots (PNG + SVG)
6. ✅ Produce CSV files and human-readable summary

**Results saved to**: `results/YYYYMMDD_HHMMSS/`

### 📊 What You Get

#### Data Files
- `final_summary_median.csv` - Median times and speedups
- `summary_median.csv` - All configurations benchmarked
- `raw_times.csv` - Individual run times
- `selected_cpu_variants.json` - Best CPU implementations
- `selected_cuda_variant.json` - Best GPU implementation

#### Plots (300 DPI PNG + Vector SVG)
- `final_time_vs_N.png` - Performance comparison
- `speedup_final.png` - Speedup visualization
- `strong_scaling_N6000.png` - Thread scaling
- `time_per_interaction_vs_N.png` - Efficiency metric

#### Analysis
- `summary.txt` - Human-readable findings with interpretation

### 🔬 Methodology

The suite implements scientific best practices:

- **Warmup**: 1 run before timing (eliminates cold-start)
- **Repeats**: 5-7 timed runs per configuration
- **Statistics**: Median (robust to outliers) + standard deviation
- **Thermal management**: 1-second sleep between runs
- **OpenMP tuning**: `OMP_PLACES=cores`, `OMP_PROC_BIND=close`
- **Auto-selection**: Best variants chosen at canonical N=6000
- **Reproducibility**: Full system metadata captured

### 📈 Expected Results

At N=6000, steps=10, threads=32:

| Implementation | Expected | Reason |
|---------------|----------|--------|
| **Best Sequential** | Symmetric | 50% fewer operations |
| **Best OpenMP** | Non-Symmetric | No sync overhead |
| **Best Overall** | CUDA | Massive parallelism |

**Typical speedups** (your hardware may vary):
- Sequential Symmetric vs Non-Sym: **~2x**
- OpenMP vs Sequential: **10-25x** (32 cores)
- CUDA vs Sequential: **50-200x** (depends on GPU)

### 🎯 Use Cases

#### For Research Papers
```bash
bash bench/run_all_benchmarks.sh
# Use plots from results/*/plots/ in your paper
# Cite methodology and system specs from metadata
```

#### For Quick Testing
```bash
# Edit scripts: REPEATS=3, limit SIZES array
bash bench/run_all_benchmarks.sh
# Results in ~10 minutes
```

#### For Teaching/Demos
```bash
# Run CPU-only
bash bench/run_cpu_baseline.sh
python3 bench/parse_and_plot.py results/$(ls -t results | head -1)/
# Show strong_scaling plots to demonstrate Amdahl's law
```

### 📚 Documentation

- **Full details**: See [bench/README.md](bench/README.md)
- **Quick reference**: See [bench/QUICKSTART.md](bench/QUICKSTART.md)  
- **Step-by-step**: See [bench/CHECKLIST.md](bench/CHECKLIST.md)
- **Design**: See [bench/IMPLEMENTATION_SUMMARY.md](bench/IMPLEMENTATION_SUMMARY.md)

### ✅ Validation

Test the parsing pipeline:
```bash
python3 bench/test_parsing.py
# Should output: ✓ ALL TESTS PASSED
```

### 🔧 Requirements

- Python 3.6+ with `numpy` and `matplotlib`
- Compiled binaries: `nbody` (CPU), `nbody_cuda` (GPU, optional)
- Bash shell

Install Python dependencies:
```bash
pip3 install numpy matplotlib
```

### 📝 Example Output

```
════════════════════════════════════════════════════════════════
SELECTED IMPLEMENTATIONS:
Best Sequential:  seq_sym (22.34 ± 0.08 s)
Best OpenMP:      omp_nonsym (0.98 ± 0.03 s)  
Best CUDA:        cuda (0.15 ± 0.01 s)

PERFORMANCE RESULTS (median times in seconds):
N        Implementation            Time (s)     Speedup    
─────────────────────────────────────────────────────────────
6000     seq_sym                   22.3400      1.00x      
6000     omp_nonsym                0.9800       22.79x     
6000     CUDA                      0.1500       148.93x    ***

INTERPRETATION:
• At N=6000, GPU is 148.9x faster than best CPU variant.
• OpenMP achieves 71.2% parallel efficiency (32 threads).
• Sequential symmetric is 2.02x faster than non-symmetric.
════════════════════════════════════════════════════════════════
```

### 🎓 For Publication

The benchmark suite automatically collects all information needed for the "Experimental Setup" section of your paper:

```latex
% Example methods section
All experiments performed with 1 warmup run followed by 7 timed 
repetitions. Median times reported with standard deviations. 
OpenMP configured with OMP_PLACES=cores and OMP_PROC_BIND=close.

Hardware: [from cpu_info.txt]
- CPU: Intel Xeon ...
- GPU: NVIDIA RTX ...
- Memory: ...

Software: [from metadata]
- GCC: 11.2.0
- NVCC: 12.0
- OpenMP: 4.5
```

### 🤝 Contributing

The suite is designed to be extensible:
- Add new experiment scripts in `bench/`
- Update `parse_and_plot.py` for new log formats
- Extend plotting functions for custom visualizations

### 📄 License

Same as main repository.

---

**Ready to benchmark?**

```bash
cd bench
bash run_all_benchmarks.sh
```

Then check `results/*/summary.txt` for your findings!
