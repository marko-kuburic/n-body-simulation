# Benchmarking Checklist

Use this checklist to ensure proper benchmark execution and analysis.

## Pre-Flight Checks

### ☐ 1. Compile Binaries
```bash
make              # CPU binary
make cuda         # CUDA binary (if available)
ls -lh nbody nbody_cuda*
```

### ☐ 2. Verify Python Environment
```bash
python3 --version  # Should be 3.6+
python3 -c "import numpy, matplotlib" && echo "✓ OK"
```

### ☐ 3. Check System Resources
```bash
# CPU
lscpu | grep -E "Model name|CPU\(s\)|Thread"

# GPU (if applicable)
nvidia-smi --query-gpu=name,memory.total --format=csv

# Memory (need enough for large N)
free -h
```

### ☐ 4. Ensure Clean Environment
```bash
# No heavy processes running
top -b -n 1 | head -20

# Good thermal state (CPU not throttling)
# On Linux: sensors or cat /sys/class/thermal/thermal_zone*/temp
```

### ☐ 5. Make Scripts Executable
```bash
chmod +x bench/*.sh
ls -lh bench/*.sh  # Should show 'x' permission
```

## Configuration Checks

### ☐ 6. Review Parameters (Optional)

Edit `bench/run_cpu_baseline.sh`:
```bash
STRONG_N=6000           # OK for most systems
THREADS=(1 2 4 8 16 24 32)  # Adjust to your CPU core count
SIZES=(1000 2000 4000 6000 8000 10000 12000)  # OK
REPEATS=5               # Use 3 for quick test, 5-7 for publication
```

Edit `bench/run_final_comparison.sh`:
```bash
REPEATS=7               # More repeats for final results
```

### ☐ 7. Estimate Runtime
- Quick test (REPEATS=3, 3 sizes): ~10 minutes
- Full baseline (REPEATS=5, 7 sizes): ~30-60 minutes
- Publication-ready (REPEATS=7+): ~60-90 minutes

## Execution

### ☐ 8. Run Benchmark Suite
```bash
# Full benchmark (recommended)
bash bench/run_all_benchmarks.sh

# OR run phases individually:
bash bench/run_cpu_baseline.sh
bash bench/run_cuda_all.sh
bash bench/run_final_comparison.sh
```

### ☐ 9. Monitor Progress
```bash
# Watch output for errors
# Look for lines like:
#   "Run 1/5..."
#   "Testing N=6000..."
#   "Saved: results/..."

# Check results directory is being populated
watch -n 5 'ls -lh results/$(ls -t results | head -1)/*.log | wc -l'
```

### ☐ 10. Check for Errors
```bash
RESULTS=$(ls -td results/*/ | head -1)
if [ -f "${RESULTS}errors.log" ]; then
    echo "⚠ Errors occurred:"
    cat "${RESULTS}errors.log"
else
    echo "✓ No errors"
fi
```

## Post-Processing

### ☐ 11. Verify Parsing
```bash
RESULTS=$(ls -td results/*/ | head -1)

# Should see these files:
ls ${RESULTS}summary.txt
ls ${RESULTS}final_summary_median.csv
ls ${RESULTS}plots/*.png
```

### ☐ 12. Review Summary
```bash
RESULTS=$(ls -td results/*/ | head -1)
cat ${RESULTS}summary.txt
```

### ☐ 13. Inspect Plots
```bash
RESULTS=$(ls -td results/*/ | head -1)

# List all plots
ls -lh ${RESULTS}plots/

# View on desktop
xdg-open ${RESULTS}plots/final_time_vs_N.png 2>/dev/null || \
  open ${RESULTS}plots/final_time_vs_N.png 2>/dev/null || \
  echo "View manually: ${RESULTS}plots/"
```

### ☐ 14. Validate Results Sanity

Check that results make sense:
```bash
RESULTS=$(ls -td results/*/ | head -1)

# Sequential symmetric should be ~2x faster than non-symmetric
# OpenMP should show speedup (but maybe not linear)
# CUDA should be fastest for large N (but has overhead for small N)

# View selected variants
cat ${RESULTS}selected_cpu_variants.json
cat ${RESULTS}selected_cuda_variant.json 2>/dev/null
```

Expected patterns:
- ✓ Symmetric < Non-symmetric (sequential)
- ✓ OpenMP shows speedup vs sequential
- ✓ Speedup increases with N (up to a point)
- ✓ CUDA fastest for N ≥ 4000 (typically)

### ☐ 15. Check Statistical Quality

```bash
RESULTS=$(ls -td results/*/ | head -1)

# Standard deviations should be small relative to medians
# Look at std_s / median_s ratios in CSV
awk -F, 'NR>1 {if ($8>0) print $2, $8/$7}' ${RESULTS}final_summary_median.csv

# Should be < 0.05 (5%) for good runs
# If > 0.1 (10%), consider re-running with more repeats or idle system
```

## Data Export

### ☐ 16. Prepare for Publication

```bash
RESULTS=$(ls -td results/*/ | head -1)

# Create publication-ready package
mkdir -p paper_data
cp ${RESULTS}summary.txt paper_data/
cp ${RESULTS}final_summary_median.csv paper_data/
cp ${RESULTS}plots/*.svg paper_data/  # Vector graphics for paper
cp ${RESULTS}*_metadata.json paper_data/

# Create archive
tar -czf benchmark_results_$(date +%Y%m%d).tar.gz paper_data/
```

### ☐ 17. Document System Specs

```bash
RESULTS=$(ls -td results/*/ | head -1)

# Extract key specs for paper methods section
echo "System Specifications for Paper:"
echo "================================="
cat ${RESULTS}cpu_info.txt | grep "Model name"
cat ${RESULTS}cpu_info.txt | grep "CPU(s):"
cat ${RESULTS}gpu_info.txt | grep "Product Name" 2>/dev/null
cat ${RESULTS}gcc_version.txt | head -1
cat ${RESULTS}nvcc_version.txt | grep "release" 2>/dev/null
```

## Troubleshooting

### ☐ 18. If Results Look Wrong

**Problem**: Timings are inconsistent (high std)
- Solution: Ensure system is idle, reduce thermal throttling
- Re-run: `bash bench/run_final_comparison.sh` (after fixing issues)

**Problem**: CUDA not faster than CPU
- Check: Small N → GPU overhead dominates (expected)
- Check: `gpu_info.txt` to verify GPU is being used
- Solution: Focus on N ≥ 4000 for GPU advantage

**Problem**: OpenMP slower than sequential
- Check: Thread count matches CPU cores
- Check: `OMP_PLACES` and `OMP_PROC_BIND` set correctly
- Solution: Verify with `echo $OMP_NUM_THREADS`

**Problem**: Symmetric not faster than non-symmetric (sequential)
- Check: Implementations correct (fewer operations)
- Solution: This might indicate algorithmic difference in your code

### ☐ 19. Re-parse Without Re-running

If you just want to regenerate plots:
```bash
RESULTS=results/20260201_214334  # Your results directory
python3 bench/parse_and_plot.py ${RESULTS} --final
```

### ☐ 20. Clean Up (Optional)

```bash
# Keep only final results, remove intermediate
ls -d results/*/ | head -n -1 | xargs rm -rf

# Or keep all for reproducibility
echo "Keeping all results for reproducibility"
```

## Final Checklist

- [ ] Binaries compiled and tested
- [ ] Python dependencies installed
- [ ] System idle and cool
- [ ] Parameters configured
- [ ] Benchmarks run successfully
- [ ] Logs generated (100+ files typical)
- [ ] CSVs generated (summary_median.csv, final_summary_median.csv, etc.)
- [ ] Plots generated (8-10 PNG+SVG files)
- [ ] summary.txt reviewed and makes sense
- [ ] Selected variants identified (JSON files)
- [ ] Metadata collected (system specs)
- [ ] Results validated (sanity checks passed)
- [ ] Data exported for publication
- [ ] No errors in errors.log (or errors are documented)

## Quick Commands Reference

```bash
# Run everything
bash bench/run_all_benchmarks.sh

# Get latest results path
RESULTS=$(ls -td results/*/ | head -1)

# View summary
cat ${RESULTS}summary.txt

# View CSV in terminal
column -t -s, ${RESULTS}final_summary_median.csv | less -S

# Extract speedups
awk -F, 'NR>1 {print $1, $7}' ${RESULTS}final_summary_median.csv

# List plots
ls ${RESULTS}plots/

# Test parsing
python3 bench/test_parsing.py

# Re-parse existing results
python3 bench/parse_and_plot.py ${RESULTS} --final
```

## Time Budget

- [ ] Pre-flight checks: 5 minutes
- [ ] Configuration review: 5 minutes
- [ ] Benchmark execution: 30-60 minutes (unattended)
- [ ] Post-processing & validation: 10 minutes
- [ ] Data export & documentation: 10 minutes

**Total: ~60-90 minutes** (mostly automated)

## Success Criteria

✅ All phases completed without errors  
✅ Summary.txt shows expected performance patterns  
✅ Plots are clear and publication-ready  
✅ Standard deviations are reasonable (< 5-10%)  
✅ Selected variants make algorithmic sense  
✅ Metadata complete for reproducibility  
✅ CSV files ready for further analysis  

---

**Ready to run?** Start with:
```bash
bash bench/run_all_benchmarks.sh
```

Then come back to this checklist to verify everything worked correctly!
