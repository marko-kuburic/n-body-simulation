#!/bin/bash
# ============================================================================
# Master Benchmarking Script
# Runs complete benchmarking workflow: CPU baseline → CUDA → Final comparison
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║         N-Body Simulation: Complete Benchmarking Suite                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# Check prerequisites
# ============================================================================
echo "Checking prerequisites..."

if [[ ! -x "./nbody_naive" ]]; then
    echo "ERROR: CPU binary './nbody_naive' not found or not executable"
    echo "Please compile first: make"
    exit 1
fi

# Make scripts executable
chmod +x bench/run_cpu_baseline.sh
chmod +x bench/run_cuda_all.sh
chmod +x bench/run_final_comparison.sh

# Check Python dependencies
if ! python3 -c "import numpy, matplotlib" 2>/dev/null; then
    echo "ERROR: Required Python packages not found"
    echo "Install with: pip3 install numpy matplotlib"
    exit 1
fi

echo "✓ All prerequisites satisfied"
echo ""

# ============================================================================
# PHASE 1: CPU Baseline Benchmark
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 1: CPU Baseline Benchmark                                           ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

./bench/run_cpu_baseline.sh

# Find the most recent CPU results directory
CPU_RESULTS=$(ls -td results/*/ | head -1)
echo ""
echo "Parsing CPU results..."
python3 bench/parse_and_plot.py "${CPU_RESULTS}"

echo ""
echo "✓ Phase 1 complete: CPU baseline benchmarked and analyzed"
echo ""

# ============================================================================
# PHASE 2: CUDA Benchmark (if CUDA binaries exist)
# ============================================================================

# Check for CUDA binaries
CUDA_FOUND=false
for pattern in ./nbody_cuda* ./cuda_* ./*cuda*; do
    if [[ -x ${pattern} && -f ${pattern} ]]; then
        CUDA_FOUND=true
        break
    fi
done

if [[ "${CUDA_FOUND}" == "true" ]]; then
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║ PHASE 2: CUDA Benchmark                                                   ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    ./bench/run_cuda_all.sh
    
    # Find the most recent CUDA results directory
    CUDA_RESULTS=$(ls -td results/*/ | head -1)
    echo ""
    echo "Parsing CUDA results..."
    python3 bench/parse_and_plot.py "${CUDA_RESULTS}"
    
    echo ""
    echo "✓ Phase 2 complete: CUDA benchmarked and analyzed"
    echo ""
else
    echo "⚠ Phase 2 skipped: No CUDA binaries found"
    echo ""
fi

# ============================================================================
# PHASE 3: Final Comparison
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 3: Final Comparative Benchmark                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

./bench/run_final_comparison.sh

# Find the most recent final results directory
FINAL_RESULTS=$(ls -td results/*/ | head -1)
echo ""
echo "Parsing final comparison results..."
python3 bench/parse_and_plot.py "${FINAL_RESULTS}" --final

echo ""
echo "✓ Phase 3 complete: Final comparison analyzed"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ BENCHMARKING COMPLETE                                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Final results directory: ${FINAL_RESULTS}"
echo ""
echo "Key outputs:"
echo "  • Summary:              ${FINAL_RESULTS}summary.txt"
echo "  • Median times:         ${FINAL_RESULTS}final_summary_median.csv"
echo "  • Raw times:            ${FINAL_RESULTS}final_raw_times.csv"
echo "  • Plots:                ${FINAL_RESULTS}plots/"
echo ""
echo "View summary:"
echo "  cat ${FINAL_RESULTS}summary.txt"
echo ""
echo "View plots:"
echo "  ls -lh ${FINAL_RESULTS}plots/"
echo ""

# Display summary if it exists
if [[ -f "${FINAL_RESULTS}summary.txt" ]]; then
    echo "════════════════════════════════════════════════════════════════════════════"
    cat "${FINAL_RESULTS}summary.txt"
fi
