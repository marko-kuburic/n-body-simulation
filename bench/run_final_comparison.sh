#!/bin/bash
# ============================================================================
# EXPERIMENT 3: Final Comparative Benchmark
# Compares selected best CPU variants vs best CUDA variant
# ============================================================================

set -e

# Configuration
SIZES=(1000 2000 4000 6000 8000 10000 12000)
STEPS=5
REPEATS=7  # More repeats for final numbers

# Setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Set OpenMP environment
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=32

# Check if selection files exist
CPU_SELECTION_FILE=""
CUDA_SELECTION_FILE=""

# Find most recent CPU selection
for dir in results/*/; do
    if [[ -f "${dir}/selected_cpu_variants.json" ]]; then
        CPU_SELECTION_FILE="${dir}/selected_cpu_variants.json"
    fi
done

# Find most recent CUDA selection
for dir in results/*/; do
    if [[ -f "${dir}/selected_cuda_variant.json" ]]; then
        CUDA_SELECTION_FILE="${dir}/selected_cuda_variant.json"
    fi
done

if [[ -z "${CPU_SELECTION_FILE}" ]]; then
    echo "ERROR: No selected_cpu_variants.json found. Run CPU baseline first."
    exit 1
fi

echo "=========================================="
echo "Final Comparative Benchmark"
echo "Started at: $(date)"
echo "Results directory: ${RESULTS_DIR}"
echo "CPU selection: ${CPU_SELECTION_FILE}"
if [[ -n "${CUDA_SELECTION_FILE}" ]]; then
    echo "CUDA selection: ${CUDA_SELECTION_FILE}"
else
    echo "No CUDA selection found (CPU-only comparison)"
fi
echo "=========================================="

# Copy selection files
cp "${CPU_SELECTION_FILE}" "${RESULTS_DIR}/selected_cpu_variants.json"
if [[ -n "${CUDA_SELECTION_FILE}" ]]; then
    cp "${CUDA_SELECTION_FILE}" "${RESULTS_DIR}/selected_cuda_variant.json"
fi

# ============================================================================
# CPU Benchmarks (selected variants will be parsed from logs)
# ============================================================================
echo ""
echo "Phase 1: CPU Benchmarks (all variants)"
echo "=========================================="

BINARY="./nbody_naive"

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Binary ${BINARY} not found or not executable"
    exit 1
fi

for N in "${SIZES[@]}"; do
    echo "  Testing N=${N}, steps=${STEPS}..."
    
    # Warmup
    echo "    Warmup..."
    ${BINARY} ${N} ${STEPS} > /dev/null 2>&1 || true
    sleep 1
    
    # Timed runs
    for run in $(seq 1 ${REPEATS}); do
        logfile="${RESULTS_DIR}/final_cpu_N${N}_S${STEPS}_run${run}.log"
        echo "    Run ${run}/${REPEATS}..."
        ${BINARY} ${N} ${STEPS} > "${logfile}" 2>&1
        sleep 1
    done
done

# ============================================================================
# CUDA Benchmark (if CUDA binary exists)
# ============================================================================

if [[ -n "${CUDA_SELECTION_FILE}" ]]; then
    # Extract CUDA binary path from selection file (we'll need to be flexible here)
    # For now, try to find CUDA binaries again
    
    echo ""
    echo "Phase 2: CUDA Benchmark (best variant)"
    echo "=========================================="
    
    # Detect CUDA binaries
    CUDA_BINARIES=()
    for pattern in "./nbody_cuda*" "./cuda_*" "./*cuda*"; do
        for binary in ${pattern}; do
            if [[ -x "${binary}" && -f "${binary}" ]]; then
                CUDA_BINARIES+=("${binary}")
            fi
        done
    done
    CUDA_BINARIES=($(printf '%s\n' "${CUDA_BINARIES[@]}" | sort -u))
    
    if [[ ${#CUDA_BINARIES[@]} -eq 0 ]]; then
        echo "WARNING: No CUDA binaries found. Skipping CUDA benchmark."
    else
        # Use first CUDA binary (or all of them - parsing will select best)
        for binary in "${CUDA_BINARIES[@]}"; do
            cuda_name=$(basename "${binary}")
            cuda_name="${cuda_name#nbody_}"
            cuda_name="${cuda_name%.cu}"
            
            echo "  Testing ${binary} (${cuda_name})..."
            
            for N in "${SIZES[@]}"; do
                echo "    N=${N}, steps=${STEPS}..."
                
                # Warmup
                echo "      Warmup..."
                ${binary} ${N} ${STEPS} > /dev/null 2>&1 || {
                    echo "      WARNING: Warmup failed"
                    continue
                }
                sleep 1
                
                # Timed runs
                for run in $(seq 1 ${REPEATS}); do
                    logfile="${RESULTS_DIR}/final_cuda_${cuda_name}_N${N}_S${STEPS}_run${run}.log"
                    echo "      Run ${run}/${REPEATS}..."
                    
                    if ${binary} ${N} ${STEPS} > "${logfile}" 2>&1; then
                        sleep 1
                    else
                        echo "      WARNING: Run ${run} failed"
                        echo "${binary} N=${N} steps=${STEPS} run${run} failed" >> "${RESULTS_DIR}/errors.log"
                    fi
                done
            done
        done
    fi
fi

# ============================================================================
# Collect metadata
# ============================================================================
echo ""
echo "Collecting system metadata..."

cat > "${RESULTS_DIR}/final_metadata.json" <<EOF
{
  "experiment": "final_comparison",
  "timestamp": "${TIMESTAMP}",
  "hostname": "$(hostname)",
  "uname": "$(uname -a)",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "omp_settings": {
    "OMP_PLACES": "${OMP_PLACES}",
    "OMP_PROC_BIND": "${OMP_PROC_BIND}",
    "OMP_NUM_THREADS": 32
  },
  "parameters": {
    "sizes": [$(IFS=,; echo "${SIZES[*]}")],
    "steps": ${STEPS},
    "repeats": ${REPEATS}
  }
}
EOF

# System info
lscpu > "${RESULTS_DIR}/cpu_info.txt" 2>&1 || true
nvidia-smi > "${RESULTS_DIR}/gpu_info.txt" 2>&1 || echo "nvidia-smi not available" > "${RESULTS_DIR}/gpu_info.txt"

# Git commit (if in repo)
if [[ -d .git ]]; then
    git rev-parse HEAD > "${RESULTS_DIR}/git_commit.txt" 2>&1 || true
fi

echo ""
echo "=========================================="
echo "Final Comparative Benchmark Complete"
echo "Results saved to: ${RESULTS_DIR}"
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python3 bench/parse_and_plot.py ${RESULTS_DIR} --final"
echo "  2. Check: ${RESULTS_DIR}/final_summary_median.csv"
echo "  3. Check plots in: ${RESULTS_DIR}/plots/"
echo "  4. Read: ${RESULTS_DIR}/summary.txt"
