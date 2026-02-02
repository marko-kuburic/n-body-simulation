#!/bin/bash
# ============================================================================
# EXPERIMENT 2: CUDA Benchmark
# Benchmarks all available CUDA implementations
# ============================================================================

set -e

# Configuration
SIZES=(1000 2000 4000 6000 8000 10000 12000)
TIMESTEPS=(5 10 20 40 80)
REPEATS=5

# Setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "CUDA Benchmark"
echo "Started at: $(date)"
echo "Results directory: ${RESULTS_DIR}"
echo "=========================================="

# ============================================================================
# Detect CUDA binaries
# ============================================================================
echo ""
echo "Detecting CUDA binaries..."

CUDA_BINARIES=()

# Pattern-based detection
for pattern in "./nbody_cuda*" "./cuda_*" "./*cuda*"; do
    for binary in ${pattern}; do
        if [[ -x "${binary}" && -f "${binary}" ]]; then
            CUDA_BINARIES+=("${binary}")
        fi
    done
done

# Remove duplicates
CUDA_BINARIES=($(printf '%s\n' "${CUDA_BINARIES[@]}" | sort -u))

if [[ ${#CUDA_BINARIES[@]} -eq 0 ]]; then
    echo "WARNING: No CUDA binaries found matching patterns: nbody_cuda*, cuda_*, *cuda*"
    echo "Skipping CUDA benchmark."
    exit 0
fi

echo "Found ${#CUDA_BINARIES[@]} CUDA binary/binaries:"
for binary in "${CUDA_BINARIES[@]}"; do
    echo "  - ${binary}"
done

# ============================================================================
# Run benchmarks for each CUDA binary
# ============================================================================

for binary in "${CUDA_BINARIES[@]}"; do
    # Extract a clean name for this CUDA variant
    cuda_name=$(basename "${binary}")
    cuda_name="${cuda_name#nbody_}"
    cuda_name="${cuda_name%.cu}"
    
    echo ""
    echo "=========================================="
    echo "Benchmarking: ${binary} (variant: ${cuda_name})"
    echo "=========================================="
    
    # Problem size sweep
    for N in "${SIZES[@]}"; do
        for steps in 10; do  # Use steps=10 for size sweep
            echo "  N=${N}, steps=${steps}..."
            
            # Warmup
            echo "    Warmup..."
            ${binary} ${N} ${steps} > /dev/null 2>&1 || {
                echo "    WARNING: Warmup failed for ${binary} with N=${N}, steps=${steps}"
                echo "${binary} N=${N} steps=${steps} warmup_failed" >> "${RESULTS_DIR}/errors.log"
                continue
            }
            sleep 1
            
            # Timed runs
            for run in $(seq 1 ${REPEATS}); do
                logfile="${RESULTS_DIR}/cuda_${cuda_name}_N${N}_S${steps}_run${run}.log"
                echo "    Run ${run}/${REPEATS}..."
                
                if ${binary} ${N} ${steps} > "${logfile}" 2>&1; then
                    sleep 1
                else
                    echo "    WARNING: Run ${run} failed"
                    echo "${binary} N=${N} steps=${steps} run${run} failed" >> "${RESULTS_DIR}/errors.log"
                fi
            done
        done
    done
    
    # Timestep sweep (N=6000)
    for steps in "${TIMESTEPS[@]}"; do
        N=6000
        echo "  N=${N}, steps=${steps}..."
        
        # Warmup
        echo "    Warmup..."
        ${binary} ${N} ${steps} > /dev/null 2>&1 || {
            echo "    WARNING: Warmup failed for ${binary} with N=${N}, steps=${steps}"
            echo "${binary} N=${N} steps=${steps} warmup_failed" >> "${RESULTS_DIR}/errors.log"
            continue
        }
        sleep 1
        
        # Timed runs
        for run in $(seq 1 ${REPEATS}); do
            logfile="${RESULTS_DIR}/cuda_${cuda_name}_N${N}_S${steps}_run${run}.log"
            echo "    Run ${run}/${REPEATS}..."
            
            if ${binary} ${N} ${steps} > "${logfile}" 2>&1; then
                sleep 1
            else
                echo "    WARNING: Run ${run} failed"
                echo "${binary} N=${N} steps=${steps} run${run} failed" >> "${RESULTS_DIR}/errors.log"
            fi
        done
    done
done

# ============================================================================
# Collect metadata
# ============================================================================
echo ""
echo "Collecting system metadata..."

cat > "${RESULTS_DIR}/cuda_metadata.json" <<EOF
{
  "experiment": "cuda_benchmark",
  "timestamp": "${TIMESTAMP}",
  "hostname": "$(hostname)",
  "uname": "$(uname -a)",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "cuda_binaries": [
$(for binary in "${CUDA_BINARIES[@]}"; do echo "    \"${binary}\""; done | paste -sd,)
  ],
  "parameters": {
    "sizes": [$(IFS=,; echo "${SIZES[*]}")],
    "timesteps": [$(IFS=,; echo "${TIMESTEPS[*]}")],
    "repeats": ${REPEATS}
  }
}
EOF

# GPU info
echo "  GPU info..."
nvidia-smi > "${RESULTS_DIR}/gpu_info.txt" 2>&1 || echo "nvidia-smi not available" > "${RESULTS_DIR}/gpu_info.txt"

# NVCC version
echo "  NVCC info..."
nvcc --version > "${RESULTS_DIR}/nvcc_version.txt" 2>&1 || echo "nvcc not available" > "${RESULTS_DIR}/nvcc_version.txt"

# Git commit (if in repo)
if [[ -d .git ]]; then
    git rev-parse HEAD > "${RESULTS_DIR}/git_commit.txt" 2>&1 || true
fi

echo ""
echo "=========================================="
echo "CUDA Benchmark Complete"
echo "Results saved to: ${RESULTS_DIR}"
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python3 bench/parse_and_plot.py ${RESULTS_DIR}"
echo "  2. Check: ${RESULTS_DIR}/cuda_summary.json"
echo "  3. Check: ${RESULTS_DIR}/selected_cuda_variant.json"
