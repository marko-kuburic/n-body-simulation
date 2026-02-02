#!/bin/bash
#
# Unified CPU vs GPU Benchmarking Suite
# ======================================
# Runs comparable benchmarks on both OpenMP (CPU) and CUDA (GPU)
# with identical problem sizes for direct performance comparison
#

set -e

RESULTS_DIR="results_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "================================================================================"
echo "CPU vs GPU N-BODY COMPARISON BENCHMARK"
echo "================================================================================"
echo "Start time:     $(date)"
echo "Results dir:    $RESULTS_DIR"
echo "================================================================================"
echo ""

# Problem sizes for comparison
PROBLEM_SIZES=(1000 2000 4000 6000 8000 10000 12000)
TIMESTEPS=10
REPEATS=3

# CPU threads to test (use max available for best CPU performance)
MAX_THREADS=$(nproc)
CPU_THREADS=$MAX_THREADS

# ============================================================================
# BUILD GPU BINARY IN DOCKER (to avoid GCC 15 host issues)
# ============================================================================

echo "Building CUDA binary in Docker container..."
podman run --rm \
    --security-opt=label=disable \
    -v "$PWD":/work:Z \
    -w /work \
    localhost/nbody-cuda \
    bash -c "make cuda" > /dev/null 2>&1

if [ ! -f "./nbody_cuda" ]; then
    echo "ERROR: Failed to build CUDA binary"
    exit 1
fi
echo "✓ CUDA binary ready"
echo ""

# ============================================================================
# EXPERIMENT 1: Problem Size Scaling (CPU vs GPU)
# ============================================================================

echo "================================================================================"
echo "EXPERIMENT: PROBLEM SIZE SCALING (CPU vs GPU)"
echo "================================================================================"
echo "Fixed: timesteps=$TIMESTEPS, CPU threads=$CPU_THREADS"
echo "Variable: N = {${PROBLEM_SIZES[@]}}"
echo ""

for N in "${PROBLEM_SIZES[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "N=$N particles"
    echo "--------------------------------------------------------------------------------"
    
    # Run CPU benchmark
    echo "  [CPU] Running with $CPU_THREADS threads..."
    for rep in $(seq 1 $REPEATS); do
        echo "    Repetition $rep/$REPEATS..."
        OMP_NUM_THREADS=$CPU_THREADS ./nbody_naive $N $TIMESTEPS >> "$RESULTS_DIR/cpu_size_N${N}_rep${rep}.log" 2>&1
    done
    
    # Run GPU benchmark (in Docker)
    echo "  [GPU] Running CUDA..."
    for rep in $(seq 1 $REPEATS); do
        echo "    Repetition $rep/$REPEATS..."
        podman run --rm \
            --device nvidia.com/gpu=all \
            --security-opt=label=disable \
            --ipc=host \
            -v "$PWD":/work:Z \
            -w /work \
            localhost/nbody-cuda \
            bash -c "/work/nbody_cuda $N $TIMESTEPS" >> "$RESULTS_DIR/gpu_size_N${N}_rep${rep}.log" 2>&1
    done
    
    echo "  ✓ Complete"
done

echo ""
echo "✓ Problem size scaling complete"
echo ""

# ============================================================================
# EXPERIMENT 2: Fixed Large Problem (Best Performance Comparison)
# ============================================================================

echo "================================================================================"
echo "EXPERIMENT: LARGE PROBLEM PERFORMANCE"
echo "================================================================================"
echo "Single large problem to show peak performance difference"
echo "N=12000, timesteps=20"
echo ""

LARGE_N=12000
LARGE_STEPS=20

echo "  [CPU] Running with $CPU_THREADS threads..."
for rep in $(seq 1 $REPEATS); do
    echo "    Repetition $rep/$REPEATS..."
    OMP_NUM_THREADS=$CPU_THREADS ./nbody_naive $LARGE_N $LARGE_STEPS >> "$RESULTS_DIR/cpu_large_rep${rep}.log" 2>&1
done

echo "  [GPU] Running CUDA..."
for rep in $(seq 1 $REPEATS); do
    echo "    Repetition $rep/$REPEATS..."
    podman run --rm \
        --device nvidia.com/gpu=all \
        --security-opt=label=disable \
        --ipc=host \
        -v "$PWD":/work:Z \
        -w /work \
        localhost/nbody-cuda \
        bash -c "/work/nbody_cuda $LARGE_N $LARGE_STEPS" >> "$RESULTS_DIR/gpu_large_rep${rep}.log" 2>&1
done

echo "  ✓ Complete"
echo ""

# ============================================================================
# COLLECT SYSTEM METADATA
# ============================================================================

echo "Collecting system metadata..."

cat > "$RESULTS_DIR/metadata.json" << EOF
{
  "date": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "cpu": {
    "model": "$(lscpu | grep 'Model name' | cut -d: -f2 | xargs)",
    "cores": $(nproc --all),
    "threads_used": $CPU_THREADS,
    "governor": "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  },
  "gpu": {
    "model": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
    "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
    "memory_gb": "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1/1024}')"
  },
  "problem_sizes": [$(IFS=,; echo "${PROBLEM_SIZES[*]}")],
  "timesteps": $TIMESTEPS,
  "repeats": $REPEATS
}
EOF

echo "✓ Metadata saved"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "================================================================================"
echo "BENCHMARKING COMPLETE"
echo "================================================================================"
echo "End time:       $(date)"
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Next steps:"
echo "  1. Parse results: python3 parse_comparison.py $RESULTS_DIR"
echo "  2. Generate plots: python3 plot_comparison.py $RESULTS_DIR"
echo "================================================================================"
