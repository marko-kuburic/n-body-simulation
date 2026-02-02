#!/bin/bash
# ============================================================================
# EXPERIMENT 1: CPU Baseline Benchmark
# Benchmarks all four CPU variants with strong scaling, size sweep, timestep sweep
# ============================================================================

set -e

# Configuration
STRONG_N=6000
STRONG_STEPS=10
THREADS=(1 2 4 8 16 24 32)
SIZES=(1000 2000 4000 6000 8000 10000 12000)
TIMESTEPS=(5 10 20 40 80)
REPEATS=5

# Setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Set OpenMP environment
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Binary path
BINARY="./nbody_naive"

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Binary ${BINARY} not found or not executable"
    exit 1
fi

echo "=========================================="
echo "CPU Baseline Benchmark"
echo "Started at: $(date)"
echo "Results directory: ${RESULTS_DIR}"
echo "=========================================="

# ============================================================================
# STRONG SCALING (vary threads, fixed N and steps)
# ============================================================================
echo ""
echo "Phase 1: Strong Scaling (N=${STRONG_N}, steps=${STRONG_STEPS})"
echo "=========================================="
echo "  Note: Sequential variants only tested with 1 thread (they don't parallelize)"
echo ""

for threads in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=${threads}
    
    # Sequential variants only make sense with 1 thread
    if [[ ${threads} -eq 1 ]]; then
        echo "  Testing with ${threads} thread (all 4 variants)..."
    else
        echo "  Testing with ${threads} threads (OpenMP variants only)..."
    fi
    
    # Warmup
    echo "    Warmup..."
    ${BINARY} ${STRONG_N} ${STRONG_STEPS} > /dev/null 2>&1 || true
    sleep 1
    
    # Timed runs
    for run in $(seq 1 ${REPEATS}); do
        logfile="${RESULTS_DIR}/cpu_strong_T${threads}_N${STRONG_N}_S${STRONG_STEPS}_run${run}.log"
        echo "    Run ${run}/${REPEATS}..."
        ${BINARY} ${STRONG_N} ${STRONG_STEPS} > "${logfile}" 2>&1
        sleep 1
    done
done

# ============================================================================
# SIZE SWEEP (vary N, fixed threads and steps)
# ============================================================================
echo ""
echo "Phase 2: Problem Size Sweep (threads=32, steps=10)"
echo "=========================================="

export OMP_NUM_THREADS=32

for N in "${SIZES[@]}"; do
    echo "  Testing N=${N}..."
    
    # Warmup
    echo "    Warmup..."
    ${BINARY} ${N} 10 > /dev/null 2>&1 || true
    sleep 1
    
    # Timed runs
    for run in $(seq 1 ${REPEATS}); do
        logfile="${RESULTS_DIR}/cpu_size_N${N}_run${run}.log"
        echo "    Run ${run}/${REPEATS}..."
        ${BINARY} ${N} 10 > "${logfile}" 2>&1
        sleep 1
    done
done

# ============================================================================
# TIMESTEP SWEEP (vary steps, fixed N and threads)
# ============================================================================
echo ""
echo "Phase 3: Timestep Sweep (threads=32, N=6000)"
echo "=========================================="

export OMP_NUM_THREADS=32

for steps in "${TIMESTEPS[@]}"; do
    echo "  Testing steps=${steps}..."
    
    # Warmup
    echo "    Warmup..."
    ${BINARY} 6000 ${steps} > /dev/null 2>&1 || true
    sleep 1
    
    # Timed runs
    for run in $(seq 1 ${REPEATS}); do
        logfile="${RESULTS_DIR}/cpu_steps_S${steps}_run${run}.log"
        echo "    Run ${run}/${REPEATS}..."
        ${BINARY} 6000 ${steps} > "${logfile}" 2>&1
        sleep 1
    done
done

# ============================================================================
# Collect metadata
# ============================================================================
echo ""
echo "Collecting system metadata..."

cat > "${RESULTS_DIR}/cpu_metadata.json" <<EOF
{
  "experiment": "cpu_baseline",
  "timestamp": "${TIMESTAMP}",
  "hostname": "$(hostname)",
  "uname": "$(uname -a)",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "omp_settings": {
    "OMP_PLACES": "${OMP_PLACES}",
    "OMP_PROC_BIND": "${OMP_PROC_BIND}"
  },
  "parameters": {
    "strong_scaling": {
      "N": ${STRONG_N},
      "steps": ${STRONG_STEPS},
      "threads": [$(IFS=,; echo "${THREADS[*]}")]
    },
    "size_sweep": {
      "sizes": [$(IFS=,; echo "${SIZES[*]}")],
      "steps": 10,
      "threads": 32
    },
    "timestep_sweep": {
      "N": 6000,
      "timesteps": [$(IFS=,; echo "${TIMESTEPS[*]}")],
      "threads": 32
    },
    "repeats": ${REPEATS}
  }
}
EOF

# CPU info
echo "  CPU info..."
lscpu > "${RESULTS_DIR}/cpu_info.txt" 2>&1 || true

# GCC version
echo "  Compiler info..."
gcc --version > "${RESULTS_DIR}/gcc_version.txt" 2>&1 || true

# Git commit (if in repo)
if [[ -d .git ]]; then
    git rev-parse HEAD > "${RESULTS_DIR}/git_commit.txt" 2>&1 || true
fi

echo ""
echo "=========================================="
echo "CPU Baseline Benchmark Complete"
echo "Results saved to: ${RESULTS_DIR}"
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python3 bench/parse_and_plot.py ${RESULTS_DIR}"
echo "  2. Check: ${RESULTS_DIR}/cpu_summary.json"
echo "  3. Check: ${RESULTS_DIR}/selected_cpu_variants.json"
