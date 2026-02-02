#!/bin/bash
# ============================================================================
# CUDA Benchmark via Docker
# Runs CUDA benchmarks inside Docker container with same params as CPU
# ============================================================================

set -e

# Configuration - MUST MATCH CPU benchmark parameters
SIZES=(1000 2000 4000 6000 8000 10000 12000)
TIMESTEPS=(5 10 20 40 80)
REPEATS=5
CANONICAL_N=6000
CANONICAL_STEPS=10

# Setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Detect container runtime
if command -v docker &> /dev/null; then
    RUNTIME="docker"
    GPU_FLAG="--gpus all"
elif command -v podman &> /dev/null; then
    RUNTIME="podman"
    GPU_FLAG="--device nvidia.com/gpu=all"
else
    echo "ERROR: Neither docker nor podman found"
    echo "Please install Docker or Podman for GPU benchmarking"
    exit 1
fi

echo "=========================================="
echo "CUDA Benchmark via Docker"
echo "Started at: $(date)"
echo "Results directory: ${RESULTS_DIR}"
echo "Using: ${RUNTIME}"
echo "=========================================="

# Build Docker image if needed
if ! ${RUNTIME} images | grep -q nbody-cuda; then
    echo "Building Docker image..."
    bash docker-build.sh
fi

# Build CUDA binary inside container
echo "Building CUDA binary inside container..."
${RUNTIME} run --rm \
    ${GPU_FLAG} \
    --security-opt=label=disable \
    -v "$PWD":/work:Z \
    -w /work \
    -e "NVCC_ARCH=${NVCC_ARCH:--gencode=arch=compute_80,code=sm_80}" \
    nbody-cuda \
    bash -c "make clean; make cuda" || {
        echo "ERROR: Failed to build CUDA binary"
        exit 1
    }

echo "✓ CUDA binary built successfully"
echo ""

# CUDA binary name inside container (relative to /work which is mounted PWD)
CUDA_BINARY="./nbody_cuda"

# ============================================================================
# SIZE SWEEP (vary N, fixed steps)
# ============================================================================
echo ""
echo "Phase 1: Problem Size Sweep (steps=${CANONICAL_STEPS})"
echo "=========================================="

for N in "${SIZES[@]}"; do
    echo "  Testing N=${N}..."
    
    # Warmup
    echo "    Warmup..."
    ${RUNTIME} run --rm \
        ${GPU_FLAG} \
        --security-opt=label=disable \
        -v "$PWD":/work:Z \
        -w /work \
        -e "NVCC_ARCH=${NVCC_ARCH:--gencode=arch=compute_80,code=sm_80}" \
        nbody-cuda \
        ${CUDA_BINARY} ${N} ${CANONICAL_STEPS} > /dev/null 2>&1 || {
            echo "    WARNING: Warmup failed for N=${N}"
            echo "cuda_docker N=${N} steps=${CANONICAL_STEPS} warmup_failed" >> "${RESULTS_DIR}/errors.log"
            continue
        }
    sleep 1
    
    # Timed runs
    for run in $(seq 1 ${REPEATS}); do
        logfile="${RESULTS_DIR}/cuda_docker_N${N}_S${CANONICAL_STEPS}_run${run}.log"
        echo "    Run ${run}/${REPEATS}..."
        
        if ${RUNTIME} run --rm \
            ${GPU_FLAG} \
            --security-opt=label=disable \
            -v "$PWD":/work:Z \
            -w /work \
            -e "NVCC_ARCH=${NVCC_ARCH:--gencode=arch=compute_80,code=sm_80}" \
            nbody-cuda \
            ${CUDA_BINARY} ${N} ${CANONICAL_STEPS} > "${logfile}" 2>&1; then
            sleep 1
        else
            echo "    WARNING: Run ${run} failed"
            echo "cuda_docker N=${N} steps=${CANONICAL_STEPS} run${run} failed" >> "${RESULTS_DIR}/errors.log"
        fi
    done
done

# ============================================================================
# TIMESTEP SWEEP (vary steps, fixed N)
# ============================================================================
echo ""
echo "Phase 2: Timestep Sweep (N=${CANONICAL_N})"
echo "=========================================="

for steps in "${TIMESTEPS[@]}"; do
    echo "  Testing steps=${steps}..."
    
    # Warmup
    echo "    Warmup..."
    ${RUNTIME} run --rm \
        ${GPU_FLAG} \
        --security-opt=label=disable \
        -v "$PWD":/work:Z \
        -w /work \
        -e "NVCC_ARCH=${NVCC_ARCH:--gencode=arch=compute_80,code=sm_80}" \
        nbody-cuda \
        ${CUDA_BINARY} ${CANONICAL_N} ${steps} > /dev/null 2>&1 || {
            echo "    WARNING: Warmup failed for steps=${steps}"
            echo "cuda_docker N=${CANONICAL_N} steps=${steps} warmup_failed" >> "${RESULTS_DIR}/errors.log"
            continue
        }
    sleep 1
    
    # Timed runs
    for run in $(seq 1 ${REPEATS}); do
        logfile="${RESULTS_DIR}/cuda_docker_N${CANONICAL_N}_S${steps}_run${run}.log"
        echo "    Run ${run}/${REPEATS}..."
        
        if ${RUNTIME} run --rm \
            ${GPU_FLAG} \
            --security-opt=label=disable \
            -v "$PWD":/work:Z \
            -w /work \
            -e "NVCC_ARCH=${NVCC_ARCH:--gencode=arch=compute_80,code=sm_80}" \
            nbody-cuda \
            ${CUDA_BINARY} ${CANONICAL_N} ${steps} > "${logfile}" 2>&1; then
            sleep 1
        else
            echo "    WARNING: Run ${run} failed"
            echo "cuda_docker N=${CANONICAL_N} steps=${steps} run${run} failed" >> "${RESULTS_DIR}/errors.log"
        fi
    done
done

# ============================================================================
# FINAL COMPARISON RUNS (same N and steps as final CPU)
# ============================================================================
echo ""
echo "Phase 3: Final Comparison Data (steps=5)"
echo "=========================================="

FINAL_STEPS=5
FINAL_REPEATS=7

for N in "${SIZES[@]}"; do
    echo "  Testing N=${N}, steps=${FINAL_STEPS}..."
    
    # Warmup
    echo "    Warmup..."
    ${RUNTIME} run --rm \
        ${GPU_FLAG} \
        --security-opt=label=disable \
        -v "$PWD":/work:Z \
        -w /work \
        -e "NVCC_ARCH=${NVCC_ARCH:--gencode=arch=compute_80,code=sm_80}" \
        nbody-cuda \
        ${CUDA_BINARY} ${N} ${FINAL_STEPS} > /dev/null 2>&1 || {
            echo "    WARNING: Warmup failed for N=${N}"
            echo "cuda_docker N=${N} steps=${FINAL_STEPS} warmup_failed" >> "${RESULTS_DIR}/errors.log"
            continue
        }
    sleep 1
    
    # Timed runs
    for run in $(seq 1 ${FINAL_REPEATS}); do
        logfile="${RESULTS_DIR}/final_cuda_docker_N${N}_S${FINAL_STEPS}_run${run}.log"
        echo "    Run ${run}/${FINAL_REPEATS}..."
        
        if ${RUNTIME} run --rm \
            ${GPU_FLAG} \
            --security-opt=label=disable \
            -v "$PWD":/work:Z \
            -w /work \
            -e "NVCC_ARCH=${NVCC_ARCH:--gencode=arch=compute_80,code=sm_80}" \
            nbody-cuda \
            ${CUDA_BINARY} ${N} ${FINAL_STEPS} > "${logfile}" 2>&1; then
            sleep 1
        else
            echo "    WARNING: Run ${run} failed"
            echo "cuda_docker N=${N} steps=${FINAL_STEPS} run${run} failed" >> "${RESULTS_DIR}/errors.log"
        fi
    done
done

# ============================================================================
# Collect metadata
# ============================================================================
echo ""
echo "Collecting system metadata..."

cat > "${RESULTS_DIR}/cuda_metadata.json" <<EOF
{
  "experiment": "cuda_docker_benchmark",
  "timestamp": "${TIMESTAMP}",
  "hostname": "$(hostname)",
  "uname": "$(uname -a)",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "runtime": "${RUNTIME}",
  "container_image": "nbody-cuda",
  "cuda_binary": "${CUDA_BINARY}",
  "parameters": {
    "sizes": [$(IFS=,; echo "${SIZES[*]}")],
    "timesteps": [$(IFS=,; echo "${TIMESTEPS[*]}")],
    "canonical_n": ${CANONICAL_N},
    "canonical_steps": ${CANONICAL_STEPS},
    "final_steps": ${FINAL_STEPS},
    "repeats": ${REPEATS},
    "final_repeats": ${FINAL_REPEATS}
  }
}
EOF

# GPU info from container
echo "  GPU info..."
${RUNTIME} run --rm \
    ${GPU_FLAG} \
    --security-opt=label=disable \
    nbody-cuda \
    nvidia-smi > "${RESULTS_DIR}/gpu_info.txt" 2>&1 || echo "nvidia-smi not available" > "${RESULTS_DIR}/gpu_info.txt"

# NVCC version from container
echo "  NVCC info..."
${RUNTIME} run --rm \
    ${GPU_FLAG} \
    --security-opt=label=disable \
    nbody-cuda \
    nvcc --version > "${RESULTS_DIR}/nvcc_version.txt" 2>&1 || echo "nvcc not available" > "${RESULTS_DIR}/nvcc_version.txt"

# Git commit (if in repo)
if [[ -d .git ]]; then
    git rev-parse HEAD > "${RESULTS_DIR}/git_commit.txt" 2>&1 || true
fi

echo ""
echo "=========================================="
echo "CUDA Docker Benchmark Complete"
echo "Results saved to: ${RESULTS_DIR}"
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python3 bench/parse_and_plot.py ${RESULTS_DIR}"
echo "  2. Check: ${RESULTS_DIR}/cuda_summary.json"
echo "  3. For comparison: python3 bench/compare_cpu_gpu.py <cpu_results_dir> ${RESULTS_DIR}"
