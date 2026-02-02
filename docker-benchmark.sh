#!/bin/bash
# Run full CUDA benchmark inside Docker

set -e

# Detect container runtime
if command -v docker &> /dev/null; then
    RUNTIME="docker"
    GPU_FLAG="--gpus all"
elif command -v podman &> /dev/null; then
    RUNTIME="podman"
    GPU_FLAG="--device nvidia.com/gpu=all"
else
    echo "Error: Neither docker nor podman found"
    exit 1
fi

echo "========================================"
echo "CUDA N-body Benchmark (Docker)"
echo "========================================"
echo ""

# Build if image doesn't exist
if ! $RUNTIME images | grep -q nbody-cuda; then
    echo "Building Docker image..."
    ./docker-build.sh
fi

# Run benchmark
$RUNTIME run --rm -it \
    $GPU_FLAG \
    --security-opt=label=disable \
    --ipc=host \
    -v "$PWD":/work:Z \
    -w /work \
    -e "NVCC_ARCH=${NVCC_ARCH:--gencode=arch=compute_80,code=sm_80}" \
    nbody-cuda \
    bash -c "make cuda && ./benchmark_cuda.sh"
