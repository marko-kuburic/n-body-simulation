#!/bin/bash
# ============================================================================
# Separate CPU and GPU Benchmarking Workflow
# Runs CPU locally, GPU in Docker, then compares results
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  N-Body Benchmarking: CPU + GPU (Docker) Comparison                       ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Parse options
RUN_CPU=true
RUN_GPU=true
COMPARE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            RUN_GPU=false
            COMPARE=false
            shift
            ;;
        --gpu-only)
            RUN_CPU=false
            shift
            ;;
        --no-compare)
            COMPARE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cpu-only] [--gpu-only] [--no-compare]"
            exit 1
            ;;
    esac
done

# ============================================================================
# PHASE 1: CPU Benchmark (Local)
# ============================================================================

if [ "$RUN_CPU" = true ]; then
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║ PHASE 1: CPU Benchmark (Local)                                            ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check prerequisites
    if [[ ! -x "./nbody_naive" ]]; then
        echo "ERROR: CPU binary './nbody_naive' not found"
        echo "Compiling..."
        make clean && make
    fi
    
    if ! python3 -c "import numpy, matplotlib" 2>/dev/null; then
        echo "ERROR: Required Python packages not found"
        echo "Install with: pip3 install numpy matplotlib"
        exit 1
    fi
    
    # Run CPU baseline
    echo "Running CPU baseline benchmark..."
    bash bench/run_cpu_baseline.sh
    
    # Get CPU results directory
    CPU_RESULTS=$(ls -td results/*/ | head -1)
    echo ""
    echo "Parsing CPU results..."
    python3 bench/parse_and_plot.py "${CPU_RESULTS}"
    
    echo ""
    echo "✓ Phase 1 complete"
    echo "  CPU results: ${CPU_RESULTS}"
    echo ""
else
    echo "Skipping CPU benchmark (use --cpu-only to run only CPU)"
    # Find the most recent CPU results with proper summary
    CPU_RESULTS=$(find results -name "selected_cpu_variants.json" -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
    if [ -z "$CPU_RESULTS" ]; then
        echo "ERROR: No existing CPU results found"
        echo "Run CPU benchmark first: bash bench/run_cpu_baseline.sh"
        exit 1
    fi
    echo "Using existing CPU results: ${CPU_RESULTS}"
    echo ""
fi

# ============================================================================
# PHASE 2: GPU Benchmark (Docker)
# ============================================================================

if [ "$RUN_GPU" = true ]; then
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║ PHASE 2: GPU Benchmark (Docker)                                           ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check Docker/Podman
    if ! command -v docker &> /dev/null && ! command -v podman &> /dev/null; then
        echo "ERROR: Neither docker nor podman found"
        echo "Install Docker or Podman for GPU benchmarking"
        exit 1
    fi
    
    # Run GPU benchmark
    echo "Running GPU benchmark via Docker..."
    bash bench/run_cuda_docker.sh
    
    # Get GPU results directory
    GPU_RESULTS=$(ls -td results/*/ | head -1)
    echo ""
    echo "Parsing GPU results..."
    python3 bench/parse_and_plot.py "${GPU_RESULTS}"
    
    echo ""
    echo "✓ Phase 2 complete"
    echo "  GPU results: ${GPU_RESULTS}"
    echo ""
else
    echo "Skipping GPU benchmark (use --gpu-only to run only GPU)"
    if [ "$RUN_CPU" = false ]; then
        GPU_RESULTS=$(ls -td results/*/ | grep -v comparison | head -1)
        echo "Using existing GPU results: ${GPU_RESULTS}"
    fi
    echo ""
fi

# ============================================================================
# PHASE 3: Comparison
# ============================================================================

if [ "$COMPARE" = true ]; then
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║ PHASE 3: CPU vs GPU Comparison                                            ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Find CPU and GPU results
    if [ -z "$CPU_RESULTS" ]; then
        echo "Finding CPU results..."
        CPU_RESULTS=$(find results -name "selected_cpu_variants.json" -printf '%h\n' | head -1)
    fi
    
    if [ -z "$GPU_RESULTS" ]; then
        echo "Finding GPU results..."
        GPU_RESULTS=$(find results -name "cuda_metadata.json" -printf '%h\n' | head -1)
    fi
    
    if [ -z "$CPU_RESULTS" ] || [ -z "$GPU_RESULTS" ]; then
        echo "ERROR: Could not find both CPU and GPU results"
        echo "CPU results: ${CPU_RESULTS:-not found}"
        echo "GPU results: ${GPU_RESULTS:-not found}"
        echo ""
        echo "Run benchmarks first:"
        echo "  bash bench/run_cpu_baseline.sh"
        echo "  bash bench/run_cuda_docker.sh"
        exit 1
    fi
    
    echo "Comparing results..."
    echo "  CPU: ${CPU_RESULTS}"
    echo "  GPU: ${GPU_RESULTS}"
    echo ""
    
    COMPARISON_DIR="results/comparison_$(date +%Y%m%d_%H%M%S)"
    python3 bench/compare_cpu_gpu.py "${CPU_RESULTS}" "${GPU_RESULTS}" --output "${COMPARISON_DIR}"
    
    echo ""
    echo "✓ Phase 3 complete"
    echo "  Comparison: ${COMPARISON_DIR}"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ BENCHMARKING COMPLETE                                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ "$RUN_CPU" = true ]; then
    echo "CPU Results:"
    echo "  Directory: ${CPU_RESULTS}"
    echo "  Summary:   ${CPU_RESULTS}summary.txt"
    echo "  Plots:     ${CPU_RESULTS}plots/"
    echo ""
fi

if [ "$RUN_GPU" = true ]; then
    echo "GPU Results:"
    echo "  Directory: ${GPU_RESULTS}"
    echo "  Summary:   ${GPU_RESULTS}cuda_summary.json"
    echo ""
fi

if [ "$COMPARE" = true ]; then
    echo "Comparison Results:"
    echo "  Directory: ${COMPARISON_DIR}"
    echo "  Summary:   ${COMPARISON_DIR}comparison_summary.txt"
    echo "  CSV:       ${COMPARISON_DIR}cpu_gpu_comparison.csv"
    echo "  Plots:     ${COMPARISON_DIR}plots/"
    echo ""
    echo "View comparison:"
    echo "  cat ${COMPARISON_DIR}comparison_summary.txt"
fi

echo "════════════════════════════════════════════════════════════════════════════"
