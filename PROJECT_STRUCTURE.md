# N-Body Simulation Project Structure

## Essential Files (Kept)

### Source Code
- **nbody_naive.c** - CPU implementation with OpenMP parallelization
- **cuda/nbody_cuda.cu** - GPU CUDA implementation (naive + tiled kernels)
- **cuda/cuda_compat.h** - CUDA compatibility headers
- **cuda/cuda_wrapper.h** - CUDA wrapper utilities

### Build System
- **Makefile** - Build targets for CPU and CUDA binaries
- **Dockerfile** - CUDA 12.2 + GCC 11 environment (fixes host GCC 15 incompatibility)

### Docker Scripts
- **docker-build.sh** - Build Docker image
- **docker-run.sh** - Run container with GPU access (CDI)
- **docker-benchmark.sh** - Run benchmarks inside container

### Benchmark Suite
- **benchmark_comparison.sh** - Unified CPU vs GPU benchmark (1K-12K particles)
- **parse_comparison.py** - Extract timings, calculate speedup/GFlops, generate CSV
- **plot_comparison.py** - Generate 5 publication-ready figures

### Documentation
- **README.md** - Project overview
- **BUILD_AND_RUN.md** - Build and execution instructions
- **COMPARISON_GUIDE.md** - How to run CPU/GPU comparison benchmarks

### Results
- **results_comparison_20260201_214138/** - Partial benchmark data
- **results_comparison_20260201_214334/** - Complete benchmark with plots (LATEST)

## Deleted Files (Unnecessary)

### Old Scripts
- analyze_benchmark.py, analyze_benchmark_extended.py
- plot_results.py
- benchmark.sh, benchmark_rigorous.sh, benchmark_cuda.sh, run_benchmark_all.sh
- docker-test-compile.sh

### Old Documentation
- BENCHMARKING.md, GPU_BENCHMARKING.md, QUICKSTART_GPU.md, DOCKER_QUICKSTART.md

### Old Results
- results.txt, strong_scaling.csv, timestep_scaling.csv, problem_size_scaling.csv
- Old result directories (8 directories from failed/partial runs)

### Logs
- benchmark_output.txt, benchmark_final.log, benchmark_fixed.log
- comparison_run.log, benchmark_clean.log

### Obsolete Tools
- tools/ directory
- cuda/patch_cuda_for_gcc15.sh, cuda/build_with_docker.sh

## Quick Start

1. Build Docker image: `./docker-build.sh`
2. Run comparison benchmark: `./benchmark_comparison.sh`
3. Generate plots: `python3 plot_comparison.py results_comparison_YYYYMMDD_HHMMSS/`
4. Find figures in: `results_comparison_YYYYMMDD_HHMMSS/plots/`

All plots are publication-ready (PNG 300 DPI + SVG) for your term paper!
