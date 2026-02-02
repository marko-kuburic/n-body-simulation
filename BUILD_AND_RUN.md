# Complete Build and Run Instructions

## Project Structure

```
HPC/
├── nbody_naive.c                    # CPU implementation (4 variants)
├── cuda/
│   └── nbody_cuda.cu                # GPU implementation (2 kernels)
├── Makefile                         # Build system
│
├── benchmark_rigorous.sh            # CPU benchmarking
├── benchmark_cuda.sh                # GPU benchmarking  
├── run_benchmark_all.sh             # Unified pipeline
│
├── analyze_benchmark.py             # CPU parser (original)
├── analyze_benchmark_extended.py    # Extended parser (CPU+GPU)
├── plot_results.py                  # Visualization
│
├── README.md                        # Main documentation
├── BENCHMARKING.md                  # CPU methodology
├── GPU_BENCHMARKING.md              # GPU methodology
└── QUICKSTART_GPU.md                # 5-minute GPU guide
```

## Building

### 1. CPU Only
```bash
make
# Or explicitly:
make nbody_naive
```

**Output**: `nbody_naive` executable

**Test**:
```bash
./nbody_naive 1000 5
```

### 2. GPU Only
```bash
make cuda
```

**Requirements**:
- CUDA Toolkit 11.0+
- NVIDIA GPU with compute capability 6.0+

**Architecture Override** (if needed):
```bash
# Find your GPU's compute capability: https://developer.nvidia.com/cuda-gpus
# RTX 30-series (Ampere, 8.6):
NVCC_ARCH="-gencode=arch=compute_86,code=sm_86" make cuda

# RTX 40-series (Ada, 8.9):
NVCC_ARCH="-gencode=arch=compute_89,code=sm_89" make cuda

# Multiple architectures:
NVCC_ARCH="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86" make cuda
```

**Output**: `nbody_cuda` executable

**Test**:
```bash
./nbody_cuda 1000 5
```

### 3. Both CPU and GPU
```bash
make both
```

**Clean build**:
```bash
make clean
make both
```

## Running Benchmarks

### Quick Single Runs

**CPU**:
```bash
# Default threads (system max)
./nbody_naive 6000 10

# Specific thread count
OMP_NUM_THREADS=8 ./nbody_naive 6000 10
```

**GPU**:
```bash
# Both kernels (default)
./nbody_cuda 6000 10

# Specific kernel
./nbody_cuda 6000 10 --mode=tiled
./nbody_cuda 6000 10 --mode=naive
```

### Comprehensive Benchmarks

**CPU Only** (produces results.txt):
```bash
./benchmark_rigorous.sh results_cpu.txt
python3 analyze_benchmark.py results_cpu.txt
```

**GPU Only** (produces results_cuda_<timestamp>/ directory):
```bash
./benchmark_cuda.sh
# Auto-generates directory name with timestamp
```

**Both CPU + GPU + Plots** (recommended):
```bash
./run_benchmark_all.sh --both
```

This runs:
1. `make both` (unless --skip-build)
2. CPU benchmarks → `results_<timestamp>/cpu_results.txt`
3. GPU benchmarks → `results_<timestamp>/cuda/*.log`
4. Parsing → CSV files in `results_<timestamp>/`
5. Plotting → PNG/SVG in `results_<timestamp>/figures/`

**Options**:
```bash
# CPU benchmarks only
./run_benchmark_all.sh --cpu

# GPU benchmarks only
./run_benchmark_all.sh --gpu

# Skip compilation (binaries already built)
./run_benchmark_all.sh --both --skip-build
```

## Analysis and Visualization

### 1. Parse Benchmarks to CSV

**CPU**:
```bash
python3 analyze_benchmark.py results_cpu.txt
# Generates: strong_scaling.csv, problem_size_scaling.csv, timestep_scaling.csv
```

**GPU**:
```bash
python3 analyze_benchmark_extended.py results_cuda_*/problem_size_scaling.log --cuda
python3 analyze_benchmark_extended.py results_cuda_*/timestep_scaling.log --cuda
python3 analyze_benchmark_extended.py results_cuda_*/kernel_comparison.log --cuda
# Generates: cuda_problem_size.csv, cuda_timestep_scaling.csv, cuda_kernel_comparison.csv
```

### 2. Generate Plots

```bash
python3 plot_results.py --output-dir=figures
```

**Requirements**:
```bash
pip3 install matplotlib
# Optional but recommended:
pip3 install pandas
```

**Output**:
- `cpu_strong_scaling.png` - Speedup and efficiency vs threads
- `cpu_problem_size_scaling.png` - Time and GFlops vs N
- `cuda_problem_size_scaling.png` - GPU kernel comparison
- `cuda_transfer_overhead.png` - Kernel vs total time breakdown
- `cpu_vs_gpu_comparison.png` - Speedup analysis

All plots also saved as `.svg` for publication quality.

## Workflow Examples

### Scenario 1: Quick CPU Performance Check
```bash
make
OMP_NUM_THREADS=1 ./nbody_naive 5000 10  # Sequential
OMP_NUM_THREADS=16 ./nbody_naive 5000 10 # Parallel
```

### Scenario 2: Quick GPU Performance Check
```bash
make cuda
./nbody_cuda 5000 10
```

### Scenario 3: Full Scientific Study

**Step 1**: Build everything
```bash
make both
```

**Step 2**: Run comprehensive benchmarks
```bash
./run_benchmark_all.sh --both
```

**Step 3**: Review results
```bash
cd results_<timestamp>
ls *.csv              # Raw data
ls figures/*.png      # Visualizations
cat cuda/metadata.json  # System info
```

**Step 4**: Custom analysis
```python
import pandas as pd
cpu = pd.read_csv('strong_scaling.csv')
gpu = pd.read_csv('cuda_problem_size.csv')
# Your analysis here
```

### Scenario 4: Optimize and Compare

**Baseline**:
```bash
./nbody_cuda 10000 5 --mode=tiled
# Note the kernel time
```

**Modify** `cuda/nbody_cuda.cu` (e.g., change TILE_SIZE)

**Rebuild and test**:
```bash
make cuda
./nbody_cuda 10000 5 --mode=tiled
# Compare kernel time
```

### Scenario 5: Publication-Ready Figures

```bash
# Run full pipeline
./run_benchmark_all.sh --both

# Navigate to results
cd results_*/figures/

# Figures are publication-ready:
# - High resolution (300 DPI)
# - Vector format (.svg) available
# - Labeled axes, legends, grid
```

## Configuration

### CPU Thread Control
```bash
export OMP_NUM_THREADS=16
./nbody_naive 6000 10
```

### GPU Architecture
Edit `Makefile`:
```makefile
NVCC_ARCH ?= -gencode=arch=compute_86,code=sm_86
```

Or override:
```bash
NVCC_ARCH="-gencode=arch=compute_89,code=sm_89" make cuda
```

### Benchmark Problem Sizes

Edit `benchmark_cuda.sh`:
```bash
SIZE_SWEEP_N=(1000 2000 4000 6000 8000 10000 12000 16000)
```

Reduce for faster runs or memory constraints.

### Plot Styling

Edit `plot_results.py`:
```python
plt.style.use('seaborn-v0_8-darkgrid')  # Change style
COLORS = ['#1f77b4', '#ff7f0e', ...]     # Change colors
```

## Troubleshooting

### Build Issues

**Problem**: `nvcc: command not found`
**Solution**: Install CUDA Toolkit or ensure it's in PATH
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem**: Architecture mismatch errors
**Solution**: Set correct compute capability (see Building section)

**Problem**: `lm not found`
**Solution**: Install build essentials
```bash
sudo apt install build-essential  # Debian/Ubuntu
sudo dnf install gcc-c++          # Fedora
```

### Runtime Issues

**Problem**: GPU out of memory
**Solution**: Reduce problem sizes in benchmark scripts

**Problem**: Slow GPU performance
**Solution**: Check FP64 capability - consumer GPUs have limited double precision

**Problem**: Incorrect results
**Solution**: Run with `--verify` flag:
```bash
./nbody_cuda 1000 5 --verify
```

### Analysis Issues

**Problem**: `ImportError: No module named 'matplotlib'`
**Solution**:
```bash
pip3 install matplotlib
```

**Problem**: Empty CSV files
**Solution**: Check that logs match expected format. View sample:
```bash
head -50 results_cuda_*/problem_size_scaling.log
```

## Performance Expectations

### CPU (Intel i9-13900HX, 32 threads)

| Problem Size | Sequential | OpenMP | Speedup |
|--------------|------------|--------|---------|
| N=1000 | 0.037s | 0.008s | 4.7× |
| N=6000 | 1.81s | 0.18s | 10.1× |
| N=12000 | 3.6s | 0.32s | 11.3× |

### GPU (RTX 4070)

| Problem Size | CPU OpenMP | GPU Tiled | Speedup |
|--------------|------------|-----------|---------|
| N=1000 | 0.008s | 0.003s | 2.7× |
| N=6000 | 0.18s | 0.015s | 12× |
| N=12000 | 0.32s | 0.025s | 13× |

**Note**: Consumer GPUs have 1:32 FP64:FP32 performance ratio. Datacenter GPUs (A100, H100) will show much higher speedups.

## Next Steps

- **Learn methodology**: Read [BENCHMARKING.md](BENCHMARKING.md) and [GPU_BENCHMARKING.md](GPU_BENCHMARKING.md)
- **Quick GPU start**: See [QUICKSTART_GPU.md](QUICKSTART_GPU.md)
- **Customize**: Modify source code and re-benchmark
- **Publish**: Use SVG figures and cite the code

## Support

For issues:
1. Check error messages in terminal output
2. Verify system requirements (`nvidia-smi`, `nvcc --version`)
3. Review documentation files
4. Check `CUDA_CHECK()` macro outputs for GPU errors
