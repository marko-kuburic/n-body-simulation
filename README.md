# N-Body Gravitational Simulation: CPU + GPU Comparative Performance Study

A comprehensive implementation of the direct N-body gravitational simulation algorithm in C (CPU) and CUDA (GPU), providing multiple variants to demonstrate performance trade-offs across different architectures, algorithmic optimizations, and parallelization strategies.

## Overview

This implementation suite is designed for educational and research purposes, particularly suitable for HPC systems papers and parallel programming courses. It demonstrates fundamental concepts in:

- **Algorithmic optimization** (symmetric vs. non-symmetric formulations)
- **CPU parallel programming** (OpenMP parallelization strategies)
- **GPU acceleration** (CUDA kernel optimization with shared memory)
- **Performance analysis** (work reduction vs. synchronization overhead)
- **Cache-coherence effects** (atomic operations and false sharing)
- **Heterogeneous computing** (CPU vs GPU performance characterization)

## Implementations

### CPU Variants (4 versions)

#### 1️⃣ Sequential - Non-Symmetric (Baseline)

**The naïve all-pairs approach:**
- Computes forces for all pairs (i, j) where j ≠ i
- Each interaction computed twice
- Computational complexity: **O(N²)** force evaluations
- Memory pattern: Sequential reads, local accumulation (cache-friendly)
- **Purpose:** Baseline for correctness validation and performance comparison

#### 2️⃣ Sequential - Symmetric (Newton's Third Law)

**Exploits Newton's third law (F_ij = -F_ji):**
- Computes forces only for pairs where j > i
- Applies equal and opposite forces simultaneously
- Computational complexity: **O(N(N-1)/2)** ≈ 50% arithmetic reduction
- Memory pattern: Non-local writes (less cache-friendly)
- **Key insight:** In sequential execution, arithmetic reduction typically dominates memory overhead

#### 3️⃣ OpenMP - Non-Symmetric (Ideal Parallelization)

**Embarrassingly parallel implementation:**
- Parallelizes outer loop with `#pragma omp parallel for`
- **No synchronization required** (each thread updates distinct particles)
- **No atomic operations** needed
- Expected scaling: Near-linear speedup (memory bandwidth limited)
- **Purpose:** Demonstrates "gold standard" parallel implementation with minimal overhead

#### 4️⃣ OpenMP - Symmetric (Synchronization Overhead)

**Demonstrates the cost of synchronization:**
- Parallelizes symmetric formulation using atomic operations
- **Requires `#pragma omp atomic`** for thread-safe force accumulation
- Trade-off: Reduced arithmetic vs. synchronization overhead
- Performance characteristics:
  - Small thread counts: May outperform non-symmetric (50% less work)
  - Large thread counts: Typically underperforms due to contention
  - Cache-line ping-pong effects become dominant
- **Purpose:** Demonstrates that algorithmic efficiency ≠ parallel efficiency

### GPU Variants (2 CUDA kernels)

#### 5️⃣ CUDA Non-Symmetric (Naive Global Memory)

**Direct GPU port of CPU algorithm:**
- One CUDA thread per particle
- Global memory reads for all positions
- No shared memory optimization
- **Characteristics:**
  - Simple implementation
  - Memory bandwidth-bound
  - Poor memory coalescing
- **Purpose:** GPU baseline, demonstrates need for optimization

#### 6️⃣ CUDA Tiled (Shared Memory Optimized)

**Optimized kernel with tiling:**
- Cooperative tile loading into shared memory
- Block-level synchronization (`__syncthreads()`)
- Better memory coalescing (16-byte aligned)
- **Characteristics:**
  - 2-4× faster than naive kernel
  - Compute-bound on modern GPUs
  - Reduced global memory traffic
- **Purpose:** Demonstrates GPU optimization techniques

## Key Educational Insights

### Why Symmetric OpenMP Uses Atomics

When multiple threads process different particle pairs simultaneously, they may need to update the same particle's forces concurrently:

```
Thread A: Processing pair (0, 5) → updates particle 5
Thread B: Processing pair (3, 5) → updates particle 5  [RACE CONDITION!]
```

Without atomics, this creates **race conditions** leading to incorrect results. Atomic operations ensure thread safety but introduce:
- Memory serialization points
- Cache-coherence protocol overhead (MESI/MOESI state transitions)
- False sharing effects (adjacent memory locations in same cache line)

### CPU vs GPU Architecture Trade-offs

**CPU Strengths:**
- High single-thread performance
- Large caches (L3: 32-64 MB)
- Sophisticated branch prediction
- Low latency memory access

**GPU Strengths:**
- Massive parallelism (thousands of cores)
- High memory bandwidth (500+ GB/s)
- Efficient for regular, data-parallel workloads
- Cost-effective performance/watt

**N-body Implications:**
- Small N (< 2000): CPU competitive due to low overhead
- Large N (> 5000): GPU dominates (bandwidth advantage)
- Transfer overhead: Amortized over multiple timesteps

### Performance Analysis

**Expected Results:**

| Version | Arithmetic Work | Synchronization | Typical Behavior |
|---------|----------------|-----------------|------------------|
| 1. Seq Non-Sym | 100% | None | Baseline |
| 2. Seq Symmetric | ~50% | None | **~1.5-2x faster** |
| 3. OMP Non-Sym | 100% | None | **Near-linear speedup** |
| 4. OMP Symmetric | ~50% | Heavy (atomics) | Often **slowest** |
| 5. CUDA Naive | 100% | None | **10-30× vs CPU seq** |
| 6. CUDA Tiled | 100% | Block-level | **100-200× vs CPU seq** |

**Crossover Analysis:**
The symmetric OpenMP version's performance depends on:
- **Problem size:** Larger N increases arithmetic dominance
- **Thread count:** More threads → more contention
- **Cache architecture:** L3 size, coherence protocol latency
- **Memory bandwidth:** Can become bottleneck for all versions

GPU performance depends on:
- **Problem size:** Larger N increases GPU utilization
- **Transfer amortization:** Multiple timesteps reduce overhead fraction
- **FP64 capability:** Consumer GPUs have 1:32 FP64:FP32 ratio

## Compilation

### CPU Only
```bash
make
```

### CUDA (GPU)
```bash
make cuda
# Or specify architecture:
NVCC_ARCH="-gencode=arch=compute_86,code=sm_86" make cuda
```

### Both
```bash
make both
```

Or manually:
```bash
# CPU
gcc -O3 -Wall -Wextra -std=c99 -fopenmp -lm -o nbody_naive nbody_naive.c

# CUDA
nvcc -O3 --std=c++11 -gencode=arch=compute_80,code=sm_80 -o nbody_cuda cuda/nbody_cuda.cu
```

**Compiler flags explained:**
- `-O3`: Aggressive optimization (vectorization, loop unrolling)
- `-fopenmp`: Enable OpenMP support (CPU)
- `-lm`: Link math library (for `sqrt()`)
- `-std=c99`: Use C99 standard (CPU)
- `--std=c++11`: C++11 for CUDA
- `-gencode`: Specify GPU compute capability (adjust for your GPU)

## Usage

### CPU Binary
```bash
./nbody_naive [n_particles] [n_steps] [dt]
```

**Parameters:**
- `n_particles`: Number of particles (default: 2000)
- `n_steps`: Number of simulation timesteps (default: 10)
- `dt`: Timestep size (default: 0.01)

**Examples:**
```bash
# Default parameters (2000 particles, 10 steps)
./nbody_naive

# Small problem for quick testing
./nbody_naive 500 5

# Large problem for performance analysis
./nbody_naive 5000 20

# Custom timestep
./nbody_naive 1000 10 0.005
```

### CUDA Binary
```bash
./nbody_cuda <n_particles> <n_steps> [--mode=naive|tiled|both]
```

**Parameters:**
- `n_particles`: Number of particles (required)
- `n_steps`: Number of timesteps (required)
- `--mode`: Kernel selection (default: both)
- `--verify`: Run verification (optional)
- `--seed=N`: Random seed (default: 42)

**Examples:**
```bash
# Run both kernels (default)
./nbody_cuda 6000 10

# Test only tiled kernel
./nbody_cuda 10000 5 --mode=tiled

# Verify correctness
./nbody_cuda 1000 5 --verify

# Custom seed for reproducibility
./nbody_cuda 5000 10 --seed=12345
```

## OpenMP Configuration

Control thread count with `OMP_NUM_THREADS`:

```bash
# Use 4 threads
export OMP_NUM_THREADS=4
./nbody_naive 2000 10

# Or inline
OMP_NUM_THREADS=8 ./nbody_naive 2000 10
```

## Example Output

```
=============================================================
N-Body Simulation: Comparative Performance Study
=============================================================
Problem size:      2000 particles
Timesteps:         10
Timestep size:     0.01
OpenMP threads:    4
=============================================================

1. Sequential Non-Symmetric:   0.258450 seconds  [Baseline]
2. Sequential Symmetric:       0.140722 seconds  [Speedup: 1.84x]
3. OpenMP Non-Symmetric:       0.060772 seconds  [Speedup: 4.25x]
4. OpenMP Symmetric:           2.493101 seconds  [Speedup: 0.10x]

=============================================================
Analysis:
=============================================================
Symmetric arithmetic reduction:  45.6%
OpenMP non-sym parallel eff:     106.3%
OpenMP symmetric vs non-sym:     0.02x (slower)
=============================================================
```

**Interpretation:**
- Sequential symmetric achieves 84% speedup from work reduction and cache benefits
- OpenMP non-symmetric shows 106% apparent efficiency (superlinear due to cache effects)
- OpenMP symmetric is **41x slower** than non-symmetric due to massive atomic contention
- This demonstrates: **minimizing synchronization is critical for parallel performance**

## Automated Benchmarking

### CPU Benchmarks
Run comprehensive CPU performance analysis:
```bash
./benchmark_rigorous.sh results_cpu.txt
python3 analyze_benchmark.py results_cpu.txt
```

Generates CSV files:
- `strong_scaling.csv` - Thread scaling analysis
- `problem_size_scaling.csv` - N² complexity verification  
- `timestep_scaling.csv` - Temporal linearity check

See [BENCHMARKING.md](BENCHMARKING.md) for methodology details.

### GPU Benchmarks
Run CUDA performance analysis:
```bash
./benchmark_cuda.sh results_cuda_<timestamp>
python3 analyze_benchmark_extended.py results_cuda_*/*.log --cuda
```

Generates CSV files:
- `cuda_problem_size.csv` - GPU scaling and kernel comparison
- `cuda_timestep_scaling.csv` - Temporal scaling
- `cuda_kernel_comparison.csv` - Naive vs tiled analysis

See [GPU_BENCHMARKING.md](GPU_BENCHMARKING.md) for GPU-specific methodology.

### Unified Pipeline (CPU + GPU + Plots)
Run everything and generate publication-ready figures:
```bash
./run_benchmark_all.sh --both
```

This will:
1. Build CPU and CUDA binaries
2. Run comprehensive benchmarks for both
3. Parse all logs into CSV files
4. Generate matplotlib plots in `results_<timestamp>/figures/`

Output figures:
- `cpu_strong_scaling.png` - Thread speedup and efficiency
- `cpu_problem_size_scaling.png` - Time and GFlops vs N
- `cuda_problem_size_scaling.png` - GPU kernel analysis
- `cuda_transfer_overhead.png` - Kernel vs total time
- `cpu_vs_gpu_comparison.png` - Heterogeneous performance

### Plotting Only
If you already have CSV files:
```bash
python3 plot_results.py --output-dir=figures
```

Requires: `pip3 install matplotlib`

## Benchmark Results (CPU)

The following results were obtained on a 32-core CPU system using the automated benchmark suite.

### Strong Scaling Analysis (2000 particles, 10 timesteps)

| Threads | Seq Non-Sym | Seq Symmetric | OMP Non-Sym | OMP Symmetric | Parallel Eff (Non-Sym) |
|---------|-------------|---------------|-------------|---------------|------------------------|
| 1       | 0.218s      | 0.142s (1.53x)| 0.200s      | 0.585s (0.37x)| 108.7%                |
| 2       | 0.200s      | 0.140s (1.42x)| 0.108s      | 1.269s (0.16x)| 92.7%                 |
| 4       | 0.258s      | 0.141s (1.84x)| 0.061s      | 2.493s (0.10x)| 106.3%                |
| 8       | 0.220s      | 0.141s (1.56x)| 0.072s      | 1.719s (0.13x)| 38.3%                 |
| 16      | 0.200s      | 0.141s (1.42x)| 0.046s      | 0.548s (0.37x)| 27.3%                 |
| 32      | 0.204s      | 0.150s (1.36x)| 0.036s      | 0.293s (0.69x)| 17.6%                 |

**Key Observations:**
- Sequential symmetric provides consistent **1.36-1.84x speedup** across all thread counts
- OpenMP non-symmetric shows excellent scaling: **4.25x** at 4 threads, **5.62x** at 32 threads
- Parallel efficiency stays above 90% up to 4 threads, then drops due to memory bandwidth saturation
- OpenMP symmetric is **10-50x slower** than non-symmetric at 2-8 threads
- At 32 threads, atomic contention still dominates: **8.1x slower** than non-symmetric

### Problem Size Scaling (32 threads, 5 timesteps)

| Particles | Seq Non-Sym | Seq Symmetric | OMP Non-Sym | OMP Symmetric | Speedup (Non-Sym) | Par. Eff |
|-----------|-------------|---------------|-------------|---------------|-------------------|----------|
| 500       | 0.0070s     | 0.0044s (1.57x)| 0.0037s    | 0.0188s (0.37x)| 1.87x            | 5.8%    |
| 1,000     | 0.0276s     | 0.0236s (1.17x)| 0.0088s    | 0.0601s (0.46x)| 3.13x            | 9.8%    |
| 2,000     | 0.1214s     | 0.0708s (1.71x)| 0.0168s    | 0.1453s (0.84x)| 7.23x            | 22.6%   |
| 4,000     | 0.4026s     | 0.2811s (1.43x)| 0.0454s    | 0.4324s (0.93x)| 8.86x            | 27.7%   |
| 8,000     | 1.6279s     | 1.1228s (1.45x)| 0.1635s    | 1.3318s (1.22x)| 9.96x            | 31.1%   |
| 16,000    | 6.3666s     | 4.6001s (1.38x)| 0.5723s    | 4.3279s (1.47x)| 11.12x           | 34.8%   |
| 32,000    | 25.6072s    | 18.1719s (1.41x)| 2.1059s    | 14.0512s (1.82x)| 12.16x           | 38.0%   |

**Key Observations:**
- Sequential symmetric maintains **1.17-1.71x speedup** across all problem sizes
- OpenMP non-symmetric shows improving speedup with problem size: **1.87x** (N=500) → **12.16x** (N=32K)
- Parallel efficiency improves dramatically with N: **5.8%** (N=500) → **38.0%** (N=32K)
- Critical crossover: At N≥8000, OpenMP symmetric becomes faster than sequential
- At N=32000, symmetric OpenMP achieves **1.82x speedup** despite atomic overhead
- **Lesson:** Large problems favor symmetric formulation even with synchronization cost

### Timestep Invariance (4000 particles, 32 threads)

| Timesteps | Seq Non-Sym | Seq Symmetric | OMP Non-Sym | OMP Symmetric | Speedup (Non-Sym) |
|-----------|-------------|---------------|-------------|---------------|-------------------|
| 5         | 0.3989s     | 0.2820s (1.41x)| 0.0487s    | 0.4422s (0.90x)| 8.19x            |
| 10        | 0.8002s     | 0.5620s (1.42x)| 0.1084s    | 0.8694s (0.92x)| 7.38x            |
| 20        | 1.5993s     | 1.1238s (1.42x)| 0.1995s    | 1.7293s (0.92x)| 8.02x            |
| 40        | 3.1872s     | 2.2454s (1.42x)| 0.3859s    | 3.4202s (0.93x)| 8.26x            |
| 80        | 6.3702s     | 4.5097s (1.41x)| 0.7868s    | 6.8794s (0.93x)| 8.10x            |
| 160       | 12.7187s    | 9.1701s (1.39x)| 1.7773s    | 13.7135s (0.93x)| 7.16x            |
| 320       | 25.5739s    | 18.0370s (1.42x)| 2.7344s    | 27.415s (0.93x)| 9.35x            |

**Key Observations:**
- All speedups are **independent of timestep count** (expected: timesteps are perfectly sequential)
- Sequential symmetric consistently **1.39-1.42x faster** (arithmetic advantage stable)
- OpenMP non-symmetric achieves **7-9x speedup** across all timestep counts
- OpenMP symmetric remains **0.90-0.93x** (still losing despite large problem size)
- **Insight:** Synchronization overhead per timestep remains constant; total overhead scales with time

### Critical Analysis

**Why Sequential Symmetric Wins:**
- Arithmetic work reduction: 50% fewer force calculations
- No synchronization overhead in sequential code
- Consistent **1.4x advantage** regardless of problem size or timestep count
- **Result: Reliable, simple speedup with no parallelization complexity**

**Why OpenMP Non-Symmetric Dominates for Parallel:**
- Zero synchronization overhead (embarrassingly parallel)
- Each thread updates only its own particles (excellent cache locality)
- Scales from 1.87x (N=500) to 12.16x (N=32K)
- **Result: Best scalability, but requires sufficient work to amortize overhead**

**Why OpenMP Symmetric is Complex:**
- **Small problems (N<4000):** Atomic overhead dominates → 0.37-0.93x slower
- **Medium problems (N≈4000-8000):** Transition zone → 0.90-1.22x
- **Large problems (N>8000):** More work per atomic operation → 1.22-1.82x faster
- Cache-line ping-pong effects persist at all sizes but amortized better at large N
- **Key insight:** Synchronization cost is roughly constant, but amortizes better with increasing arithmetic work

### Performance Recommendations

| Problem Size | Recommended Version | Rationale | Expected Speedup |
|--------------|---------------------|-----------|------------------|
| N < 1,000    | Sequential Symmetric | Parallelization overhead > benefit (poor eff <10%) | 1.2-1.6x |
| 1K < N < 4K  | OpenMP Non-Symmetric (4-8 threads) | Overhead manageable, good efficiency | 3-8x |
| 4K < N < 8K  | OpenMP Non-Symmetric (8-16 threads) | Peak efficiency region | 8-10x |
| N > 8K       | OpenMP Symmetric OR Non-Sym (32 threads) | Both viable; symmetric wins at very large N | 9-12x |
| N > 100K     | Tree methods (Barnes-Hut, FMM) | O(N²) becomes prohibitive; use O(N log N) | 100-1000x |

**Important:** These results demonstrate a fundamental principle in HPC: **algorithmic work reduction does not guarantee better parallel performance**. The OpenMP symmetric version does 50% less arithmetic but is often dramatically slower due to synchronization overhead. Only at very large problem sizes does the work reduction outweigh atomic operation costs. This illustrates Amdahl's law and the critical importance of minimizing synchronization in parallel algorithm design.

## Performance Experiments

### Varying Thread Count

```bash
for threads in 1 2 4 8 16; do
    echo "Threads: $threads"
    OMP_NUM_THREADS=$threads ./nbody_naive 2000 10
done
```

### Varying Problem Size

```bash
for n in 500 1000 2000 4000; do
    echo "Problem size: $n"
    ./nbody_naive $n 5
done
```

### Strong Scaling Study

```bash
# Fixed problem size, varying threads
OMP_NUM_THREADS=1 ./nbody_naive 5000 10
OMP_NUM_THREADS=2 ./nbody_naive 5000 10
OMP_NUM_THREADS=4 ./nbody_naive 5000 10
OMP_NUM_THREADS=8 ./nbody_naive 5000 10
```

## Implementation Details

### Numerical Method
- **Time integration:** Forward Euler (first-order)
- **Gravitational constant:** G = 1.0 (arbitrary units)
- **Softening parameter:** ε = 10⁻⁹ (prevents singularities)
- **Force computation:** F = G·m_i·m_j / (r² + ε²)^(3/2)

### Memory Layout
- Structure-of-arrays (SoA) would be more cache-efficient
- Current implementation uses array-of-structures (AoS) for clarity

### Scheduling Strategy
- **Non-symmetric:** `schedule(static)` - equal work per iteration
- **Symmetric:** `schedule(dynamic)` - handles load imbalance (particle 0 has most work)

## Academic Use

This code is ideal for:
- **HPC courses:** Teaching parallel programming concepts
- **Systems papers:** Demonstrating synchronization overhead
- **Performance analysis:** Understanding Amdahl's law and strong scaling
- **Algorithm design:** Balancing arithmetic intensity vs. communication cost

### Topics Illustrated
✅ Algorithmic complexity analysis  
✅ Cache-coherence effects  
✅ OpenMP programming patterns  
✅ Atomic operations and race conditions  
✅ Strong scaling and parallel efficiency  
✅ Performance measurement and profiling  

## Limitations

- **Integration scheme:** Forward Euler is not energy-conserving (use Velocity Verlet for accuracy)
- **Complexity:** O(N²) becomes prohibitive beyond ~10K particles (consider Barnes-Hut or FMM)
- **Vectorization:** Code not optimized for SIMD (consider SoA layout and intrinsics)
- **Memory:** All-to-all access pattern limits bandwidth utilization

## Further Optimizations

Beyond this baseline implementation:
1. **Blocking/tiling** for better cache locality
2. **SIMD vectorization** with AVX-512 or similar
3. **Structure-of-Arrays (SoA)** memory layout
4. **GPU acceleration** with CUDA or OpenCL
5. **Tree-based methods** (Barnes-Hut, Fast Multipole Method)
6. **Hybrid MPI+OpenMP** for distributed memory systems

## License

Public domain / MIT License - free for academic and educational purposes.

## References

- Hockney, R. W., & Eastwood, J. W. (1988). *Computer Simulation Using Particles*
- Barnes, J., & Hut, P. (1986). A hierarchical O(N log N) force-calculation algorithm. *Nature*
- OpenMP Architecture Review Board (2018). *OpenMP Application Programming Interface*

---

**Note:** This implementation prioritizes clarity and educational value over raw performance. Production N-body codes typically use more sophisticated algorithms and optimizations.
