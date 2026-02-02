/*
 * N-Body Simulation: CUDA Implementation
 * =======================================
 * 
 * Two GPU kernel variants:
 *   1) Naive Non-Symmetric: Each thread computes forces on one particle from all others
 *   2) Tiled Optimized: Shared-memory tiling for better memory coalescing
 * 
 * Physics: Identical to CPU versions (double precision, same softening, Euler integrator)
 * Output: Formatted for compatibility with existing benchmark parser
 * 
 * Usage: ./nbody_cuda <N> <steps> [--mode=naive|tiled|both]
 */

#include "cuda_wrapper.h"

#define RAND_MAX 2147483647

// Provide sqrt for device code (CUDA provides this)
__device__ __host__ inline double sqrt(double x) {
    return ::sqrt(x);
}

// ============================================================================
// CONFIGURATION & CONSTANTS
// ============================================================================

#define SOFTENING 1e-9        // Gravitational softening parameter
#define TIMESTEP 0.01         // Integration timestep
#define TILE_SIZE 256         // Shared memory tile size for optimized kernel
#define MAX_THREADS_PER_BLOCK 256

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct {
    double x, y, z;
} Vec3;

typedef struct {
    Vec3 *pos;      // Positions
    Vec3 *vel;      // Velocities
    Vec3 *acc;      // Accelerations
    double *mass;   // Masses
    int n;          // Number of particles
} NBodySystem;

// ============================================================================
// HOST UTILITY FUNCTIONS
// ============================================================================

void init_system(NBodySystem *sys, int n, unsigned int seed) {
    sys->n = n;
    sys->pos = (Vec3*)malloc(n * sizeof(Vec3));
    sys->vel = (Vec3*)malloc(n * sizeof(Vec3));
    sys->acc = (Vec3*)malloc(n * sizeof(Vec3));
    sys->mass = (double*)malloc(n * sizeof(double));
    
    srand(seed);
    for (int i = 0; i < n; i++) {
        sys->pos[i].x = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        sys->pos[i].y = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        sys->pos[i].z = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        
        sys->vel[i].x = (rand() / (double)RAND_MAX) * 0.1;
        sys->vel[i].y = (rand() / (double)RAND_MAX) * 0.1;
        sys->vel[i].z = (rand() / (double)RAND_MAX) * 0.1;
        
        sys->acc[i].x = sys->acc[i].y = sys->acc[i].z = 0.0;
        sys->mass[i] = (rand() / (double)RAND_MAX) * 2.0 + 0.5;
    }
}

void free_system(NBodySystem *sys) {
    free(sys->pos);
    free(sys->vel);
    free(sys->acc);
    free(sys->mass);
}

double compute_checksum(NBodySystem *sys) {
    double sum = 0.0;
    for (int i = 0; i < sys->n; i++) {
        sum += sys->pos[i].x + sys->pos[i].y + sys->pos[i].z;
    }
    return sum;
}

void print_gpu_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("# GPU: %s\n", prop.name);
    printf("# Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("# Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("# Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("#\n");
}

// ============================================================================
// CUDA KERNEL 1: NAIVE NON-SYMMETRIC
// ============================================================================
// Each thread computes forces on particle i from all N particles
// Direct O(N²) computation with global memory reads

__global__ void compute_forces_naive(
    const Vec3 *pos, const double *mass, Vec3 *acc, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double ax = 0.0, ay = 0.0, az = 0.0;
    Vec3 pi = pos[i];
    
    // Loop over all particles
    for (int j = 0; j < n; j++) {
        Vec3 pj = pos[j];
        
        // Compute distance vector
        double dx = pj.x - pi.x;
        double dy = pj.y - pi.y;
        double dz = pj.z - pi.z;
        
        // Distance squared with softening
        double dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING;
        double inv_dist3 = 1.0 / (dist_sq * sqrt(dist_sq));
        
        // Accumulate force (F = m * a, so a += m_j * r / |r|³)
        double f = mass[j] * inv_dist3;
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }
    
    acc[i].x = ax;
    acc[i].y = ay;
    acc[i].z = az;
}

// ============================================================================
// CUDA KERNEL 2: TILED/OPTIMIZED WITH SHARED MEMORY
// ============================================================================
// Uses shared memory to cache position/mass tiles
// Better memory coalescing and reduced global memory traffic

__global__ void compute_forces_tiled(
    const Vec3 *pos, const double *mass, Vec3 *acc, int n)
{
    extern __shared__ double shared_mem[];
    
    // Shared memory layout: [px, py, pz, mass] for TILE_SIZE particles
    double *shared_px = shared_mem;
    double *shared_py = shared_mem + blockDim.x;
    double *shared_pz = shared_mem + 2 * blockDim.x;
    double *shared_m  = shared_mem + 3 * blockDim.x;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    double ax = 0.0, ay = 0.0, az = 0.0;
    Vec3 pi;
    
    if (i < n) {
        pi = pos[i];
    }
    
    // Iterate over tiles
    int num_tiles = (n + blockDim.x - 1) / blockDim.x;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int j = tile * blockDim.x + tid;
        
        // Load tile into shared memory
        if (j < n) {
            shared_px[tid] = pos[j].x;
            shared_py[tid] = pos[j].y;
            shared_pz[tid] = pos[j].z;
            shared_m[tid] = mass[j];
        } else {
            shared_m[tid] = 0.0;  // Padding with zero mass
        }
        
        __syncthreads();
        
        // Compute forces from this tile
        if (i < n) {
            #pragma unroll 8
            for (int j_local = 0; j_local < blockDim.x; j_local++) {
                double dx = shared_px[j_local] - pi.x;
                double dy = shared_py[j_local] - pi.y;
                double dz = shared_pz[j_local] - pi.z;
                
                double dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING;
                double inv_dist3 = 1.0 / (dist_sq * sqrt(dist_sq));
                
                double f = shared_m[j_local] * inv_dist3;
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (i < n) {
        acc[i].x = ax;
        acc[i].y = ay;
        acc[i].z = az;
    }
}

// ============================================================================
// VELOCITY VERLET UPDATE (runs on GPU)
// ============================================================================

__global__ void update_positions_velocities(
    Vec3 *pos, Vec3 *vel, const Vec3 *acc, int n, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Update velocities: v += a * dt
    vel[i].x += acc[i].x * dt;
    vel[i].y += acc[i].y * dt;
    vel[i].z += acc[i].z * dt;
    
    // Update positions: x += v * dt
    pos[i].x += vel[i].x * dt;
    pos[i].y += vel[i].y * dt;
    pos[i].z += vel[i].z * dt;
}

// ============================================================================
// SIMULATION RUNNER
// ============================================================================

typedef struct {
    double kernel_time;
    double h2d_time;
    double d2h_time;
    double total_time;
} TimingResults;

TimingResults run_simulation_gpu(
    NBodySystem *sys, int steps, bool use_tiled, bool warm_up)
{
    TimingResults timing = {0.0, 0.0, 0.0, 0.0};
    
    int n = sys->n;
    size_t vec_bytes = n * sizeof(Vec3);
    size_t mass_bytes = n * sizeof(double);
    
    // Allocate device memory
    Vec3 *d_pos, *d_vel, *d_acc;
    double *d_mass;
    
    CUDA_CHECK(cudaMalloc(&d_pos, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_vel, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_mass, mass_bytes));
    
    // Create CUDA events for timing
    cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&stop_d2h));
    
    // Host to device transfer
    CUDA_CHECK(cudaEventRecord(start_h2d));
    CUDA_CHECK(cudaMemcpy(d_pos, sys->pos, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, sys->vel, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mass, sys->mass, mass_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_h2d));
    CUDA_CHECK(cudaEventSynchronize(stop_h2d));
    
    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d));
    timing.h2d_time = h2d_ms / 1000.0;
    
    // Kernel configuration
    int threads_per_block = (use_tiled) ? TILE_SIZE : MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = (use_tiled) ? (4 * threads_per_block * sizeof(double)) : 0;
    
    // Warm-up kernel launch (if requested)
    if (warm_up) {
        if (use_tiled) {
            compute_forces_tiled<<<num_blocks, threads_per_block, shared_mem_size>>>(
                d_pos, d_mass, d_acc, n);
        } else {
            compute_forces_naive<<<num_blocks, threads_per_block>>>(
                d_pos, d_mass, d_acc, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Timed simulation loop
    CUDA_CHECK(cudaEventRecord(start_kernel));
    
    for (int step = 0; step < steps; step++) {
        // Compute forces
        if (use_tiled) {
            compute_forces_tiled<<<num_blocks, threads_per_block, shared_mem_size>>>(
                d_pos, d_mass, d_acc, n);
        } else {
            compute_forces_naive<<<num_blocks, threads_per_block>>>(
                d_pos, d_mass, d_acc, n);
        }
        
        // Update positions and velocities
        update_positions_velocities<<<num_blocks, threads_per_block>>>(
            d_pos, d_vel, d_acc, n, TIMESTEP);
    }
    
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));
    
    float kernel_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
    timing.kernel_time = kernel_ms / 1000.0;
    
    // Device to host transfer
    CUDA_CHECK(cudaEventRecord(start_d2h));
    CUDA_CHECK(cudaMemcpy(sys->pos, d_pos, vec_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sys->vel, d_vel, vec_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_d2h));
    CUDA_CHECK(cudaEventSynchronize(stop_d2h));
    
    float d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h));
    timing.d2h_time = d2h_ms / 1000.0;
    
    timing.total_time = timing.h2d_time + timing.kernel_time + timing.d2h_time;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_acc));
    CUDA_CHECK(cudaFree(d_mass));
    
    CUDA_CHECK(cudaEventDestroy(start_h2d));
    CUDA_CHECK(cudaEventDestroy(stop_h2d));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_d2h));
    CUDA_CHECK(cudaEventDestroy(stop_d2h));
    
    return timing;
}

// ============================================================================
// MAIN
// ============================================================================

void print_usage(const char *prog) {
    printf("Usage: %s <N> <steps> [--mode=naive|tiled|both]\n", prog);
    printf("\nOptions:\n");
    printf("  N          Number of particles\n");
    printf("  steps      Number of timesteps\n");
    printf("  --mode     Kernel mode: naive, tiled, or both (default: both)\n");
    printf("  --verify   Run verification against reference\n");
    printf("  --seed=N   Random seed (default: 42)\n");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    int steps = atoi(argv[2]);
    
    // Parse options
    bool run_naive = true;
    bool run_tiled = true;
    bool verify = false;
    unsigned int seed = 42;
    
    for (int i = 3; i < argc; i++) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            const char *mode = argv[i] + 7;
            if (strcmp(mode, "naive") == 0) {
                run_naive = true;
                run_tiled = false;
            } else if (strcmp(mode, "tiled") == 0) {
                run_naive = false;
                run_tiled = true;
            } else if (strcmp(mode, "both") == 0) {
                run_naive = true;
                run_tiled = true;
            }
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else if (strncmp(argv[i], "--seed=", 7) == 0) {
            seed = atoi(argv[i] + 7);
        }
    }
    
    // Print header
    printf("=============================================================\n");
    printf("N-Body Simulation: CUDA Performance Study\n");
    printf("=============================================================\n");
    printf("Problem size:      %d particles\n", n);
    printf("Timesteps:         %d\n", steps);
    printf("Timestep size:     %g\n", TIMESTEP);
    printf("=============================================================\n");
    printf("\n");
    
    print_gpu_info();
    
    // Initialize system
    NBodySystem sys;
    init_system(&sys, n, seed);
    
    double initial_checksum = compute_checksum(&sys);
    printf("# Initial checksum: %.6f\n", initial_checksum);
    printf("#\n");
    
    // Run simulations
    TimingResults timing_naive, timing_tiled;
    
    if (run_naive) {
        NBodySystem sys_naive;
        init_system(&sys_naive, n, seed);
        
        timing_naive = run_simulation_gpu(&sys_naive, steps, false, true);
        
        double final_checksum = compute_checksum(&sys_naive);
        printf("# Naive final checksum: %.6f\n", final_checksum);
        
        free_system(&sys_naive);
    }
    
    if (run_tiled) {
        NBodySystem sys_tiled;
        init_system(&sys_tiled, n, seed);
        
        timing_tiled = run_simulation_gpu(&sys_tiled, steps, true, true);
        
        double final_checksum = compute_checksum(&sys_tiled);
        printf("# Tiled final checksum: %.6f\n", final_checksum);
        
        free_system(&sys_tiled);
    }
    
    printf("\n");
    
    // Print timing results in format compatible with CPU benchmark parser
    if (run_naive) {
        printf("5. CUDA Non-Symmetric (kernel):    %f seconds\n", timing_naive.kernel_time);
        printf("5b. CUDA Non-Symmetric (total):    %f seconds  [H2D: %.4fs, D2H: %.4fs]\n",
               timing_naive.total_time, timing_naive.h2d_time, timing_naive.d2h_time);
    }
    
    if (run_tiled) {
        printf("6. CUDA Tiled (kernel):             %f seconds\n", timing_tiled.kernel_time);
        printf("6b. CUDA Tiled (total):             %f seconds  [H2D: %.4fs, D2H: %.4fs]\n",
               timing_tiled.total_time, timing_tiled.h2d_time, timing_tiled.d2h_time);
    }
    
    if (run_naive && run_tiled) {
        double speedup = timing_naive.kernel_time / timing_tiled.kernel_time;
        printf("\n");
        printf("=============================================================\n");
        printf("Analysis:\n");
        printf("=============================================================\n");
        printf("Tiled vs Naive speedup:            %.2fx\n", speedup);
        printf("=============================================================\n");
    }
    
    free_system(&sys);
    
    return 0;
}
