/*
 * ============================================================================
 * N-Body Gravitational Simulation: Comparative Study of Parallelization
 * Strategies and Algorithmic Formulations
 * ============================================================================
 *
 * This implementation provides four distinct versions of the direct N-body
 * algorithm, designed to demonstrate the performance trade-offs between
 * symmetric/non-symmetric formulations and sequential/parallel execution.
 *
 * ALGORITHMIC FORMULATIONS:
 * -------------------------
 *
 * 1. NON-SYMMETRIC (Naïve All-Pairs):
 *    Each particle i computes forces from all other particles j (j ≠ i).
 *    Computational complexity: O(N²) force evaluations
 *    Each interaction is computed twice (once for i→j, once for j→i)
 *    Memory access pattern: Each thread reads only its assigned particles
 *
 * 2. SYMMETRIC (Newton's Third Law):
 *    Forces are computed only for pairs where j > i.
 *    Newton's third law (F_ij = -F_ji) is exploited to update both particles.
 *    Computational complexity: O(N(N-1)/2) ≈ 50% reduction in arithmetic
 *    Memory access pattern: Requires concurrent writes to multiple particles
 *
 * PARALLELIZATION STRATEGIES:
 * ---------------------------
 *
 * 3. OpenMP NON-SYMMETRIC:
 *    Parallelizes outer loop over particles with no data dependencies.
 *    Each thread updates only its assigned particle i.
 *    No synchronization required → ideal scaling characteristics.
 *    Expected speedup: Near-linear with thread count (Amdahl's law permitting).
 *
 * 4. OpenMP SYMMETRIC:
 *    Parallelizes the symmetric formulation using atomic operations.
 *    Multiple threads may update the same particle j concurrently.
 *    Requires #pragma omp atomic for thread-safe force accumulation.
 *    Trade-off: Reduced arithmetic vs. synchronization overhead and
 *    cache-coherence traffic (false sharing, memory contention).
 *
 * PERFORMANCE ANALYSIS:
 * ---------------------
 * The symmetric OpenMP version demonstrates an important HPC principle:
 * algorithmic work reduction does not always translate to wall-clock speedup.
 * Atomic operations introduce serialization points and cache-line ping-pong
 * effects that can dominate the savings from reduced floating-point operations.
 * This effect becomes more pronounced with increasing thread count and problem
 * size, illustrating the critical importance of minimizing synchronization in
 * parallel algorithms.
 *
 * NUMERICAL METHOD:
 * -----------------
 * Time integration: Forward Euler (velocity Verlet would be more stable)
 * Gravitational constant: G = 1.0
 * Softening parameter: ε = 10⁻⁹ (prevents singularities at r → 0)
 * Force computation: F = G * m_i * m_j / (r² + ε²)^(3/2)
 *
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct {
    double x, y, z;      // Position components
    double vx, vy, vz;   // Velocity components
    double mass;         // Particle mass
} Particle;

static void init_particles(Particle *p, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; ++i) {
        p[i].x = (double)rand() / (double)RAND_MAX - 0.5;
        p[i].y = (double)rand() / (double)RAND_MAX - 0.5;
        p[i].z = (double)rand() / (double)RAND_MAX - 0.5;
        p[i].vx = 0.0;
        p[i].vy = 0.0;
        p[i].vz = 0.0;
        p[i].mass = (double)rand() / (double)RAND_MAX + 0.01;
    }
}

/*
 * ============================================================================
 * VERSION 1: Sequential - Non-Symmetric (Baseline)
 * ============================================================================
 * 
 * This is the naïve all-pairs implementation where each particle i computes
 * forces from all other particles j (j ≠ i). Each interaction is computed
 * twice: once when processing particle i, and again when processing particle j.
 * 
 * Computational work: N(N-1) force evaluations
 * Memory access: Sequential reads, local accumulation (cache-friendly)
 * Parallelization potential: Embarrassingly parallel (no data dependencies)
 * 
 * This version serves as the baseline for both correctness validation and
 * performance comparison.
 */
static void step_sequential_nonsymmetric(Particle *p, int n, double dt, double softening) {
    const double G = 1.0;

    // Phase 1: Force computation and velocity update
    for (int i = 0; i < n; ++i) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        const double xi = p[i].x;
        const double yi = p[i].y;
        const double zi = p[i].z;

        // Accumulate forces from all other particles
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            
            const double dx = p[j].x - xi;
            const double dy = p[j].y - yi;
            const double dz = p[j].z - zi;
            const double dist2 = dx * dx + dy * dy + dz * dz + softening * softening;
            const double invDist = 1.0 / sqrt(dist2);
            const double invDist3 = invDist * invDist * invDist;
            const double s = G * p[j].mass * invDist3;
            
            fx += dx * s;
            fy += dy * s;
            fz += dz * s;
        }

        // Update velocity using accumulated force
        p[i].vx += fx * dt;
        p[i].vy += fy * dt;
        p[i].vz += fz * dt;
    }

    // Phase 2: Position update
    for (int i = 0; i < n; ++i) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

/*
 * ============================================================================
 * VERSION 2: Sequential - Symmetric (Newton's Third Law)
 * ============================================================================
 * 
 * This version exploits Newton's third law (F_ij = -F_ji) by computing each
 * pairwise interaction only once. When computing the force between particles
 * i and j, we simultaneously update both particles with equal and opposite
 * forces.
 * 
 * Computational work: N(N-1)/2 force evaluations (~50% reduction)
 * Memory access: Non-local writes to force arrays (less cache-friendly)
 * Arithmetic intensity: Approximately doubled compared to non-symmetric
 * 
 * In sequential execution, this formulation provides a significant speedup
 * due to the reduced number of force evaluations, despite slightly worse
 * memory access patterns.
 */
static void step_sequential_symmetric(Particle *p, int n, double dt, double softening) {
    const double G = 1.0;

    // Allocate force accumulation arrays
    double *fx = (double *)calloc(n, sizeof(double));
    double *fy = (double *)calloc(n, sizeof(double));
    double *fz = (double *)calloc(n, sizeof(double));

    // Phase 1: Force computation using symmetric formulation
    // Only loop over pairs where j > i
    for (int i = 0; i < n - 1; ++i) {
        const double xi = p[i].x;
        const double yi = p[i].y;
        const double zi = p[i].z;
        const double mi = p[i].mass;

        for (int j = i + 1; j < n; ++j) {
            const double dx = p[j].x - xi;
            const double dy = p[j].y - yi;
            const double dz = p[j].z - zi;
            const double dist2 = dx * dx + dy * dy + dz * dz + softening * softening;
            const double invDist = 1.0 / sqrt(dist2);
            const double invDist3 = invDist * invDist * invDist;
            
            // Compute force magnitude components
            const double f = G * invDist3;
            const double fij_x = dx * f;
            const double fij_y = dy * f;
            const double fij_z = dz * f;
            
            // Apply Newton's third law: equal and opposite forces
            fx[i] += fij_x * p[j].mass;  // Force on i due to j
            fy[i] += fij_y * p[j].mass;
            fz[i] += fij_z * p[j].mass;
            
            fx[j] -= fij_x * mi;         // Force on j due to i (opposite)
            fy[j] -= fij_y * mi;
            fz[j] -= fij_z * mi;
        }
    }

    // Phase 2: Velocity update from accumulated forces
    for (int i = 0; i < n; ++i) {
        p[i].vx += fx[i] * dt;
        p[i].vy += fy[i] * dt;
        p[i].vz += fz[i] * dt;
    }

    // Phase 3: Position update from velocities
    for (int i = 0; i < n; ++i) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }

    free(fx);
    free(fy);
    free(fz);
}

/*
 * ============================================================================
 * VERSION 3: OpenMP - Non-Symmetric (Ideal Parallelization)
 * ============================================================================
 * 
 * This version parallelizes the non-symmetric formulation using OpenMP.
 * The outer loop over particles is distributed across threads, with each
 * thread computing forces for its assigned subset of particles.
 * 
 * Key characteristics:
 * - No data dependencies between threads (embarrassingly parallel)
 * - No synchronization required (each thread writes to distinct particles)
 * - No atomic operations needed
 * - Excellent cache behavior (each thread reads its local particle data)
 * - Expected scaling: Near-linear speedup (limited by memory bandwidth)
 * 
 * This represents the "gold standard" for parallel N-body implementations
 * in terms of avoiding synchronization overhead. The primary limitation is
 * the doubled arithmetic work compared to the symmetric formulation.
 */
static void step_openmp_nonsymmetric(Particle *p, int n, double dt, double softening) {
    const double G = 1.0;

    // Phase 1: Parallel force computation and velocity update
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        const double xi = p[i].x;
        const double yi = p[i].y;
        const double zi = p[i].z;

        // Each thread accumulates forces for its assigned particles
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            
            const double dx = p[j].x - xi;
            const double dy = p[j].y - yi;
            const double dz = p[j].z - zi;
            const double dist2 = dx * dx + dy * dy + dz * dz + softening * softening;
            const double invDist = 1.0 / sqrt(dist2);
            const double invDist3 = invDist * invDist * invDist;
            const double s = G * p[j].mass * invDist3;
            
            fx += dx * s;
            fy += dy * s;
            fz += dz * s;
        }

        // Update velocity (thread-safe: each thread updates only its own particle)
        p[i].vx += fx * dt;
        p[i].vy += fy * dt;
        p[i].vz += fz * dt;
    }

    // Phase 2: Parallel position update
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

/*
 * ============================================================================
 * VERSION 4: OpenMP - Symmetric (Demonstrating Synchronization Overhead)
 * ============================================================================
 * 
 * This version parallelizes the symmetric formulation, which creates an
 * interesting performance trade-off: reduced arithmetic work but increased
 * synchronization overhead.
 * 
 * CONCURRENCY CHALLENGE:
 * When thread A processes pair (i, j) and thread B processes pair (i, k),
 * both threads need to update particle i simultaneously. Without proper
 * synchronization, this leads to race conditions and incorrect results.
 * 
 * SOLUTION: OpenMP Atomic Operations
 * We use #pragma omp atomic to ensure thread-safe updates to particle forces.
 * Each atomic operation involves:
 * 1. Memory fence to ensure consistency
 * 2. Hardware-level lock or compare-and-swap operation
 * 3. Cache-coherence protocol overhead (MESI/MOESI state transitions)
 * 
 * PERFORMANCE IMPLICATIONS:
 * - Atomic operations serialize access to memory locations
 * - Cache-line ping-pong: When multiple threads update adjacent memory
 *   locations, cache lines bounce between processors (false sharing)
 * - Memory contention increases with thread count
 * - Overhead can exceed the savings from reduced arithmetic operations
 * 
 * EXPECTED BEHAVIOR:
 * For small thread counts: May outperform non-symmetric (50% less work)
 * For large thread counts: May underperform due to contention overhead
 * Crossover point depends on: problem size, thread count, cache architecture
 * 
 * This version demonstrates a fundamental HPC principle: algorithmic
 * efficiency (FLOPs) must be balanced against parallel efficiency
 * (communication and synchronization costs).
 */
static void step_openmp_symmetric(Particle *p, int n, double dt, double softening) {
    const double G = 1.0;

    // Global force arrays (shared across all threads)
    double *fx = (double *)calloc(n, sizeof(double));
    double *fy = (double *)calloc(n, sizeof(double));
    double *fz = (double *)calloc(n, sizeof(double));

    // Phase 1: Parallel force computation with atomic updates
    // Dynamic scheduling handles load imbalance (particle 0 has most work)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n - 1; ++i) {
        const double xi = p[i].x;
        const double yi = p[i].y;
        const double zi = p[i].z;
        const double mi = p[i].mass;

        // Thread-local accumulators for particle i (no contention)
        double fx_i = 0.0;
        double fy_i = 0.0;
        double fz_i = 0.0;

        for (int j = i + 1; j < n; ++j) {
            const double dx = p[j].x - xi;
            const double dy = p[j].y - yi;
            const double dz = p[j].z - zi;
            const double dist2 = dx * dx + dy * dy + dz * dz + softening * softening;
            const double invDist = 1.0 / sqrt(dist2);
            const double invDist3 = invDist * invDist * invDist;
            
            // Compute force components
            const double f = G * invDist3;
            const double fij_x = dx * f;
            const double fij_y = dy * f;
            const double fij_z = dz * f;
            
            // Accumulate force on particle i locally (no atomic needed)
            fx_i += fij_x * p[j].mass;
            fy_i += fij_y * p[j].mass;
            fz_i += fij_z * p[j].mass;
            
            // Apply opposite force to particle j using atomic operations
            // Required because multiple threads may update the same j
            #pragma omp atomic
            fx[j] -= fij_x * mi;
            #pragma omp atomic
            fy[j] -= fij_y * mi;
            #pragma omp atomic
            fz[j] -= fij_z * mi;
        }

        // Write particle i's accumulated force (no atomic needed: one writer)
        fx[i] = fx_i;
        fy[i] = fy_i;
        fz[i] = fz_i;
    }

    // Phase 2: Parallel velocity update
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        p[i].vx += fx[i] * dt;
        p[i].vy += fy[i] * dt;
        p[i].vz += fz[i] * dt;
    }

    // Phase 3: Parallel position update
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }

    free(fx);
    free(fy);
    free(fz);
}

/*
 * ============================================================================
 * MAIN: Comparative Performance Evaluation
 * ============================================================================
 */
int main(int argc, char **argv) {
    int n = 2000;
    int steps = 10;
    double dt = 0.01;
    double softening = 1e-9;

    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        steps = atoi(argv[2]);
    }
    if (argc > 3) {
        dt = atof(argv[3]);
    }

    printf("=============================================================\n");
    printf("N-Body Simulation: Comparative Performance Study\n");
    printf("=============================================================\n");
    printf("Problem size:      %d particles\n", n);
    printf("Timesteps:         %d\n", steps);
    printf("Timestep size:     %g\n", dt);
    printf("OpenMP threads:    %d\n", omp_get_max_threads());
    printf("=============================================================\n\n");

    // Allocate memory for all four versions
    Particle *p1 = (Particle *)malloc((size_t)n * sizeof(Particle));
    Particle *p2 = (Particle *)malloc((size_t)n * sizeof(Particle));
    Particle *p3 = (Particle *)malloc((size_t)n * sizeof(Particle));
    Particle *p4 = (Particle *)malloc((size_t)n * sizeof(Particle));
    
    if (!p1 || !p2 || !p3 || !p4) {
        fprintf(stderr, "Allocation failed\n");
        free(p1); free(p2); free(p3); free(p4);
        return 1;
    }

    // Initialize all versions with identical initial conditions
    init_particles(p1, n, 42u);
    for (int i = 0; i < n; ++i) {
        p2[i] = p1[i];
        p3[i] = p1[i];
        p4[i] = p1[i];
    }

    // Version 1: Sequential Non-Symmetric (Baseline)
    double start = omp_get_wtime();
    for (int s = 0; s < steps; ++s) {
        step_sequential_nonsymmetric(p1, n, dt, softening);
    }
    double time1 = omp_get_wtime() - start;
    printf("1. Sequential Non-Symmetric:  %9.6f seconds  [Baseline]\n", time1);

    // Version 2: Sequential Symmetric
    start = omp_get_wtime();
    for (int s = 0; s < steps; ++s) {
        step_sequential_symmetric(p2, n, dt, softening);
    }
    double time2 = omp_get_wtime() - start;
    printf("2. Sequential Symmetric:      %9.6f seconds  [Speedup: %.2fx]\n", 
           time2, time1 / time2);

    // Version 3: OpenMP Non-Symmetric
    start = omp_get_wtime();
    for (int s = 0; s < steps; ++s) {
        step_openmp_nonsymmetric(p3, n, dt, softening);
    }
    double time3 = omp_get_wtime() - start;
    printf("3. OpenMP Non-Symmetric:      %9.6f seconds  [Speedup: %.2fx]\n", 
           time3, time1 / time3);

    // Version 4: OpenMP Symmetric
    start = omp_get_wtime();
    for (int s = 0; s < steps; ++s) {
        step_openmp_symmetric(p4, n, dt, softening);
    }
    double time4 = omp_get_wtime() - start;
    printf("4. OpenMP Symmetric:          %9.6f seconds  [Speedup: %.2fx]\n", 
           time4, time1 / time4);

    printf("\n=============================================================\n");
    printf("Analysis:\n");
    printf("=============================================================\n");
    printf("Symmetric arithmetic reduction:  %.1f%%\n", 
           (1.0 - time2 / time1) * 100.0);
    printf("OpenMP non-sym parallel eff:     %.1f%%\n", 
           (time1 / time3 / omp_get_max_threads()) * 100.0);
    printf("OpenMP symmetric vs non-sym:     %.2fx %s\n",
           time3 / time4,
           time4 < time3 ? "(faster)" : "(slower)");
    printf("=============================================================\n");

    free(p1);
    free(p2);
    free(p3);
    free(p4);
    return 0;
}
