#!/usr/bin/env python3
"""
Generate figures for N-body benchmark report.
Creates: strong scaling, formulation comparison, size scaling plots.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

mpl.use('Agg')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

# Create figures directory
os.makedirs('figures', exist_ok=True)

# =============================================================================
# DATA FROM BENCHMARK RESULTS
# =============================================================================

# Strong scaling data (N=6000, steps=10, threads=[1,2,4,8,16,24,32])
threads = [1, 2, 4, 8, 16, 24, 32]

# From summary_median.csv - strong_scaling experiment
omp_nonsym_times = [1.813667, 1.114349, 0.559432, 0.291829, 0.342464, 0.233363, 0.178518]
omp_sym_times = [5.260966, 9.749897, 22.334717, 14.056635, 3.860161, 2.332812, 1.646942]
seq_sym_time = 1.275324  # Baseline for speedup

# Calculate speedups (relative to sequential symmetric)
omp_nonsym_speedup = [seq_sym_time / t for t in omp_nonsym_times]
omp_sym_speedup = [seq_sym_time / t for t in omp_sym_times]

# Ideal speedup
ideal_speedup = threads

# Size sweep data (threads=32)
sizes = [1000, 2000, 4000, 6000, 8000, 10000, 12000]

# Sequential times
seq_nonsym_size = [0.05008, 0.201771, 0.801216, 1.796762, 3.198836, 4.989877, 8.804456]
seq_sym_size = [0.035543, 0.141851, 0.567212, 1.275863, 2.267964, 3.54754, 6.246697]

# OpenMP times (32 threads)
omp_nonsym_size = [0.010932, 0.028721, 0.084821, 0.178872, 0.313639, 0.476791, 0.687039]
omp_sym_size = [0.093591, 0.284719, 0.855043, 1.646348, 2.632241, 3.785473, 5.147511]

# CUDA times
cuda_size = [0.016812, 0.033257, 0.058434, 0.086361, 0.119244, 0.298214, 0.354094]

# =============================================================================
# FIGURE 1: Strong Scaling (OpenMP)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Execution time
ax1.plot(threads, omp_nonsym_times, 'o-', label='OpenMP несиметрична', linewidth=2, markersize=8, color='C0')
ax1.plot(threads, omp_sym_times, 's-', label='OpenMP симетрична', linewidth=2, markersize=8, color='C1')
ax1.axhline(y=seq_sym_time, color='gray', linestyle='--', label='Секв. симетрична (базна линија)')
ax1.set_xlabel('Број нити', fontsize=12)
ax1.set_ylabel('Време извршавања [s]', fontsize=12)
ax1.set_title('Време извршавања vs. број нити (N=6000)', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(threads)
ax1.set_yscale('log')

# Right: Speedup
ax2.plot(threads, omp_nonsym_speedup, 'o-', label='OpenMP несиметрична', linewidth=2, markersize=8, color='C0')
ax2.plot(threads, omp_sym_speedup, 's-', label='OpenMP симетрична', linewidth=2, markersize=8, color='C1')
ax2.plot(threads, ideal_speedup, 'k--', label='Идеално убрзање', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Број нити', fontsize=12)
ax2.set_ylabel('Убрзање (vs. секв. симетрична)', fontsize=12)
ax2.set_title('Убрзање vs. број нити (N=6000)', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(threads)

plt.tight_layout()
plt.savefig('figures/strong_scaling.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/strong_scaling.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/strong_scaling.pdf")

# =============================================================================
# FIGURE 2: Formulation Comparison (Sequential)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(sizes))
width = 0.35

bars1 = ax.bar(x - width/2, seq_nonsym_size, width, label='Секв. несиметрична', color='C0', alpha=0.8)
bars2 = ax.bar(x + width/2, seq_sym_size, width, label='Секв. симетрична', color='C1', alpha=0.8)

ax.set_xlabel('Број честица (N)', fontsize=12)
ax.set_ylabel('Време извршавања [s]', fontsize=12)
ax.set_title('Поређење формулација: секвенцијална имплементација', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in sizes])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add speedup annotations
for i, (t1, t2) in enumerate(zip(seq_nonsym_size, seq_sym_size)):
    speedup = t1 / t2
    ax.annotate(f'{speedup:.2f}×', xy=(i, max(t1, t2)), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=9, color='green')

plt.tight_layout()
plt.savefig('figures/formulation_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/formulation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/formulation_comparison.pdf")

# =============================================================================
# FIGURE 3: Final Comparison (Sequential vs OpenMP vs CUDA)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Execution time
ax1.plot(sizes, seq_sym_size, 'o-', label='Секв. симетрична', linewidth=2, markersize=8, color='C0')
ax1.plot(sizes, omp_nonsym_size, 's-', label='OpenMP несиметрична', linewidth=2, markersize=8, color='C1')
ax1.plot(sizes, cuda_size, '^-', label='CUDA', linewidth=2, markersize=8, color='C2')
ax1.set_xlabel('Број честица (N)', fontsize=12)
ax1.set_ylabel('Време извршавања [s]', fontsize=12)
ax1.set_title('Поређење имплементација: време извршавања', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(sizes)
ax1.set_xticklabels([str(n) for n in sizes], rotation=45)
ax1.set_yscale('log')

# Right: Speedup vs sequential
omp_speedup = [seq_sym_size[i] / omp_nonsym_size[i] for i in range(len(sizes))]
cuda_speedup = [seq_sym_size[i] / cuda_size[i] for i in range(len(sizes))]

ax2.plot(sizes, omp_speedup, 's-', label='OpenMP несиметрична', linewidth=2, markersize=8, color='C1')
ax2.plot(sizes, cuda_speedup, '^-', label='CUDA', linewidth=2, markersize=8, color='C2')
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
ax2.set_xlabel('Број честица (N)', fontsize=12)
ax2.set_ylabel('Убрзање (vs. секв. симетрична)', fontsize=12)
ax2.set_title('Убрзање у односу на секвенцијалну имплементацију', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(sizes)
ax2.set_xticklabels([str(n) for n in sizes], rotation=45)

plt.tight_layout()
plt.savefig('figures/final_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/final_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/final_comparison.pdf")

# =============================================================================
# FIGURE 4: OpenMP Formulation Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(sizes))
width = 0.35

bars1 = ax.bar(x - width/2, omp_nonsym_size, width, label='OpenMP несиметрична', color='C0', alpha=0.8)
bars2 = ax.bar(x + width/2, omp_sym_size, width, label='OpenMP симетрична', color='C1', alpha=0.8)

ax.set_xlabel('Број честица (N)', fontsize=12)
ax.set_ylabel('Време извршавања [s]', fontsize=12)
ax.set_title('Поређење формулација: OpenMP имплементација (32 нити)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in sizes])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add speedup annotations (nonsym/sym ratio - how much faster is nonsym)
for i, (t1, t2) in enumerate(zip(omp_nonsym_size, omp_sym_size)):
    speedup = t2 / t1  # sym/nonsym - how much faster is nonsym
    ax.annotate(f'{speedup:.1f}×', xy=(i, max(t1, t2)), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=9, color='green')

plt.tight_layout()
plt.savefig('figures/omp_formulation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/omp_formulation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/omp_formulation.pdf")

print("\nAll figures generated successfully!")
