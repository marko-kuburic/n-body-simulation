// Compatibility shim for CUDA 11.x + glibc 2.31+ rsqrt conflict
// This header must be included before any standard C/C++ headers

#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

// Include CUDA runtime first
#include <cuda_runtime.h>

// Now create namespace-safe wrappers to prevent glibc conflicts
// The issue: glibc declares rsqrt with noexcept(true) but CUDA declares it without
// Solution: Don't let glibc declare these functions at all

// Save CUDA's rsqrt declarations
#ifdef __CUDACC__
namespace cuda_compat {
    using ::rsqrt;
    using ::rsqrtf;
}
#endif

// Prevent glibc from declaring rsqrt by predefining it
#define rsqrt rsqrt_hidden_by_cuda
#define rsqrtf rsqrtf_hidden_by_cuda

// Now include system headers
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// Restore the names
#undef rsqrt
#undef rsqrtf

// Re-expose CUDA's version in device code
#ifdef __CUDACC__
using cuda_compat::rsqrt;
using cuda_compat::rsqrtf;
#endif

// Make sure we have sqrt for host code
#include <cmath>
using std::sqrt;

#endif // CUDA_COMPAT_H
