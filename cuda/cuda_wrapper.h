#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

// Workaround for CUDA 13.1 + GCC 14/15 + glibc 2.31+ incompatibility
// The issue: CUDA's math_functions.h declares rsqrt/rsqrtf without noexcept,
// but glibc 2.31+ declares them with noexcept(true), causing conflicts.
//
// Solution: Define rsqrt/rsqrtf as macros BEFORE including any headers,
// preventing the conflicting declarations from appearing.

// Hide the problematic glibc declarations by defining them as macros first
#define rsqrt  __nvvm_rsqrt
#define rsqrtf __nvvm_rsqrtf

// Now include CUDA headers (which will use the CUDA versions)
#include <cuda_runtime.h>

// Undefine the macros so user code can call rsqrt normally
#undef rsqrt
#undef rsqrtf

// Include standard C/C++ headers AFTER cuda_runtime.h
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#endif // CUDA_WRAPPER_H
