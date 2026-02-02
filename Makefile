CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99 -fopenmp -lm
TARGET = nbody_naive

# CUDA configuration
NVCC = nvcc
NVCC_ARCH ?= -gencode=arch=compute_80,code=sm_80
# Use system default GCC (container has GCC 11 which is compatible)
NVCC_FLAGS = -O3 --std=c++14 $(NVCC_ARCH)
CUDA_TARGET = nbody_cuda
CUDA_SRC = cuda/nbody_cuda.cu

all: $(TARGET)

cuda: $(CUDA_TARGET)

both: $(TARGET) $(CUDA_TARGET)

$(TARGET): nbody_naive.c
	$(CC) $(CFLAGS) -o $(TARGET) nbody_naive.c

$(CUDA_TARGET): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(CUDA_TARGET) $(CUDA_SRC)

clean:
	rm -f $(TARGET) $(CUDA_TARGET)

run: $(TARGET)
	./$(TARGET)

# Example runs with different problem sizes
test: $(TARGET)
	@echo "Running with 1000 particles, 10 steps:"
	./$(TARGET) 1000 10
	@echo ""
	@echo "Running with 2000 particles, 10 steps:"
	./$(TARGET) 2000 10
	@echo ""
	@echo "Running with 4000 particles, 5 steps:"
	./$(TARGET) 4000 5

.PHONY: all clean run test
