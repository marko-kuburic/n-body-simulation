# Simple CUDA N-body Docker build  
# Base image with CUDA 12.2 and GCC 11 (compatible with driver 580.x supporting CUDA 13.0)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install build essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for analysis
RUN pip3 install numpy matplotlib pandas

# Set working directory
WORKDIR /work

# Copy project files
COPY . /work/

# Default command: build and run a quick test
CMD ["bash", "-c", "cd /work && make cuda && ./nbody_cuda 1000 5"]
