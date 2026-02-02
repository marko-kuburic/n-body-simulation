#!/bin/bash
# Simple Docker build script for CUDA N-body

set -e

echo "Building Docker image for CUDA N-body..."
podman build -t nbody-cuda -f Dockerfile . || docker build -t nbody-cuda -f Dockerfile .

echo "✓ Docker image 'nbody-cuda' built successfully"
echo ""
echo "To run: ./docker-run.sh"
