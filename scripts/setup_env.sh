#!/bin/bash
set -e

echo "Setting up TokenSmith environment (Conda-only dependencies)..."

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected: $OS $ARCH"

# Check for required build tools on Linux
if [[ "$OS" == "Linux" ]]; then
    echo "Checking for required build dependencies..."
    
    # Check for build-essential
    if ! command -v gcc &> /dev/null || ! command -v g++ &> /dev/null; then
        echo "WARNING: gcc/g++ not found. You may need to install build-essential:"
        echo "  sudo apt update && sudo apt install build-essential"
    fi
    
    # Check for OpenBLAS (optional but recommended)
    if ! ldconfig -p 2>/dev/null | grep -q libopenblas; then
        echo "NOTE: OpenBLAS not found. For better CPU performance, consider installing:"
        echo "  sudo apt install libopenblas-dev"
        echo "Continuing with basic CPU support..."
    fi
fi

# Platform-specific CMAKE_ARGS for llama-cpp-python
if [[ "$OS" == "Darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        echo "Apple Silicon detected - enabling Metal support"
        export CMAKE_ARGS="-DGGML_METAL=on -DGGML_ACCELERATE=on"
        export FORCE_CMAKE=1
    fi
elif [[ "$OS" == "Linux" ]]; then
    # Check for CUDA toolkit (not just nvidia-smi)
    if command -v nvcc &> /dev/null && [ -d "/usr/local/cuda" ]; then
        echo "NVIDIA CUDA Toolkit detected - enabling CUDA support"
        export CMAKE_ARGS="-DGGML_CUDA=on"
        export FORCE_CMAKE=1
    elif command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected but CUDA toolkit not installed"
        echo "Building CPU-only version with OpenBLAS optimization"
        export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        export FORCE_CMAKE=1
    else
        echo "Building CPU-only version"
        export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        export FORCE_CMAKE=1
    fi
fi

# Install llama-cpp-python with platform-specific optimizations
# (This is one of the few packages that needs pip due to compilation flags)
if [[ -n "$CMAKE_ARGS" ]]; then
    echo "Installing llama-cpp-python with: $CMAKE_ARGS"
    CMAKE_ARGS="$CMAKE_ARGS" pip install llama-cpp-python --force-reinstall --no-cache-dir
else
    pip install llama-cpp-python
fi

echo "TokenSmith environment setup complete!"
echo "All dependencies managed by Conda."
