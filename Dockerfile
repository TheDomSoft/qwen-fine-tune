# Dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit and cuDNN (already included in the base image)
# Just verify CUDA installation
RUN nvcc --version

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with CUDA support
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt

# Set up Hugging Face cache directory
ENV HF_HOME=/workspace/.cache/huggingface
RUN mkdir -p $HF_HOME

# Default command
CMD ["bash"]