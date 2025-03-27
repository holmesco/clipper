FROM ubuntu:22.04

# Install essential tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    vim \
    gdb \
    lldb \
    valgrind \
    iputils-ping \
    libeigen3-dev \
    ffmpeg\
    libsm6 \
    libxext6 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a workspace directory
WORKDIR /workspace

# Set up shared memory
RUN mkdir -p /dev/shm && chmod 1777 /dev/shm

# Default command
CMD ["/bin/bash"]