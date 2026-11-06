FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Dependencies are installed once when this image is published, never during
# ordinary IDMesh CI jobs. Ubuntu 24.04 supplies CMake >= 3.26.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libopenmpi-dev \
        openmpi-bin \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# GitHub Actions job containers run as root; OpenMPI requires explicit opt-in.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN cmake --version && mpicxx --version && mpirun --version
