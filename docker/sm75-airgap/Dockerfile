# vLLM 2080 Ti Definitive Edition - SM75 runtime image for airgapped hosts.
#
# Based on upstream vLLM and vLLM 2080 Ti Definitive Edition
# (github.com/weicj). This image reproduces the 0.2.x build target
# (Ubuntu 26.04 userland, GCC 15, CUDA 13.0, Python 3.12, torch 2.13.0+cu130)
# inside a container, so an older host (for example Ubuntu 22.04) only needs an
# NVIDIA R580+ driver and nvidia-container-toolkit.
#
# Build on an internet-connected machine from the repository root:
#   docker build -f docker/sm75-airgap/Dockerfile -t vllm-def-sm75:0.2.2-post3 .
# A GPU is not required for the build. See DEPLOY-AIRGAPPED-2xTITAN-RTX.md.
#
# The host kernel check in build.sh cannot pass inside a container (the kernel
# is the host's), so the build runs with ALLOW_HOST_MISMATCH=1. Treat routes on
# such hosts as not validated until benchmarked.

FROM ubuntu:26.04

ARG CUDA_REPO=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
ARG MAX_JOBS=
ARG TORCH_CUDA_ARCH_LIST=7.5

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-15 g++-15 git curl ca-certificates patch \
        procps iproute2 kmod libnuma1 && \
    rm -rf /var/lib/apt/lists/*

# CUDA 13.0 toolkit only (the driver comes from the host), from NVIDIA's
# Ubuntu 24.04 apt repository as in the nvidia/cuda devel images; the runfile
# installer needs libxml2.so.2, which Ubuntu 26.04 no longer ships.
# glibc >= 2.41 needs the repository's CUDA math header patch (build.sh checks).
COPY toolchain-patches/cuda-13.0-glibc-2.41-rsqrt.patch /tmp/cuda-rsqrt.patch
RUN curl -fsSL -o /tmp/cuda-keyring.deb "$CUDA_REPO/cuda-keyring_1.1-1_all.deb" && \
    dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        cuda-minimal-build-13-0 cuda-libraries-dev-13-0 cuda-command-line-tools-13-0 && \
    rm -rf /var/lib/apt/lists/* && \
    patch -p1 -d / < /tmp/cuda-rsqrt.patch && \
    rm -f /tmp/cuda-rsqrt.patch && \
    nvcc --version

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version

WORKDIR /opt/vllm-def
COPY . /opt/vllm-def

# build.sh downloads Python 3.12, wheels, CUTLASS, triton_kernels and FlashQLA
# here, while the build machine is online. The runtime keeps nvcc/gcc so
# Triton, FlashInfer and FlashQLA JIT kernels can compile offline.
RUN if [ -z "$MAX_JOBS" ]; then unset MAX_JOBS; fi && \
    ASSUME_YES=1 ALLOW_HOST_MISMATCH=1 TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    ./build.sh && \
    rm -rf build-logs /root/.cache/uv

# Offline runtime defaults. launcher.sh helpers call python3; the image has
# no system Python, so the runtime venv (Python 3.12) provides it.
ENV PATH=/opt/vllm-def/.venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_HAS_FLASHINFER_CUBIN=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000
ENTRYPOINT ["/opt/vllm-def/docker/sm75-airgap/entrypoint.sh"]
CMD ["serve"]
