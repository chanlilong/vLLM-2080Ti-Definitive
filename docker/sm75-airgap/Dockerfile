# vLLM 2080 Ti Definitive Edition - SM75 runtime image for airgapped hosts.
#
# Based on upstream vLLM and vLLM 2080 Ti Definitive Edition
# (github.com/weicj). This image builds the 0.2.x runtime (CUDA 13.0,
# Python 3.12, torch 2.13.0+cu130) on NVIDIA's CUDA 13.0.3 devel image for
# Ubuntu 22.04, the same base as upstream vLLM's Docker build. An older host
# (for example Ubuntu 22.04) only needs an NVIDIA R580+ driver and
# nvidia-container-toolkit.
#
# Build on an internet-connected machine from the repository root:
#   docker build -f docker/sm75-airgap/Dockerfile -t vllm-def-sm75:0.2.2-post3 .
# A GPU is not required for the build. See DEPLOY-AIRGAPPED-2xTITAN-RTX.md.
#
# This differs from the fork's validated host target (Ubuntu 26.04, GCC 15,
# kernel >= 7): host code is compiled with GCC 11 and the kernel is the
# host's, so build.sh runs with ALLOW_HOST_MISMATCH=1. CUDA device code is
# still compiled by nvcc 13.0. Treat routes built this way as not validated
# until benchmarked.

FROM nvidia/cuda:13.0.3-devel-ubuntu22.04

ARG MAX_JOBS=
ARG TORCH_CUDA_ARCH_LIST=7.5

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates patch \
        procps iproute2 kmod libnuma1 && \
    rm -rf /var/lib/apt/lists/* && \
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

# Offline runtime defaults. launcher.sh helpers call python3; the runtime venv
# (Python 3.12) comes first on PATH so it provides it.
# NVIDIA_REQUIRE_CUDA drops the base image's datacenter forward-compat driver
# exceptions: the container runtime refuses to start on a driver older than
# R580 instead of failing later inside vLLM.
ENV PATH=/opt/vllm-def/.venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_HAS_FLASHINFER_CUBIN=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0"

EXPOSE 8000
ENTRYPOINT ["/opt/vllm-def/docker/sm75-airgap/entrypoint.sh"]
CMD ["serve"]
