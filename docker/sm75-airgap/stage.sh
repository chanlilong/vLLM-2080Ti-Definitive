#!/usr/bin/env bash
# Build an offline bundle on an internet-connected machine (Docker required,
# GPU not required). Run from the repository root:
#
#   docker/sm75-airgap/stage.sh
#
# Output (default ./airgap-bundle):
#   vllm-def-sm75_<version>.tar.gz   docker image      (skip: SKIP_IMAGE=1)
#   models/<name>/                   checkpoint(s)     (skip: SKIP_MODEL=1)
#   host-debs/                       nvidia-container-toolkit for the host
#                                    Ubuntu release    (skip: SKIP_DEBS=1)
#   driver/                          NVIDIA .run driver when DRIVER_VERSION set
#   run.sh, profiles/, DEPLOY-*.md, SHA256SUMS
#
# Settings:
#   OUT=./airgap-bundle  MODEL_ID=unsloth/Qwen3.8-27B-NVFP4  DRAFT_ID=
#   HOST_UBUNTU=22.04    HOST_KERNEL=   (target `uname -r`; adds build-essential
#                                        and kernel headers for a .run driver)
#   DRIVER_VERSION=      (for example 580.95.05)
#   MAX_JOBS=            (vLLM compile parallelism; build.sh picks one if empty)
#   HF_TOKEN=            (only if the model repository needs it)
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
VERSION=$(<"$ROOT/VERSION")
IMAGE=${IMAGE:-vllm-def-sm75:$VERSION}
OUT=$(realpath -m -- "${OUT:-$PWD/airgap-bundle}")
MODEL_ID=${MODEL_ID:-unsloth/Qwen3.8-27B-NVFP4}
DRAFT_ID=${DRAFT_ID:-}
HOST_UBUNTU=${HOST_UBUNTU:-22.04}
HOST_KERNEL=${HOST_KERNEL:-}
DRIVER_VERSION=${DRIVER_VERSION:-}

mkdir -p "$OUT"
cd "$ROOT"

if [[ "${SKIP_IMAGE:-0}" != "1" ]]; then
  echo "==> Building $IMAGE (vLLM compiles from source; this takes a while)"
  docker build -f docker/sm75-airgap/Dockerfile \
    --build-arg MAX_JOBS="${MAX_JOBS:-}" -t "$IMAGE" .
  echo "==> Saving image"
  docker save "$IMAGE" | gzip -1 > "$OUT/vllm-def-sm75_${VERSION}.tar.gz"
fi

download_model() {
  local repo=$1 name
  name=$(basename -- "$repo")
  echo "==> Downloading $repo"
  docker run --rm \
    -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/models/.hf-home \
    -v "$OUT/models:/models" "$IMAGE" \
    bash -c "/opt/vllm-def/.venv/bin/hf download '$repo' --local-dir '/models/$name' &&
             rm -rf /models/.hf-home '/models/$name/.cache' &&
             chown -R $(id -u):$(id -g) /models"
}

if [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  mkdir -p "$OUT/models"
  download_model "$MODEL_ID"
  [[ -n "$DRAFT_ID" ]] && download_model "$DRAFT_ID"
fi

if [[ "${SKIP_DEBS:-0}" != "1" ]]; then
  echo "==> Fetching host packages for Ubuntu $HOST_UBUNTU"
  mkdir -p "$OUT/host-debs"
  packages="nvidia-container-toolkit"
  [[ -n "$HOST_KERNEL" ]] && packages+=" build-essential linux-headers-$HOST_KERNEL"
  # The full dependency closure is shipped as a local apt repository, so apt
  # on the host resolves a consistent set even when it is behind on updates.
  docker run --rm -v "$OUT/host-debs:/out" "ubuntu:$HOST_UBUNTU" bash -euc "
    apt-get update -qq
    apt-get install -y -qq curl gpg ca-certificates dpkg-dev >/dev/null
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
      gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    cd /out && chown _apt /out
    apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts \
      --no-breaks --no-replaces --no-enhances $packages |
      grep '^[a-z0-9]' | sort -u | xargs apt-get download -qq
    dpkg-scanpackages --multiversion . /dev/null 2>/dev/null | gzip -9c > Packages.gz
    chown -R $(id -u):$(id -g) /out"
  cat > "$OUT/host-debs/install.sh" <<EOF
#!/usr/bin/env bash
# Offline install of the bundled host packages (Ubuntu $HOST_UBUNTU).
# Usage: sudo ./install.sh [package ...]   (default: $packages)
set -euo pipefail
dir=\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)
list=\$(mktemp)
echo "deb [trusted=yes] file:\$dir ./" > "\$list"
apt_opts=(-o Dir::Etc::sourcelist="\$list" -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0)
apt-get "\${apt_opts[@]}" update
if ((\$#)); then pkgs=("\$@"); else pkgs=($packages); fi
apt-get "\${apt_opts[@]}" install -y "\${pkgs[@]}"
rm -f "\$list"
EOF
  chmod +x "$OUT/host-debs/install.sh"
fi

if [[ -n "$DRIVER_VERSION" ]]; then
  echo "==> Downloading NVIDIA driver $DRIVER_VERSION"
  mkdir -p "$OUT/driver"
  curl -fL --retry 3 -o "$OUT/driver/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run" \
    "https://us.download.nvidia.com/XFree86/Linux-x86_64/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run"
fi

cp docker/sm75-airgap/run.sh "$OUT/"
cp -r docker/sm75-airgap/profiles "$OUT/"
cp DEPLOY-AIRGAPPED-2xTITAN-RTX.md "$OUT/"

echo "==> Writing checksums"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z |
  xargs -0 sha256sum > SHA256SUMS)
du -sh "$OUT"
echo "Bundle ready: $OUT"
