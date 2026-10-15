#!/usr/bin/env bash
# Start the SM75 airgap image on the target host (no network needed).
#
#   ./run.sh                     start detached, restart unless stopped
#   ./run.sh --print-config      preview the resolved route and exit
#   ./run.sh --set KEY=VALUE     extra launcher.sh options are passed through
#
# Settings (environment variables):
#   IMAGE         docker image tag            (default vllm-def-sm75:0.2.2-post3)
#   MODEL_DIR     host checkpoint directory   (required)
#   SERVED_NAME   OpenAI model name           (default: basename of MODEL_DIR)
#   PROFILE_FILE  route profile; a path inside the image, or a host file
#                 (default: the experimental TITAN RTX 3x262K profile)
#   STATE_DIR     host dir for logs and JIT caches (default /srv/vllm-def)
#   GPUS          host GPU indices for TP=2   (default 0,1)
#   PORT          host port                   (default 8000)
#   NAME          container name              (default vllm-def)
#   API_KEY       optional bearer token; skips the launcher smoke test
set -euo pipefail

IMAGE=${IMAGE:-vllm-def-sm75:0.2.2-post3}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the host checkpoint directory}
SERVED_NAME=${SERVED_NAME:-$(basename -- "$MODEL_DIR")}
PROFILE_FILE=${PROFILE_FILE:-/opt/vllm-def/docker/sm75-airgap/profiles/2xTITANRTX-nvfp4-mtp4-tq4nc-3x262K-text-only.env}
STATE_DIR=${STATE_DIR:-/srv/vllm-def}
GPUS=${GPUS:-0,1}
PORT=${PORT:-8000}
NAME=${NAME:-vllm-def}

app=/opt/vllm-def
mounts=(-v "$(realpath -- "$MODEL_DIR"):/models/$SERVED_NAME:ro")
for dir in run-logs triton-cache torchinductor-cache dot-cache root-cache; do
  mkdir -p "$STATE_DIR/$dir"
done
mounts+=(
  -v "$STATE_DIR/run-logs:$app/run-logs"
  -v "$STATE_DIR/triton-cache:$app/triton-cache"
  -v "$STATE_DIR/torchinductor-cache:$app/torchinductor-cache"
  -v "$STATE_DIR/dot-cache:$app/.cache"
  -v "$STATE_DIR/root-cache:/root/.cache"
)

# A host profile file is mounted read-only next to the shipped ones.
if [[ -f "$PROFILE_FILE" ]]; then
  mounts+=(-v "$(realpath -- "$PROFILE_FILE"):/profiles/$(basename -- "$PROFILE_FILE"):ro")
  PROFILE_FILE=/profiles/$(basename -- "$PROFILE_FILE")
fi

envs=()
if [[ -n "${API_KEY:-}" ]]; then
  envs+=(-e "VLLM_API_KEY=$API_KEY" -e SKIP_STARTUP_SMOKE=1)
fi

run_mode=(-d --restart unless-stopped)
for arg in "$@"; do
  if [[ "$arg" == "--print-config" ]]; then
    run_mode=(--rm)
  fi
done

if [[ "${run_mode[0]}" == "-d" ]] && docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Container $NAME already exists. Stop it first: docker rm -f $NAME" >&2
  exit 1
fi

docker run "${run_mode[@]}" --name "$NAME" \
  --gpus "\"device=$GPUS\"" \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --stop-timeout 90 \
  -p "$PORT:8000" \
  "${mounts[@]}" "${envs[@]}" \
  "$IMAGE" serve \
  --model-dir "/models/$SERVED_NAME" \
  --served-name "$SERVED_NAME" \
  --profile-file "$PROFILE_FILE" \
  --gpu-devices 0,1 --tp-size 2 --pp-size 1 \
  --service-scope lan --port 8000 \
  "$@"

if [[ "${run_mode[0]}" == "-d" ]]; then
  echo "Started $NAME. Follow startup with: docker logs -f $NAME"
fi
