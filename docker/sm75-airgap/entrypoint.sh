#!/usr/bin/env bash
# Container entrypoint for the SM75 airgap image.
#
#   serve [launcher.sh options]   start the service and stay in the foreground
#   <any other command>           run it as-is (bash, nvidia-smi, ...)
#
# launcher.sh starts vLLM detached and exits after its health and smoke checks.
# A container stops when PID 1 exits, so this script follows the server log and
# waits for the server process, forwarding docker stop to its process group.
set -euo pipefail

ROOT=/opt/vllm-def
cd "$ROOT"

if [[ "${1:-serve}" != "serve" ]]; then
  exec "$@"
fi
[[ $# -gt 0 ]] && shift

for arg in "$@"; do
  if [[ "$arg" == "--print-config" ]]; then
    exec ./launcher.sh --non-interactive "$@"
  fi
done

./launcher.sh --non-interactive "$@"

state_file=${STATE_FILE:-${LOG_DIR:-$ROOT/run-logs}/start-manager.state}
pid_file=$(
  # shellcheck disable=SC1090
  source "$state_file" && printf '%s' "${LAST_PID_FILE:-}"
)
log_file=$(
  # shellcheck disable=SC1090
  source "$state_file" && printf '%s' "${LAST_LOG_FILE:-}"
)
if [[ ! -f "$pid_file" ]]; then
  echo "ERROR: launcher did not record a server PID file (state: $state_file)" >&2
  exit 1
fi
server_pid=$(<"$pid_file")

tail -n 0 -F "$log_file" &
tail_pid=$!

stop_server() {
  echo "Stopping vLLM server (pid $server_pid)..."
  kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$server_pid" 2>/dev/null || true
  kill "$tail_pid" 2>/dev/null || true
  exit 0
}
trap stop_server TERM INT

while kill -0 "$server_pid" 2>/dev/null; do
  sleep 5 &
  wait $! || true
done

kill "$tail_pid" 2>/dev/null || true
echo "ERROR: vLLM server exited. Last log lines:" >&2
tail -n 50 "$log_file" >&2 || true
exit 1
