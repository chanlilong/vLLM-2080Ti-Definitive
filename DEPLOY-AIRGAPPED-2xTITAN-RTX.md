# Airgapped Deployment: 2× TITAN RTX + Qwen3.8-27B-NVFP4

Serve `unsloth/Qwen3.8-27B-NVFP4` to many users from an **airgapped server
with two TITAN RTX cards (24 GB, SM75)**, using this fork, vLLM 2080 Ti
Definitive Edition, as a Docker image. Clients connect through
[qwen38-effort-proxy](https://github.com/chanlilong/qwen38-effort-proxy), which
forces every incoming thinking-effort value to `medium`.

**Deployment = two Docker images + model weights + `compose.yaml`.**

| Container | Image | Port |
|---|---|---|
| `qwen38-proxy` | `qwen38-proxy:local` | **8080**: the only port clients use |
| `vllm-def` | `vllm-def-sm75:0.2.2-post3` | 8000 on loopback only, for debugging |

> **Status: experimental.** TITAN RTX is the same TU102/SM75 chip as the
> RTX 2080 Ti, but the shipped routes were validated on **2× 2080 Ti 22 GB with
> NVLink**. Nothing has been benchmarked yet on TITAN RTX, on PCIe-only links,
> or inside this container. Numbers marked *estimate* must be confirmed on
> your own hardware.
>
> **Verified without an SM75 GPU:** the image builds end to end; `compose.yaml`
> resolves all three routes against the real checkpoint (including the YaRN
> rope override); the proxy rewrites effort to `medium` on both API surfaces.
> **Not yet verified:** model loading, throughput and KV capacity on TITAN RTX.
> This fork compiles its CUDA kernels for SM75 only, so it cannot be smoke-tested
> on newer GPUs; the first real GPU run is on the TITAN server.

## Requirements

| Machine | Needs |
|---|---|
| **Online build machine** | Linux x86_64, Docker, about 100 GB free disk, 32 GB RAM or more. No GPU. |
| **Airgapped server** | 2× TITAN RTX, NVIDIA driver **580 or newer** (the image uses CUDA 13.0), Docker with the NVIDIA Container Toolkit, about 60 GB free disk, 64 GB RAM or more recommended |

## Step 1: Build the images and fetch the weights (online machine)

```bash
git clone https://github.com/chanlilong/vLLM-2080Ti-Definitive.git
git clone https://github.com/chanlilong/qwen38-effort-proxy.git

# vLLM image: compiles this fork for SM75. No GPU needed.
cd vLLM-2080Ti-Definitive
docker build -f docker/sm75-airgap/Dockerfile -t vllm-def-sm75:0.2.2-post3 .
docker save vllm-def-sm75:0.2.2-post3 | pigz > vllm-def-sm75_0.2.2-post3.tar.gz   # or gzip, slower

# Proxy image.
docker build --target runtime -t qwen38-proxy:local ../qwen38-effort-proxy
docker save qwen38-proxy:local | pigz > qwen38-proxy_local.tar.gz

# Weights: downloaded with the hf CLI inside the image (about 22 GB).
mkdir -p models
docker run --rm -e HF_HUB_OFFLINE=0 -v "$PWD/models:/models" \
  vllm-def-sm75:0.2.2-post3 \
  hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir /models/Qwen3.8-27B-NVFP4
```

If the build machine sits behind a TLS-inspecting proxy, the builds need the
proxy's CA certificates; downloads otherwise fail with certificate errors. The
proxy image takes them as a build secret
(`docker build --secret id=ca_cert,src=<ca-bundle.crt> ...`, see its README).

## Step 2: Copy four things to the server

| File | Size |
|---|---|
| `vllm-def-sm75_0.2.2-post3.tar.gz` | vLLM image, ~9 GB compressed (16.2 GB loaded) |
| `qwen38-proxy_local.tar.gz` | Proxy image, ~56 MB compressed |
| `models/Qwen3.8-27B-NVFP4/` | Model weights, ~22 GB |
| `docker/sm75-airgap/compose.yaml` | Service definition |

## Step 3: Load and start (airgapped server)

```bash
docker load -i vllm-def-sm75_0.2.2-post3.tar.gz
docker load -i qwen38-proxy_local.tar.gz
docker run --rm --gpus all vllm-def-sm75:0.2.2-post3 nvidia-smi   # both TITANs listed?

sudo mkdir -p /srv/models
sudo mv models/Qwen3.8-27B-NVFP4 /srv/models/

docker compose -f compose.yaml up -d
docker compose -f compose.yaml logs -f vllm   # wait for "START OK"
```

The model is mounted read-only from the host into the container in
`compose.yaml`:

```yaml
volumes:
  - ${MODEL_DIR:-/srv/models/Qwen3.8-27B-NVFP4}:/models/Qwen3.8-27B-NVFP4:ro
```

To keep the weights somewhere else, set `MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4`
in the environment or in a `.env` file next to `compose.yaml`.

The **first start compiles GPU kernels** (Triton, FlashInfer, FlashQLA) and can
take a long time. They are cached in Docker volumes, so later starts are much
faster. The service restarts automatically after a reboot.

## Step 4: Verify (through the proxy)

```bash
curl http://localhost:8080/_proxy/config      # shows force_effort: medium
curl http://localhost:8080/v1/models

curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B-NVFP4",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "max_tokens": 256}'

docker compose logs vllm | grep "GPU KV cache size"   # real KV pool size
nvidia-smi dmon -s u            # both GPUs should be busy during generation
```

Clients connect to the **proxy** only:

- OpenAI-compatible apps: `http://<server-ip>:8080/v1`
- Claude Code:

  ```bash
  ANTHROPIC_BASE_URL=http://<server-ip>:8080 \
  ANTHROPIC_API_KEY=dummy ANTHROPIC_AUTH_TOKEN=dummy \
  ANTHROPIC_DEFAULT_OPUS_MODEL=Qwen3.8-27B-NVFP4 \
  ANTHROPIC_DEFAULT_SONNET_MODEL=Qwen3.8-27B-NVFP4 \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.8-27B-NVFP4 \
  claude
  ```

### Effort forcing

This model's chat template accepts only `xhigh` (its default), `medium` and
`low` thinking effort; other values such as `high` fail. The proxy rewrites
every effort field (`reasoning_effort` on `/v1/chat/completions`,
`output_config.effort` on Anthropic `/v1/messages`) to `FORCE_EFFORT`, adds it
when an Anthropic request omits it, and strips Anthropic `thinking` blocks.
To change it, edit `FORCE_EFFORT` in `compose.yaml` and run
`docker compose up -d proxy`. Port 8000 bypasses the proxy and is published on
loopback only.

## Routes: users vs. context

`compose.yaml` ships three route profiles. Exactly one `--profile-file` line is
active; switch by moving the comment and running `docker compose up -d`.

| Route | Users × context | KV | MTP | Evidence |
|---|---|---|---|---|
| `yarn-fp8kv-1x524K` (**compose default**) | **1 × 524,288** | FP8 | off | Validated route on 2× 2080 Ti 22 GB. Not tested on TITAN. |
| `2xTITANRTX-…-tq4nc-3x262K` | 3 × 262,144 | TurboQuant 4-bit | 4 | Experimental. Same shape as the validated 2080 Ti route. |
| `2xTITANRTX-…-fp8kv-8x49K` | 8 × 49,152 | FP8 | 4 | Experimental. |

**YaRN** stretches the model's native 262,144-token limit to 524,288 by
scaling RoPE by 2.0. The launcher adds this automatically from the profile's
`ENABLE_YARN=1`. It serves **one request at a time** (others queue), has no MTP
speculative decoding, and is slower and less accurate than native context.
For many users with shorter prompts, use one of the other two routes.

All concurrent users share one KV-cache pool. The launcher refuses to start
unless `MAX_NUM_SEQS × MAX_MODEL_LEN ≤ GPU KV tokens`. A request beyond the
`MAX_NUM_SEQS` limit **queues**; it is not rejected.

| Route (NVFP4 + MTP4) | Pool on 2× 2080 Ti 22 GB (measured) | Pool on 2× TITAN RTX (*estimate*) |
|---|---|---|
| TurboQuant 4-bit KV | 837,832 tokens | ~1.0 M |
| FP8 KV | 468,978 tokens | ~0.59 M |

The TITAN estimate assumes each 24 GB card adds about 2 GB of KV cache over a
22 GB card.

**Custom shape:** read `N` from `GPU KV cache size: N tokens`, then pick values
with `MAX_NUM_SEQS × MAX_MODEL_LEN ≤ N` (for example 4 × 131072 or 12 × 32768).
Copy one of the profiles in `docker/sm75-airgap/profiles/`, edit
`MAX_MODEL_LEN` and `MAX_NUM_SEQS`, mount it and point to it:

```yaml
    volumes:
      - /srv/my-route.env:/profiles/my-route.env:ro
    command:
      - --profile-file=/profiles/my-route.env
```

Rules:

- Keep KV at `fp8` or `turboquant_4bit_nc`. **Never use FP16 KV with NVFP4**;
  it causes a known quality collapse.
- If startup runs out of memory, lower `GPU_UTIL` by 0.01–0.02.
- Avoid sending several ultra-long prompts at the same moment.

## Operating the service

| Task | Command |
|---|---|
| Logs | `docker compose logs -f vllm` / `docker compose logs -f proxy` (shows each effort rewrite) |
| Stop / start | `docker compose stop` / `docker compose start` |
| Apply `compose.yaml` changes | `docker compose up -d` |
| Require an API key | Uncomment `VLLM_API_KEY` and `SKIP_STARTUP_SMOKE` in `compose.yaml`; clients send `Authorization: Bearer <key>` (the proxy passes it through) |
| Tool calling | On by default (`--enable-auto-tool-choice=1`, parser `qwen3_xml`); remove that line to turn it off |
| Other GPUs | Change `device_ids` (host GPU numbers); keep `--gpu-devices=0,1` |
| All launcher options | `docker run --rm vllm-def-sm75:0.2.2-post3 ./launcher.sh --help` |
| Upgrade | Build and copy a new image, `docker load` it, then `docker compose up -d` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing, or Docker not restarted after installing it |
| `requirement error: unsatisfied condition: cuda>=13.0` | Host driver is older than 580 |
| `configured concurrency cannot be admitted by the allocated GPU KV cache` | Use a smaller shape (see "Users vs. context") |
| `checkpoint mmap needs vm.overcommit_memory=1` | On the host: `echo 'vm.overcommit_memory=1' \| sudo tee /etc/sysctl.d/99-vllm.conf && sudo sysctl --system` |
| Out of memory during startup | Lower `GPU_UTIL` in the profile; close other GPU processes (Xorg, an old GGUF server) |
| Startup timed out on first run | Raise `--start-timeout`; compiled kernels persist in the volumes |
| Proxy returns 502 `Cannot reach upstream vLLM` | vLLM is still starting; wait for `START OK` in `docker compose logs vllm` |
| Low speed | Check `nvidia-smi topo -p2p r`. `NS` between the GPUs (common on TITAN without NVLink) means all-reduce goes through host memory: it works, but expect lower decode speed than the 2080 Ti NVLink numbers |

## Why this is faster than a GGUF server

- **GGUF splits by layer.** llama.cpp's default multi-GPU mode gives each card
  half of the layers, so the cards take turns (about 50% utilisation each).
- **vLLM splits every layer across both cards** (tensor parallel, TP=2), so
  both GPUs compute at the same time.
- **vLLM batches users together** (continuous batching), which raises total
  throughput.
- **MTP speculative decoding** speeds up each user's generation.

## Files used by this guide

- `docker/sm75-airgap/Dockerfile`: the image. Its build context rules are in
  `Dockerfile.dockerignore`.
- `docker/sm75-airgap/compose.yaml`: the vLLM and proxy services.
- [qwen38-effort-proxy](https://github.com/chanlilong/qwen38-effort-proxy):
  the effort-forcing proxy image (separate repository).
- `docker/sm75-airgap/entrypoint.sh`: starts `launcher.sh` and keeps the
  container in the foreground.
- `docker/sm75-airgap/profiles/`: experimental TITAN RTX route profiles,
  baked into the image. They are outside the validated `profiles/` matrix.

## Credits

Built on [upstream vLLM](https://github.com/vllm-project/vllm) (Apache-2.0)
and **vLLM 2080 Ti Definitive Edition** by
[github.com/weicj](https://github.com/weicj). This airgap packaging is a
derivative deployment recipe. Keep these credits when you redistribute it.
