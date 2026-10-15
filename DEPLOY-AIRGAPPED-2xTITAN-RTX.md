# Airgapped Deployment: 2× TITAN RTX + Qwen3.8-27B-NVFP4

Step-by-step guide for serving `unsloth/Qwen3.8-27B-NVFP4` to many users on an
**airgapped Ubuntu 22.04 server with two TITAN RTX cards (24 GB, SM75, no
NVLink)**. It uses this fork, vLLM 2080 Ti Definitive Edition, packaged as a
Docker image.

> **Status: experimental.** TITAN RTX is the same TU102/SM75 chip as the
> RTX 2080 Ti. However, the shipped routes were validated on **2× 2080 Ti 22 GB
> with NVLink**. Nothing has been benchmarked yet on TITAN RTX, on PCIe-only
> links, or inside this container. Every number below marked *estimate* must
> be confirmed on your own hardware.
>
> **Verified without GPUs:**
> - The image builds end to end.
> - `--print-config` resolves both TITAN profiles inside the container.
> - The host packages install offline on stock and older Ubuntu 22.04 images.
>
> **Not yet verified:** model loading, throughput, and KV capacity on
> TITAN RTX.

## Why this is faster than a GGUF server

- **GGUF splits by layer.** llama.cpp's default multi-GPU mode gives each card
  half of the layers, so the cards take turns. That is why each GPU shows about
  50% utilisation.
- **vLLM splits every layer across both cards** (tensor parallel, TP=2), so
  both GPUs compute at the same time.
- **vLLM batches users together.** Requests from several users share each GPU
  step (continuous batching), which raises total throughput.
- **MTP speculative decoding** speeds up each user's generation.

## How much context can 2× TITAN RTX hold?

| Question | Answer |
|---|---|
| Native max per request | **262,144 tokens** (model limit; recommended) |
| Max per request with YaRN | **524,288 tokens**, one user only. Validated on 2× 22 GB 2080 Ti with the `yarn-fp8kv-1x524K` route. Slower and less accurate. Not tested on Titan. |
| Larger than 524K | No evidence; not supported here |

All concurrent users share one KV-cache pool. The launcher refuses to start
unless `MAX_NUM_SEQS × MAX_MODEL_LEN ≤ GPU KV tokens`.

| Route (NVFP4 + MTP4) | Pool on 2× 2080 Ti 22 GB (measured) | Pool on 2× TITAN RTX (*estimate*) | Shipped TITAN profile |
|---|---|---|---|
| TurboQuant 4-bit KV | 837,832 tokens | ~1.0 M | **3 users × 262K** (default) |
| FP8 KV | 468,978 tokens | ~0.59 M | **8 users × 49K** |

The Titan estimate assumes each 24 GB card adds about 2 GB of KV cache over a
22 GB card. The real pool size is printed in the startup log as
`GPU KV cache size: N tokens` (see step 9).

A request beyond the `MAX_NUM_SEQS` limit **queues**; it is not rejected.

---

## What you need

| Machine | Requirements |
|---|---|
| **Staging box** (online) | Linux x86_64, Docker, about 100 GB free disk, 32 GB RAM or more. No GPU needed. The image build took about 35 minutes on a 16-thread, 31 GB machine, plus download time. |
| **Server** (airgapped) | Ubuntu 22.04, 2× TITAN RTX, Docker, about 80 GB free disk, 64 GB RAM or more recommended, a way to copy files (USB disk or other offline transfer) |

## Step 1: Check the server

Run on the server and write down the results:

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
uname -r                       # kernel, e.g. 5.15.0-139-generic
docker --version
nvidia-smi topo -m             # GPU0<->GPU1 link: PIX/PXB/PHB/SYS (no NV#)
```

Driver **580 or newer** is required, because the image uses CUDA 13.0. If the
driver is older, step 2 also downloads a new one.

## Step 2: Build the offline bundle (staging box)

```bash
git clone https://github.com/chanlilong/vLLM-2080Ti-Definitive.git
cd vLLM-2080Ti-Definitive

# Server driver already >= 580:
docker/sm75-airgap/stage.sh

# Server driver < 580: also fetch a driver plus its build dependencies.
HOST_KERNEL=<uname -r from step 1> DRIVER_VERSION=580.95.05 \
  docker/sm75-airgap/stage.sh
```

The script does five things:

- Builds the image `vllm-def-sm75:0.2.2-post3`: Ubuntu 26.04, GCC 15,
  CUDA 13.0, Python 3.12, torch 2.13, this fork compiled for SM75.
- Saves the image to a file.
- Downloads the model.
- Fetches the nvidia-container-toolkit packages for Ubuntu 22.04 as an
  offline apt repository.
- Writes `SHA256SUMS`.

The result is `./airgap-bundle/`:

```text
vllm-def-sm75_0.2.2-post3.tar.gz   Docker image (14.3 GB unpacked)
models/Qwen3.8-27B-NVFP4/          checkpoint
host-debs/                         local apt repo + install.sh: nvidia-container-toolkit
                                   (+ gcc and kernel headers if HOST_KERNEL set)
driver/                            NVIDIA .run driver (if DRIVER_VERSION set)
run.sh  profiles/  DEPLOY-AIRGAPPED-2xTITAN-RTX.md  SHA256SUMS
```

Options: `OUT=/path`, `MAX_JOBS=8`, `HF_TOKEN=...`, and `SKIP_IMAGE=1`,
`SKIP_MODEL=1`, `SKIP_DEBS=1` to redo one part only.

## Step 3: Copy the bundle to the server

Copy `airgap-bundle/` to the server, for example to `/srv/airgap-bundle`.
Then check it:

```bash
cd /srv/airgap-bundle && sha256sum -c --quiet SHA256SUMS && echo OK
```

## Step 4: Prepare the host (server, once)

```bash
cd /srv/airgap-bundle

# 4a. Only when replacing a driver older than 580 that came from apt: stop
#     everything using the GPUs (e.g. the GGUF server), then remove it.
sudo apt purge 'nvidia-*' 'libnvidia-*'

# 4b. Host packages from the bundled local apt repo, fully offline: the
#     NVIDIA container toolkit, plus gcc and kernel headers when HOST_KERNEL
#     was set. apt also upgrades older base libraries if a package needs it.
sudo host-debs/install.sh

# 4c. Only when the driver is older than 580: install the bundled driver.
sudo sh driver/NVIDIA-Linux-x86_64-580.95.05.run --silent
sudo reboot

# 4d. Let Docker use the GPUs
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4e. Let vLLM memory-map large checkpoint files (the launcher checks this)
echo 'vm.overcommit_memory=1' | sudo tee /etc/sysctl.d/99-vllm.conf
sudo sysctl --system
```

## Step 5: Load the image and place the model

```bash
docker load -i /srv/airgap-bundle/vllm-def-sm75_0.2.2-post3.tar.gz
sudo mkdir -p /srv/models
sudo mv /srv/airgap-bundle/models/Qwen3.8-27B-NVFP4 /srv/models/

# GPU visible inside the container?
docker run --rm --gpus all vllm-def-sm75:0.2.2-post3 nvidia-smi
```

## Step 6: Check GPU peer-to-peer

```bash
nvidia-smi topo -p2p r
```

- `OK` between GPU0 and GPU1: vLLM's custom all-reduce is used.
- `NS` (not supported, common on GeForce/TITAN without NVLink): vLLM falls back
  to NCCL through host memory automatically. It still works, but expect lower
  decode speed than the published 2080 Ti NVLink numbers.

## Step 7: Preview the configuration

```bash
cd /srv/airgap-bundle
MODEL_DIR=/srv/models/Qwen3.8-27B-NVFP4 ./run.sh --print-config
```

`run.sh` settings, all set as environment variables:

| Variable | Default |
|---|---|
| `MODEL_DIR` | required |
| `PROFILE_FILE` | the 3 × 262K profile |
| `GPUS` | `0,1` |
| `PORT` | `8000` |
| `STATE_DIR` | `/srv/vllm-def` (logs and kernel caches) |
| `SERVED_NAME` | model folder name |
| `API_KEY` | none |

Any extra arguments are passed through to `launcher.sh`.

## Step 8: Start the server

Pick one of the two profiles.

**Default: 3 users × 262K**

```bash
MODEL_DIR=/srv/models/Qwen3.8-27B-NVFP4 ./run.sh --start-timeout 3600
```

**Many chat users: 8 users × 49K**

```bash
MODEL_DIR=/srv/models/Qwen3.8-27B-NVFP4 \
PROFILE_FILE=profiles/2xTITANRTX-nvfp4-mtp4-fp8kv-8x49K-text-only.env \
  ./run.sh --start-timeout 3600
```

Then follow the startup:

```bash
docker logs -f vllm-def        # wait for "START OK"
```

The **first start compiles GPU kernels** (Triton, FlashInfer, FlashQLA) and
can take a long time; `--start-timeout 3600` allows for this. The kernels are
cached in `/srv/vllm-def`, so later restarts are much faster. The container
restarts automatically after a reboot (`--restart unless-stopped`).

## Step 9: Verify

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B-NVFP4",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "max_tokens": 64}'

grep -h "GPU KV cache size" /srv/vllm-def/run-logs/*.log | tail -1   # real pool size
nvidia-smi dmon -s u            # both GPUs should be busy during generation
```

Users connect with any OpenAI-compatible client at
`http://<server-ip>:8000/v1`.

## Step 10: Tune users vs. context

1. Read `N` from `GPU KV cache size: N tokens`.
2. Pick values with `MAX_NUM_SEQS × MAX_MODEL_LEN ≤ N`, for example
   4 × 131072 or 12 × 32768.
3. Copy one of `profiles/*.env`, edit `MAX_MODEL_LEN` / `MAX_NUM_SEQS`, and
   restart:

```bash
docker rm -f vllm-def
MODEL_DIR=/srv/models/Qwen3.8-27B-NVFP4 PROFILE_FILE=/srv/my-route.env ./run.sh
```

Rules:

- Keep KV at `fp8` or `turboquant_4bit_nc`. **Never use FP16 KV with NVFP4**;
  it causes a known quality collapse.
- If startup runs out of memory, lower `GPU_UTIL` by 0.01–0.02.
- Avoid sending several ultra-long prompts at the same moment.

## Operating the service

| Task | Command |
|---|---|
| Logs | `docker logs -f vllm-def` or `/srv/vllm-def/run-logs/` |
| Stop / start | `docker stop vllm-def` / `docker start vllm-def` |
| Remove (to change settings) | `docker rm -f vllm-def` |
| Require an API key | `API_KEY=secret ./run.sh ...`, then clients send `Authorization: Bearer secret` |
| Tool calling / reasoning flags | Append launcher flags, e.g. `./run.sh --enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| All launcher options | `docker run --rm vllm-def-sm75:0.2.2-post3 ./launcher.sh --help` |
| Upgrade | Rebuild the bundle on staging, `docker load` the new image, `docker rm -f vllm-def`, then `./run.sh` |

When `API_KEY` is set, the launcher's startup smoke test is skipped, because
that test does not send a key.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Step 4b/4d was not done, or Docker was not restarted |
| `CUDA driver version is insufficient` | Host driver is older than 580 (steps 4a–4c) |
| `configured concurrency cannot be admitted by the allocated GPU KV cache` | Lower `MAX_NUM_SEQS` or `MAX_MODEL_LEN` (step 10) |
| `checkpoint mmap needs vm.overcommit_memory=1` | Step 4e |
| Out of memory during startup | Lower `GPU_UTIL` in the profile; close other GPU processes (Xorg, the old GGUF server) |
| Startup timed out on first run | Use a larger `--start-timeout`; compiled kernels persist in `STATE_DIR` |
| Low speed, GPUs not busy | Check step 6; PCIe without P2P is slower but should still keep both GPUs active |

## Files used by this guide

- `docker/sm75-airgap/Dockerfile`: image build, run by `stage.sh`. Its build
  context rules are in `Dockerfile.dockerignore`.
- `docker/sm75-airgap/entrypoint.sh`: starts `launcher.sh` and keeps the
  container in the foreground.
- `docker/sm75-airgap/stage.sh`: creates the offline bundle on the staging box.
- `docker/sm75-airgap/run.sh`: starts the container on the server.
- `docker/sm75-airgap/profiles/`: experimental TITAN RTX route profiles. They
  are outside the validated `profiles/` matrix.

## Credits

Built on [upstream vLLM](https://github.com/vllm-project/vllm) (Apache-2.0)
and **vLLM 2080 Ti Definitive Edition** by
[github.com/weicj](https://github.com/weicj). This airgap packaging is a
derivative deployment recipe. Keep these credits when you redistribute it.
