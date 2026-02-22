# MLX Distributed Ring Inference

Run large language models across multiple Mac devices using MLX's distributed ring backend over Thunderbolt networking. Split models across 2+ Macs using tensor parallelism for collaborative inference.

## Table of Contents
- [Overview](#overview)
- [Current Setup](#current-setup)
- [Prerequisites](#prerequisites)
- [Complete Setup Guide](#complete-setup-guide)
  - [Step 1: Check System Requirements](#step-1-check-system-requirements)
  - [Step 2: Install Required Software](#step-2-install-required-software)
  - [Step 3: Connect Devices via Thunderbolt](#step-3-connect-devices-via-thunderbolt)
  - [Step 4: Configure Thunderbolt Networking](#step-4-configure-thunderbolt-networking)
  - [Step 5: Set Up SSH Access](#step-5-set-up-ssh-access)
  - [Step 6: Install the Project](#step-6-install-the-project)
  - [Step 7: Run Distributed Inference](#step-7-run-distributed-inference)
- [Using the System](#using-the-system)
- [Performance](#performance)
- [RDMA / Thunderbolt 5](#rdma--thunderbolt-5)
- [Scaling: Adding More Devices](#scaling-adding-more-devices)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)

## Overview

This project distributes a large language model across multiple Mac computers connected via Thunderbolt, using MLX's ring communication backend. It uses **tensor parallelism** (`model.shard()`) to split attention heads and MLP layers across ranks, so each Mac processes a fraction of every layer in parallel.

## Current Setup

| Component | Value |
|-----------|-------|
| **Model** | `mlx-community/Qwen3-14B-4bit` |
| **Devices** | 2x Mac Mini M4 (16GB each) |
| **Parallelism** | Tensor parallelism via `model.shard()` |
| **Communication** | MLX ring backend over Thunderbolt Bridge |
| **IPC** | Unix domain socket (`/tmp/mlx_ring.sock`) |
| **API** | OpenAI-compatible at `http://localhost:8100` |
| **Master** | mini2 (192.168.5.2) |
| **Worker** | mini1 (192.168.5.1) |

## Prerequisites

### Hardware Requirements

- 2+ Mac computers with Apple Silicon (M1/M2/M3/M4)
- Minimum 16GB RAM per Mac
- Thunderbolt 3/4/5 cable(s) connecting devices
- All devices must have **equal RAM** (MLX splits evenly — see [Scaling](#scaling-adding-more-devices))

### Software Requirements

- macOS Sequoia (15.0) or later (macOS 26.2+ for RDMA)
- Python 3.11+ (tested with 3.14)
- MLX 0.30.5+, mlx-lm 0.30.7+
- `python-dotenv`, `uvicorn`, `fastapi`, `pydantic`

## Complete Setup Guide

### Step 1: Check System Requirements

On **each Mac**:

```bash
sw_vers -productVersion        # macOS version
uname -m                       # Should be arm64
sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}'  # RAM
```

### Step 2: Install Required Software

On **each Mac**:

```bash
# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python
brew install python@3.14

# MLX and dependencies (--break-system-packages for Homebrew Python)
pip3 install --break-system-packages mlx mlx-lm python-dotenv uvicorn fastapi pydantic huggingface_hub

# Verify
python3 -c "import mlx.core as mx; print('MLX', mx.__version__)"
python3 -c "import mlx_lm; print('mlx-lm', mlx_lm.__version__)"
```

### Step 3: Connect Devices via Thunderbolt

1. Connect Macs directly with a Thunderbolt cable
2. For 2 Macs: direct cable between any TB port
3. For 3+ Macs: daisy chain or use a Thunderbolt hub

### Step 4: Configure Thunderbolt Networking

On **each Mac**, go to System Settings > Network > Thunderbolt Bridge > Details > TCP/IP:

| Mac | IP Address | Subnet Mask |
|-----|-----------|-------------|
| mini1 (worker) | 192.168.5.1 | 255.255.255.0 |
| mini2 (master) | 192.168.5.2 | 255.255.255.0 |
| Additional | 192.168.5.N | 255.255.255.0 |

Verify: `ping -c 3 192.168.5.1` from mini2

### Step 5: Set Up SSH Access

```bash
# Enable Remote Login on each Mac:
# System Settings > General > Sharing > Remote Login

# On the master (mini2), set up passwordless SSH:
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -i ~/.ssh/id_ed25519 mini1@192.168.5.1

# Verify
ssh mini1@192.168.5.1 "echo 'SSH works'"
```

### Step 6: Install the Project

On the **master** (mini2):

```bash
git clone <repo-url> ~/Movies/mlx_distributed_ring_inference_v2
cd ~/Movies/mlx_distributed_ring_inference_v2
cp .env.example .env  # Edit with your model/settings
```

On **each worker**, create the matching directory:

```bash
# On mini1
mkdir -p ~/Movies/mlx_distributed_ring_inference_v2/distributed
mkdir -p ~/Movies/mlx_distributed_ring_inference_v2/config
```

The launch script auto-syncs `server.py`, `config/`, `distributed/`, and `.env` to workers.

Download the model on **every** machine:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-14B-4bit')"
```

### Step 7: Run Distributed Inference

```bash
cd ~/Movies/mlx_distributed_ring_inference_v2
./launch.sh start
```

Test:

```bash
curl http://localhost:8100/health
curl http://localhost:8100/model/info | python3 -m json.tool

curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-14b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

## Using the System

### Starting and Stopping

```bash
./launch.sh start     # Stop existing, then start
./launch.sh stop      # Stop everything
./launch.sh restart   # Stop + start
./launch.sh status    # Check processes and API health
./launch.sh test      # Run a test inference
```

### Monitoring

```bash
tail -f server.log    # Distributed server logs
tail -f api.log       # API server logs
```

### API Usage

OpenAI-compatible endpoint at `http://localhost:8100`:

```python
import requests

response = requests.post(
    "http://localhost:8100/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "What is machine learning?"}],
        "max_tokens": 100,
        "temperature": 0.7,
    },
)
data = response.json()
print(data["choices"][0]["message"]["content"])
print(data["performance"])  # tokens/sec metrics
```

### Configuration

Create a `.env` file (see `.env.example`):

```bash
MODEL_REPO=mlx-community/Qwen3-14B-4bit
API_HOST=0.0.0.0
API_PORT=8100
SOCKET_PATH=/tmp/mlx_ring.sock
NUM_DEVICES=2
DEFAULT_MAX_TOKENS=2048
MAX_SEQUENCE_LENGTH=8192
LOG_LEVEL=INFO
```

## Performance

Measured with Qwen3-14B-4bit on 2x Mac Mini M4 16GB over Thunderbolt Bridge (TCP):

| Metric | Value |
|--------|-------|
| **Prompt eval** | 15-65 tok/s (scales with prompt length) |
| **Generation** | ~12 tok/s |
| **Model load time** | ~7 seconds |
| **Memory per device** | ~8 GB (half the 4-bit model) |
| **Network** | Thunderbolt Bridge TCP (~10-20 Gbps effective) |

## RDMA / Thunderbolt 5

MLX supports RDMA (Remote Direct Memory Access) over Thunderbolt 5 via the `jaccl` backend. This bypasses the TCP/IP stack entirely for ~80 Gb/s direct memory transfers between machines.

### Requirements

- **macOS 26.2 or later** on all machines
- **Thunderbolt 5** cable and ports
- **Apple Silicon M4** or later
- RDMA must be **enabled in Recovery Mode** on each machine

### Enabling RDMA

On **each Mac**:

1. Shut down the Mac
2. Boot into Recovery Mode (hold power button on Apple Silicon)
3. Open Terminal from the Utilities menu
4. Enable RDMA (follow Apple's instructions for your macOS version)
5. Restart normally

### Launching with RDMA

Once RDMA is enabled, update `launch.sh` to use the `jaccl` backend:

```bash
# Change --backend ring to --backend jaccl
mlx.launch --hostfile hosts.json --backend jaccl --verbose python3 server.py
```

The hostfile needs an `rdma` field per host specifying RDMA device paths:

```json
[
    {"ssh": "127.0.0.1", "ips": ["192.168.5.2"], "rdma": [null, "rdma_device_path"]},
    {"ssh": "mini1@192.168.5.1", "ips": ["192.168.5.1"], "rdma": ["rdma_device_path", null]}
]
```

The `null` entries are for self-connections (each host's own position in the list).

### Expected Performance Improvement

| Backend | Bandwidth | Latency | Best For |
|---------|-----------|---------|----------|
| `ring` (TCP over TB) | ~10-20 Gbps | Higher | Works out of the box |
| `jaccl` (RDMA over TB5) | ~80 Gbps | Minimal | Maximum throughput |

RDMA should significantly improve generation speed since tensor parallelism requires all-reduce communication on every layer.

### Mac Studio Note

On Mac Studio, the Thunderbolt port **adjacent to the Ethernet port** may not support RDMA. Use a different TB5 port.

## Scaling: Adding More Devices

### How MLX Distributes Work

MLX's tensor parallelism (`model.shard()`) divides attention heads and MLP weights **equally** across all ranks. With N devices, each gets `1/N` of the computation per layer.

**All devices must have equal RAM.** MLX does not support heterogeneous clusters — it cannot give more weight to a machine with more memory. This is a known limitation ([GitHub Issue #1804](https://github.com/ml-explore/mlx/issues/1804)).

### Scaling Options

| Configuration | Total RAM | Max Model (4-bit) | Notes |
|--------------|-----------|-------------------|-------|
| 2x Mac Mini M4 16GB | 32 GB | ~24 GB models | Current setup |
| 3x Mac Mini M4 16GB | 48 GB | ~36 GB models | Add one more Mini |
| 4x Mac Mini M4 16GB | 64 GB | ~48 GB models | Requires daisy-chain or hub |
| 1x Mac Studio M4 128GB | 128 GB | ~100 GB models | Single machine, no network overhead |

### Adding a Device

1. Assign a TB Bridge IP (e.g., `192.168.5.3`)
2. Set up SSH keys from master
3. Install dependencies and download the model
4. Update `hosts.json` in `launch.sh`:

```json
[
    {"ssh": "127.0.0.1", "ips": ["192.168.5.2"]},
    {"ssh": "mini1@192.168.5.1", "ips": ["192.168.5.1"]},
    {"ssh": "user@192.168.5.3", "ips": ["192.168.5.3"]}
]
```

### Heterogeneous Clusters

If you need mixed-RAM devices (e.g., a Mac Studio + Mac Minis), consider:

- **[Exo](https://github.com/exo-explore/exo)** — Supports RAM-proportional layer allocation
- **[mzbac/mlx_sharding](https://github.com/mzbac/mlx_sharding)** — Manual `--start-layer`/`--end-layer` per node

### Tensor Parallelism vs Pipeline Parallelism

| Approach | Method | Communication | Supported Models |
|----------|--------|--------------|-----------------|
| **Tensor** (`shard()`) | Splits heads/MLP across ranks | All-reduce every layer | Most models (Qwen3, Llama, etc.) |
| **Pipeline** (`pipeline()`) | Splits layers across ranks | Point-to-point between stages | DeepSeek V3, Ministral only |

This project uses tensor parallelism. Pipeline parallelism is only available for specific model architectures that inherit `PipelineMixin` in mlx-lm.

## Troubleshooting

### Common Issues

#### "Model cache directory not found"
Ensure `python-dotenv` is installed and `.env` has the correct `MODEL_REPO`:
```bash
pip3 install --break-system-packages python-dotenv
```
Also download the model on the failing machine:
```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-14B-4bit')"
```

#### "'Qwen3Model' object has no attribute 'pipeline'"
The model doesn't support pipeline parallelism. Use `model.shard(group)` in `distributed/utils.py` instead of `model.model.pipeline(group)`.

#### localhost treated as remote (SSH to self)
MLX 0.30.5+ requires `"127.0.0.1"` (not `"localhost"`) in `hosts.json` for local detection. The launcher checks `host == "127.0.0.1"` as a strict string match.

#### "server.py not found" on remote node
Add `--cwd /path/on/remote/machine` to the `mlx.launch` command. Without it, MLX sends the master's CWD to all nodes. The CWD must be valid on the **remote** machine. The local node inherits CWD from the parent process.

#### "No module named 'uvicorn'"
```bash
pip3 install --break-system-packages uvicorn fastapi pydantic
```

#### Connection refused on API port
1. Check server.log for errors: `cat server.log`
2. Ensure port 8100 is free: `lsof -i :8100`
3. Stop and restart: `./launch.sh restart`

### Logs

```bash
cat server.log    # Distributed inference server
cat api.log       # FastAPI server
```

## Architecture

```
mini2 (192.168.5.2)              mini1 (192.168.5.1)
┌──────────────────────┐         ┌──────────────────────┐
│  Master Node         │         │  Worker Node         │
│  ┌────────────────┐  │  Ring   │  ┌────────────────┐  │
│  │  server.py     │◄─┼────────┼──►  server.py     │  │
│  │  Rank 0        │  │  (TB)  │  │  Rank 1        │  │
│  │  All layers    │  │        │  │  All layers    │  │
│  │  (1/2 heads)   │  │        │  │  (1/2 heads)   │  │
│  └───────┬────────┘  │        │  └────────────────┘  │
│          │ Unix sock │        │                      │
│  ┌───────▼────────┐  │        │                      │
│  │  api.py        │  │        │                      │
│  │  FastAPI       │  │        │                      │
│  │  :8100         │  │        │                      │
│  └────────────────┘  │        │                      │
└──────────────────────┘         └──────────────────────┘
```

With tensor parallelism, **every layer exists on every node**, but each node processes only its fraction of the attention heads and MLP neurons. All-reduce operations synchronize results across ranks after each layer.

## Project Structure

```
mlx_distributed_ring_inference_v2/
├── launch.sh          # Launcher — manages hosts.json, file sync, mlx.launch, API
├── server.py          # Distributed MLX inference server (runs on all ranks)
├── api.py             # FastAPI server — OpenAI-compatible endpoint (rank 0 only)
├── .env               # Configuration (model, network, performance settings)
├── config/            # Configuration facades (Pydantic models)
├── distributed/       # Distributed utilities (sharding, prompt broadcast)
├── hosts.json         # Generated by launch.sh — MLX host definitions
├── requirements.txt   # Python dependencies
└── pyproject.toml     # Project metadata
```

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) by Apple's machine learning research team
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) for model loading and sharding
- [Awni Hannun's distributed inference guide](https://gist.github.com/awni/ec071fd27940698edd14a4191855bba6)

## License

MIT License
