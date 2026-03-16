# MLX Distributed Ring Inference

Run large language models across multiple Mac devices using MLX's distributed ring backend over Thunderbolt networking. Port 8100.

| | |
|---|---|
| **Stack** | Python 3.14 · MLX · FastAPI · Pydantic |
| **Hardware** | 2x Mac Mini M4 16GB via Thunderbolt Bridge |
| **Model** | mlx-community/Qwen3-14B-4bit (tensor parallelism) |
| **Runs on** | mini1 + mini2 (native macOS, no K8s) |

## What MLX Distributed Ring Inference Does

When an OpenAI-compatible chat completion request arrives at the FastAPI server on port 8100, the API process serializes the prompt and forwards it to the distributed inference server over a Unix domain socket (`/tmp/mlx_ring.sock`). The inference server runs as a multi-rank MLX process launched by `mlx.launch`, with rank 0 on the master machine and rank 1 on the worker connected via Thunderbolt Bridge TCP at ~10-20 Gbps.

The inference server uses tensor parallelism (`model.shard()`) to split attention heads and MLP layers evenly across ranks. When rank 0 receives a prompt, it broadcasts the text and generation parameters to all ranks using `mx.distributed.all_sum` over a fixed-size byte buffer. Every rank then tokenizes the prompt identically, and all ranks participate in `stream_generate` together. After each transformer layer, an all-reduce operation synchronizes partial results across the ring. Rank 0 collects the generated tokens, assembles performance metrics (prompt eval tok/s, generation tok/s), and sends the response back through the socket to the API server.

The API server formats the response into OpenAI-compatible JSON with usage statistics and performance data, then returns it to the caller. A simple conversation cache on rank 0 tracks previous prompts per conversation ID, enabling cache-hit detection for repeated prefixes. The entire system is stateless beyond this in-memory cache -- there is no database, no persistent storage, and no authentication. The project runs exclusively on macOS with Apple Silicon and does not deploy to Kubernetes.

## Features

- **Tensor parallelism via model.shard()** — splits attention heads and MLP neurons across ranks so each Mac processes only its fraction of every layer, enabling models larger than a single machine's memory
- **OpenAI-compatible API** — exposes `/v1/chat/completions` following the OpenAI format so existing clients and tools work without modification
- **Unix domain socket IPC** — decouples the FastAPI server from the distributed inference process so the API can restart independently without reloading the model
- **Automatic file sync** — `launch.sh` detects the local machine, generates `hosts.json`, and SCPs server code, config, and `.env` to the worker before launching
- **Prompt broadcasting** — encodes prompts as padded byte arrays and uses `all_sum` to distribute text across ranks without requiring shared filesystem access
- **Conversation caching** — tracks prompt prefixes per conversation ID on rank 0 for cache-hit detection on repeated contexts
- **RDMA-ready architecture** — switching from `ring` (TCP) to `jaccl` (RDMA over Thunderbolt 5) requires only a backend flag change and `rdma` fields in `hosts.json`, enabling ~80 Gbps transfers
- **Validated configuration** — Pydantic-style dataclass config with type-safe environment variable loading, cross-config property delegation, and helpful validation error messages
- **Machine auto-detection** — `launch.sh` inspects local network interfaces to determine whether it is running on mini1 or mini2 and configures master/worker roles accordingly

## Quick Start

```bash
pip3 install --break-system-packages mlx mlx-lm python-dotenv uvicorn fastapi pydantic huggingface_hub
cp .env.example .env    # Edit with your model/settings
./launch.sh start       # Syncs files to worker, loads model, starts API
curl http://localhost:8100/health
```

## API Endpoints

### Chat `/v1/chat`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Generate chat completion (OpenAI format) |

### Model

| Method | Path | Description |
|--------|------|-------------|
| GET | `/model/info` | Model metadata, context length, device count |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Health check with config summary |
| GET | `/health/live` | Liveness probe |

## Configuration

All configuration via environment variables (loaded from `.env` by python-dotenv).

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_REPO` | `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit` | HuggingFace model repository |
| `MODEL_CACHE_DIR` | `/Users/${USER}/.cache/huggingface/hub` | Model cache directory |
| `API_HOST` | `192.168.5.1` | API listen host |
| `API_PORT` | `8100` | API listen port |
| `WORKER_HOSTS` | `192.168.5.2` | Comma-separated worker IPs |
| `WORKER_SSH` | `mini2@192.168.5.2` | Comma-separated worker SSH strings |
| `SOCKET_PATH` | `/tmp/mlx_ring.sock` | Unix socket for API-to-server IPC |
| `NUM_DEVICES` | `2` | Number of distributed ranks |
| `DISTRIBUTED_BACKEND` | `ring` | MLX backend (ring, jaccl) |
| `KV_CACHE_MAX_SIZE` | *(none)* | Max KV-cache size (empty = no limit) |
| `KV_CACHE_RESERVED_MEMORY_MB` | `2048` | Reserved memory for KV-cache in MB |
| `MAX_SEQUENCE_LENGTH` | `4096` | Maximum sequence length |
| `MAX_PROMPT_LEN_BYTES` | `4096` | Max prompt broadcast buffer in bytes |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Request timeout |
| `POLL_INTERVAL_SECONDS` | `0.1` | Server poll interval for new requests |
| `DEFAULT_MAX_TOKENS` | `50` | Default max tokens for generation |
| `FILE_DESCRIPTOR_SOFT_LIMIT` | `2048` | Soft FD limit |
| `FILE_DESCRIPTOR_HARD_LIMIT` | `4096` | Hard FD limit |
| `MODEL_LOAD_WAIT_SECONDS` | `15` | Wait time after launching distributed server |
| `LOG_LEVEL` | `INFO` | Logging level |

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

## Directory Structure

```
mlx_distributed_ring_inference_v2/
├── launch.sh              # Launcher — auto-detects machine, syncs files, runs mlx.launch + API
├── server.py              # Distributed inference server (runs on all ranks via mlx.launch)
├── api.py                 # FastAPI server — OpenAI-compatible endpoint (rank 0 only)
├── test_config.py         # Configuration system test script
├── .env.example           # Environment variable template with all defaults
├── hosts.json             # Generated by launch.sh — MLX host definitions
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata and tool config
├── config/
│   ├── __init__.py        # Singleton config loader (get_config)
│   └── manager.py         # 7 dataclass config sections + 3 facade classes + env loading
├── distributed/
│   ├── __init__.py        # Re-exports broadcast_prompt, shard_and_load
│   └── utils.py           # Prompt broadcasting via all_sum + model shard/load logic
└── archived_features/     # Deprecated experiments (MoE, prompt cache, memory-aware sharding)
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

**All devices must have equal RAM.** MLX does not support heterogeneous clusters -- it cannot give more weight to a machine with more memory. This is a known limitation ([GitHub Issue #1804](https://github.com/ml-explore/mlx/issues/1804)).

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

## Design Principles

1. **Rank-symmetric execution** — every rank runs the same `server.py` code with the same model layers; only rank 0 additionally handles socket I/O and the API server, keeping the codebase simple
2. **Decoupled API and inference** — the FastAPI process communicates with the distributed server via Unix socket so the API can restart without reloading the ~8 GB model across two machines
3. **Zero infrastructure** — no Docker, no Kubernetes, no database; the entire system is two Python processes per machine launched by a shell script over SSH

## Testing

```bash
python3 test_config.py                    # Configuration validation tests (4 test cases)
./launch.sh test                          # Live inference smoke test (requires running server)
```

No pytest suite exists. Tests require Apple Silicon hardware with MLX installed.

## CI/CD

No CI/CD pipeline. This project runs exclusively on local Apple Silicon hardware (2x Mac Mini M4) connected via Thunderbolt. It has no Dockerfile, no GitHub Actions workflow, and does not deploy to the cluster. Code is synced to the worker machine via SCP in `launch.sh`.

## Dependencies

- **MLX** (>=0.10.0) — Apple's ML framework for distributed tensor operations and all-reduce communication
- **mlx-lm** (>=0.10.0) — model loading, tokenization, sharding, and stream generation
- **FastAPI** (>=0.100.0) — REST API framework for the OpenAI-compatible endpoint
- **Uvicorn** (>=0.23.0) — ASGI server
- **Pydantic** (>=2.0.0) — request/response validation and configuration dataclasses
- **huggingface-hub** (>=0.20.0) — model downloading and snapshot management
- **transformers** (>=4.36.0) — tokenizer support
- **NumPy** (>=1.24.0) — array utilities
- **python-dotenv** (>=1.0.0) — environment variable loading from `.env` files

## Known Issues

- **No streaming support** — `stream: true` in chat completion requests returns HTTP 501. The distributed server streams internally but the socket IPC sends the full response at once.
- **Homogeneous RAM required** — MLX tensor parallelism splits evenly across ranks with no support for heterogeneous memory. All devices must have equal RAM. See [mlx#1804](https://github.com/ml-explore/mlx/issues/1804).
- **No automated tests** — `test_config.py` covers configuration validation only. There are no unit or integration tests for the inference pipeline since MLX requires Apple Silicon hardware.
- **Archived dead code** — `archived_features/` contains deprecated experiments (MoE sharding, prompt cache, memory-aware sharding) that are not used by any live code path.
- **Default model mismatch** — `.env.example` and `config/manager.py` default to `DeepSeek-Coder-V2-Lite-Instruct-8bit`, but the live setup and README examples use `Qwen3-14B-4bit`. The actual model is controlled by the `.env` file.
