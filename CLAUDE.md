# MLX Distributed Ring Inference — Project Context

## ⏸️ DECOMMISSIONED 2026-05-14 — bridge dismantled, not currently runnable

The mini1↔mini2 Thunderbolt Bridge that this project runs over has been **physically
disconnected** (cables pulled by the user 2026-05-14). Reason: LLM inference now runs on
**pgx** (NVIDIA GB10), so the 2×M4 distributed ring is no longer needed. Nothing here is
broken — it is intentionally idle.

**Why it was pulled:** the bridge had *multiple* Thunderbolt cables between the two minis,
all enslaved to `bridge0` with macOS STP not arbitrating → a layer-2 loop → ~196k pkt/s
broadcast storm that pinned `mDNSResponder`/`netbiosd` and drove load to ~11 on both minis.

### To bring it back
1. **Reconnect exactly ONE Thunderbolt cable** between mini1 and mini2 — **not two**.
   Two cables both in `bridge0` re-creates the broadcast-storm loop. If you must have
   redundancy, enable STP on `bridge0` properly first.
2. `bridge0` re-acquires link; `192.168.5.1` (mini1) / `192.168.5.2` (mini2) come back
   (verify: `ifconfig bridge0`). `launch.sh` detects the machine off these IPs.
3. `./launch.sh start` from this dir on either mini.
4. `netbiosd` is disabled on both minis (unrelated cleanup, 2026-05-14) — leave it disabled;
   the ring does not need it.

## Current State (2026-02-22)

The distributed ring is **working** with:
- **Model**: `mlx-community/Qwen3-14B-4bit`
- **Devices**: 2x Mac Mini M4 16GB (mini2=master 192.168.5.2, mini1=worker 192.168.5.1)
- **Parallelism**: Tensor parallelism via `model.shard(group)` — NOT pipeline
- **Backend**: `ring` over Thunderbolt Bridge (TCP)
- **IPC**: Unix domain socket `/tmp/mlx_ring.sock`
- **Performance**: ~12 tok/s generation, 15-65 tok/s prompt eval
- **Repo**: `/Users/mini2/Movies/mlx_distributed_ring_inference_v2` (mini2), `/Users/mini1/Movies/mlx_distributed_ring_inference_v2` (mini1)

## Key Technical Details

- MLX 0.30.5, mlx-lm 0.30.7, Python 3.14
- `hosts.json` must use `"127.0.0.1"` (not `"localhost"`) for local node — MLX does strict string match
- `--cwd` in mlx.launch must point to **mini1's** path (`/Users/mini1/Movies/mlx_distributed_ring_inference_v2`) — mini2 inherits CWD from parent process, mini1 needs the explicit cd
- Qwen3 only supports `model.shard(group)` (tensor parallelism). `model.model.pipeline(group)` is only for DeepSeek V3 and Ministral models
- `python-dotenv` must be installed on both machines for `.env` loading
- `uvicorn`, `fastapi`, `pydantic` installed via `pip3 install --break-system-packages`

## Next Steps: Enable RDMA over Thunderbolt 5

### What needs to happen

1. **Reboot both Mac Minis into Recovery Mode** (hold power button on startup)
2. **Enable RDMA** via Terminal in Recovery Mode (macOS 26.2+ required)
3. **Reboot both machines normally**
4. **Update `launch.sh`** to use `jaccl` backend instead of `ring`:
   ```bash
   # Change this line:
   mlx.launch --hostfile hosts.json --backend ring --verbose --cwd /Users/mini1/Movies/mlx_distributed_ring_inference_v2 python3 server.py
   # To:
   mlx.launch --hostfile hosts.json --backend jaccl --verbose --cwd /Users/mini1/Movies/mlx_distributed_ring_inference_v2 python3 server.py
   ```
5. **Update `hosts.json`** in launch.sh to include `rdma` fields:
   ```json
   [
       {"ssh": "127.0.0.1", "ips": ["192.168.5.2"], "rdma": [null, "RDMA_DEVICE_PATH"]},
       {"ssh": "mini1@192.168.5.1", "ips": ["192.168.5.1"], "rdma": ["RDMA_DEVICE_PATH", null]}
   ]
   ```
   - The `null` entry is for self (each host's own position in the list)
   - `RDMA_DEVICE_PATH` needs to be discovered after RDMA is enabled — check system logs or MLX docs for the actual device identifiers
6. **Test** with `./launch.sh start` and compare tok/s against current TCP baseline (~12 tok/s gen)

### RDMA Notes
- Thunderbolt 5 RDMA provides ~80 Gb/s (~10 GB/s) bypassing TCP/IP stack
- MLX uses `jaccl` backend with InfiniBand Verbs (IBV) internally
- Mac Studio caveat: the TB port adjacent to the Ethernet port may NOT support RDMA
- The `jaccl` launcher sets `MLX_JACCL_COORDINATOR` and `MLX_IBV_DEVICES` env vars automatically

### How to discover RDMA device paths after enabling
- Check `networksetup -listallhardwareports` for Thunderbolt interfaces
- Look at MLX source: `/opt/homebrew/lib/python3.14/site-packages/mlx/_distributed_utils/launch.py` function `launch_jaccl()` for expected format
- Check MLX GitHub issues/docs for examples of `rdma` field values

## Scaling Considerations
- MLX splits work **equally** across all ranks — no heterogeneous support
- All devices must have equal RAM
- To add a 3rd Mac Mini: add entry to hosts.json, set up SSH, install deps, download model
- For mixed-RAM setups: use [Exo](https://github.com/exo-explore/exo) instead
