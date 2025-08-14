# MLX Distributed Ring Inference

Run large language models across multiple Mac devices using MLX's distributed ring backend over Thunderbolt networking. Split DeepSeek-Coder-V2-Lite (16B parameters) across 2-3 Macs for collaborative inference without needing a single powerful machine.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Complete Setup Guide](#complete-setup-guide)
  - [Step 1: Check System Requirements](#step-1-check-system-requirements)
  - [Step 2: Install Required Software](#step-2-install-required-software)
  - [Step 3: Connect Devices via Thunderbolt](#step-3-connect-devices-via-thunderbolt)
  - [Step 4: Configure Thunderbolt Networking](#step-4-configure-thunderbolt-networking)
  - [Step 5: Set Up SSH Access](#step-5-set-up-ssh-access)
  - [Step 6: Install the Project](#step-6-install-the-project)
  - [Step 7: Verify Setup](#step-7-verify-setup)
  - [Step 8: Run Distributed Inference](#step-8-run-distributed-inference)
- [Using the System](#using-the-system)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)

## Overview

This project distributes a large language model (DeepSeek-Coder-V2-Lite-Instruct) across multiple Mac computers connected via Thunderbolt, using MLX's ring communication backend. Each Mac handles a portion of the model layers, enabling inference on models too large for a single device's memory.

## Prerequisites

### Hardware Requirements

- 2-3 Mac computers with Apple Silicon (M1/M2/M3/M4)
- Minimum 16GB RAM per Mac
- Thunderbolt 3/4 cables for connection
- Tested on Mac mini M2 16GB and Mac Studio M4

### Software Requirements

- macOS Ventura (13.0) or later
- Python 3.11+
- MLX framework
- Admin access for network configuration

## Complete Setup Guide

### Step 1: Check System Requirements

On **each Mac**, open Terminal (found in Applications > Utilities) and run:

```bash
# Check macOS version (should be 13.0 or higher)
sw_vers -productVersion

# Check if you have Apple Silicon
uname -m
# Should output: arm64

# Check available memory (should be 16GB or more)
sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}'
```

### Step 2: Install Required Software

On **each Mac**, install the following:

#### 2.1 Install Homebrew (Package Manager)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Follow the instructions shown after installation to add Homebrew to PATH
# Usually involves running:
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# Verify installation
brew --version
```

#### 2.2 Install Python 3.11 or Later

```bash
# Install Python
brew install python@3.11

# Verify Python installation
python3 --version
# Should show Python 3.11.x or higher

# Install pip (Python package manager) if needed
python3 -m ensurepip --upgrade
```

#### 2.3 Install Git

```bash
# Install Git
brew install git

# Verify Git installation
git --version
```

#### 2.4 Install Project Dependencies

```bash
# Install all required packages
pip3 install -r requirements.txt

# Verify MLX installation
python3 -c "import mlx; print('MLX version:', mlx.__version__)"
```

### Step 3: Connect Devices via Thunderbolt

#### Physical Connection

1. **For 2 Macs:**
   - Connect them directly with a Thunderbolt cable
   - Use any Thunderbolt port on each Mac

2. **For 3 Macs:**
   - Option A: Daisy chain (Mac1 → Mac2 → Mac3)
   - Option B: Use a Thunderbolt hub

3. **Verify Cable Connection:**
   - You should hear a connection sound
   - The Thunderbolt icon should appear in System Settings

### Step 4: Configure Thunderbolt Networking

This creates a network between your Macs over the Thunderbolt cable.

#### On Mac #1 (Primary/Master):

1. Open **System Settings** → **Network**
2. Look for "Thunderbolt Bridge" in the list
   - If not visible, click "+" and add "Thunderbolt Bridge"
3. Click on "Thunderbolt Bridge" → "Details..."
4. Go to "TCP/IP" tab
5. Configure IPv4: **Manually**
6. IP Address: **192.168.5.1**
7. Subnet Mask: **255.255.255.0**
8. Click "OK" then "Apply"

#### On Mac #2:

1. Open **System Settings** → **Network**
2. Look for "Thunderbolt Bridge"
3. Click on "Thunderbolt Bridge" → "Details..."
4. Go to "TCP/IP" tab
5. Configure IPv4: **Manually**
6. IP Address: **192.168.5.2**
7. Subnet Mask: **255.255.255.0**
8. Click "OK" then "Apply"

#### On Mac #3 (if using 3 devices):

1. Same process as above, but use:
2. IP Address: **192.168.5.3**
3. Subnet Mask: **255.255.255.0**

#### Verify Network Connection:

On Mac #1, open Terminal and test:

```bash
# Test connection to Mac #2
ping -c 3 192.168.5.2
# Should see responses

# If using Mac #3
ping -c 3 192.168.5.3
# Should see responses
```

### Step 5: Set Up SSH Access

SSH allows the Macs to communicate and run commands on each other.

#### 5.1 Enable SSH on All Macs

On **each Mac**:

1. Open **System Settings** → **General** → **Sharing**
2. Turn ON **Remote Login**
3. Note the username shown (you'll need this)

#### 5.2 Set Up Host Names

On Mac #1, edit the hosts file to create easy names:

```bash
# Open the hosts file
sudo nano /etc/hosts

# Add these lines at the bottom:
192.168.5.1 mini1
192.168.5.2 mini2
192.168.5.3 m4

# Save with Ctrl+O, Enter, then exit with Ctrl+X
```

Do the same on Mac #2 and Mac #3.

#### 5.3 Create SSH Keys (Password-free Access)

On Mac #1, generate SSH keys:

```bash
# Generate SSH key (press Enter for all prompts)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Copy key to Mac #2 (replace 'username' with actual username from Mac #2)
ssh-copy-id -i ~/.ssh/id_ed25519 username@192.168.5.2

# If using Mac #3 (replace 'username' with actual username from Mac #3)
ssh-copy-id -i ~/.ssh/id_ed25519 username@192.168.5.3
```

#### 5.4 Test SSH Connection

```bash
# Test connection to Mac #2 (replace 'username')
ssh username@192.168.5.2 "echo 'SSH working on Mac 2'"

# Test connection to Mac #3 if applicable
ssh username@192.168.5.3 "echo 'SSH working on Mac 3'"
```

### Step 6: Install the Project

#### 6.1 Create Project Directory

On **ALL Macs**, create the same directory:

```bash
# Create shared directory
sudo mkdir -p /Users/Shared/mlx_distributed_ring_inference
sudo chmod 755 /Users/Shared/mlx_distributed_ring_inference
cd /Users/Shared
```

#### 6.2 Clone the Repository

On Mac #1 ONLY:

```bash
cd /Users/Shared
git clone https://github.com/georgiedekker/mlx_distributed_ring_inference.git
cd mlx_distributed_ring_inference
```

#### 6.3 Copy to Other Macs

From Mac #1, copy files to other Macs:

```bash
# Copy to Mac #2 (replace 'username')
scp -r /Users/Shared/mlx_distributed_ring_inference username@192.168.5.2:/Users/Shared/

# If using Mac #3 (replace 'username')
scp -r /Users/Shared/mlx_distributed_ring_inference username@192.168.5.3:/Users/Shared/
```

#### 6.4 Install Python Dependencies

On **ALL Macs**, install required packages:

```bash
cd /Users/Shared/mlx_distributed_ring_inference
pip3 install -r requirements.txt
```

#### 6.5 Make Scripts Executable

On Mac #1:

```bash
cd /Users/Shared/mlx_distributed_ring_inference
chmod +x launch.sh
```

### Step 7: Verify Setup

#### 7.1 Check Device Detection

On Mac #1:

```bash
cd /Users/Shared/mlx_distributed_ring_inference
./launch.sh detect
```

You should see:
- ✓ Found mini2 (192.168.5.2)
- ✓ Found m4 (192.168.5.3) [if using 3 devices]
- Generated hosts.json with 2-3 nodes

#### 7.2 Verify hosts.json

```bash
cat hosts.json
```

Should show something like:
```json
[
    {"ssh": "localhost", "ips": ["192.168.5.1"]},
    {"ssh": "username@192.168.5.2", "ips": ["192.168.5.2"]},
    {"ssh": "username@192.168.5.3", "ips": ["192.168.5.3"]}
]
```

### Step 8: Run Distributed Inference

#### 8.1 Start the System

On Mac #1:

```bash
cd /Users/Shared/mlx_distributed_ring_inference
./launch.sh start
```

You should see:
```
🚀 MLX Distributed Inference with DeepSeek
✓ Found mini2 (192.168.5.2)
✓ Synced server.py to mini2
✓ Distributed inference launched
✓ API server launched
✓ API endpoint: http://localhost:8100
```

#### 8.2 Check Status

```bash
./launch.sh status
```

Should show all servers running.

#### 8.3 Test the API

```bash
# Test health endpoint
curl http://localhost:8100/health

# Test chat completion
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello!"}],
    "max_tokens": 50
  }'
```

## Using the System

### Starting and Stopping

```bash
# Start the distributed system
./launch.sh start

# Stop everything
./launch.sh stop

# Restart everything
./launch.sh restart

# Check status
./launch.sh status
```

### Monitoring

```bash
# Watch server logs
tail -f server.log

# Watch API logs
tail -f api.log

# Monitor both
tail -f server.log api.log
```

### API Usage

The system provides an OpenAI-compatible API endpoint at `http://localhost:8100`.

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8100/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
)
print(response.json())
```

## Troubleshooting

### Common Issues and Solutions

#### "Connection refused" or "No route to host"

1. Check Thunderbolt cable is properly connected
2. Verify network settings:
   ```bash
   # On each Mac
   ifconfig | grep 192.168.5
   # Should show the assigned IP
   ```
3. Check firewall isn't blocking:
   - System Settings → Network → Firewall → Options
   - Add Terminal to allowed apps

#### "Permission denied" SSH errors

1. Verify SSH is enabled on all Macs
2. Check username is correct:
   ```bash
   whoami  # Shows your username
   ```
3. Regenerate SSH keys if needed

#### "Module not found" Python errors

1. Ensure you're using the right Python:
   ```bash
   which python3
   # Should be /opt/homebrew/bin/python3 or similar
   ```
2. Reinstall requirements:
   ```bash
   pip3 install -r requirements.txt --upgrade
   ```

#### Servers won't start

1. Check no other process is using port 8100:
   ```bash
   lsof -i :8100
   ```
2. Clean stop and restart:
   ```bash
   ./launch.sh stop
   sleep 5
   ./launch.sh start
   ```

#### Model download issues

The first run will download the model (several GB). Ensure:
- Stable internet connection
- Sufficient disk space (check with `df -h`)
- If download fails, delete partial files in `~/.cache/huggingface/`

### Getting Help

1. Check server logs: `cat server.log`
2. Check API logs: `cat api.log`
3. Verify all Macs can communicate: `./launch.sh detect`
4. File issues at: https://github.com/georgiedekker/mlx_distributed_ring_inference/issues

## Architecture

```
Mac #1 (192.168.5.1)       Mac #2 (192.168.5.2)       Mac #3 (192.168.5.3)
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Master Node    │◄──────►│  Worker Node    │◄──────►│  Worker Node    │
│  - API Server   │        │  - Model Layers │        │  - Model Layers │
│  - Model Layers │        │  - Ring Backend │        │  - Ring Backend │
│  - Coordinator  │        │                 │        │  (Optional)     │
└─────────────────┘        └─────────────────┘        └─────────────────┘
      Port 8100                Thunderbolt                Thunderbolt
```

## Key Features

- **Distributed Inference**: DeepSeek-Coder-V2-Lite-Instruct (16B) distributed across devices
- **Simple Setup**: One-command launch with automatic device detection
- **MLX Ring Backend**: Efficient communication over Thunderbolt using MLX's native distributed backend
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API calls
- **Conversation Caching**: Maintains context across requests for improved performance

## License

MIT License

## Project Structure

```
mlx_distributed_ring_inference/
├── launch.sh          # Main launcher script - manages distributed setup
├── server.py          # Distributed MLX inference server
├── api.py            # FastAPI server providing OpenAI-compatible endpoint
├── requirements.txt   # Python dependencies
├── LICENSE           # MIT License
└── README.md         # This file
```

## How It Works

1. **launch.sh**: Orchestrates the distributed setup
   - Creates hosts.json configuration for MLX
   - Syncs server.py to remote Macs
   - Launches MLX distributed process with ring backend
   - Starts API server for client requests

2. **server.py**: Handles distributed inference
   - Loads and shards model across devices using MLX pipeline parallelism
   - Each device handles ~30 layers of the 60-layer model
   - Communicates via MLX's ring backend over Thunderbolt
   - Processes requests from api.py via file-based IPC

3. **api.py**: Provides REST API
   - OpenAI-compatible chat completions endpoint
   - Communicates with server.py via JSON files in /tmp
   - Returns responses with token metrics

## Performance

- Model loading: ~15 seconds
- Inference speed: ~10-20 tokens/second (varies by prompt length)
- Memory usage: ~8-10GB per device
- Network: Thunderbolt provides ~20-40 Gbps bandwidth

## Acknowledgments

Built with [MLX](https://github.com/ml-explore/mlx) by Apple's machine learning research team.
Model: [DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit)