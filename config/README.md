# Configuration System

This directory contains the centralized configuration management system for MLX Distributed Inference.

## Overview

The configuration system provides a type-safe, validated way to manage all settings, replacing hardcoded values throughout the codebase. It uses `.env` files for local configuration and gracefully falls back to sensible defaults.

## Features

- **Type-safe configuration** using Python dataclasses
- **Validation with helpful error messages** (e.g., port numbers, file paths)
- **Environment variable support** via `.env` files
- **Graceful fallback** to defaults when `.env` is missing
- **Template file** (`.env.example`) for easy setup

## Quick Start

### 1. Install Dependencies

```bash
pip install python-dotenv
```

### 2. Create Your Configuration

```bash
# Copy the example file
cp .env.example .env

# Edit with your settings
nano .env
```

### 3. Use in Your Code

```python
from config import load_config

# Load configuration (automatically finds .env)
config = load_config()

# Access settings
print(config.network.api_port)  # 8100
print(config.model.repo)        # mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit
```

## Configuration Structure

The configuration is organized into seven logical sections:

```python
@dataclass
class Config:
    model: ModelConfig          # Model settings
    network: NetworkConfig      # Network/API settings
    kv_cache: KVCacheConfig     # KV-cache settings
    performance: PerformanceConfig  # Performance tuning
    distributed: DistributedConfig  # Distributed computing
    file_paths: FilePathsConfig     # File paths for IPC
    system: SystemConfig        # System-level settings
```

### ModelConfig

Settings for the ML model:
- `repo`: HuggingFace model repository (default: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit`)
- `cache_dir`: Model cache directory (default: `/Users/${USER}/.cache/huggingface/hub`)

### NetworkConfig

Network and API settings:
- `api_host`: API server host (default: `192.168.5.1`)
- `api_port`: API server port (default: `8100`, validated: 1-65535)
- `worker_hosts`: List of worker host IPs
- `worker_ssh`: List of SSH connection strings for workers

### KVCacheConfig

KV-cache memory management:
- `max_size`: Maximum KV-cache size (default: `None` = no limit)
- `reserved_memory_mb`: Reserved memory in MB (default: `2048`)
- `max_sequence_length`: Maximum sequence length (default: `4096`)

### PerformanceConfig

Performance tuning parameters:
- `max_prompt_len_bytes`: Maximum prompt length for broadcast (default: `4096`)
- `request_timeout_seconds`: Request timeout (default: `120`)
- `poll_interval_seconds`: Polling interval (default: `0.1`, validated: 0-10)
- `default_max_tokens`: Default max tokens for generation (default: `50`)

### DistributedConfig

Distributed computing settings:
- `num_devices`: Number of distributed devices (default: `2`)
- `backend`: MLX distributed backend (default: `ring`)

### FilePathsConfig

File paths for inter-process communication:
- `request_file_path`: Request file path (default: `/tmp/mlx_request.json`)
- `response_file_path`: Response file path (default: `/tmp/mlx_response.json`)
- `server_log_path`: Server log file (default: `server.log`)
- `api_log_path`: API log file (default: `api.log`)

### SystemConfig

System-level configuration:
- `file_descriptor_soft_limit`: Soft FD limit (default: `2048`)
- `file_descriptor_hard_limit`: Hard FD limit (default: `4096`)
- `model_load_wait_seconds`: Model loading wait time (default: `15`)
- `log_level`: Logging level (default: `INFO`, validated against standard levels)

## Environment Variables

See `.env.example` in the project root for the complete list of supported environment variables with documentation.

### Key Variables

```bash
# Model
MODEL_REPO=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit
MODEL_CACHE_DIR=/Users/${USER}/.cache/huggingface/hub

# Network
API_HOST=192.168.5.1
API_PORT=8100
WORKER_HOSTS=192.168.5.2
WORKER_SSH=mini2@192.168.5.2

# Performance
MAX_PROMPT_LEN_BYTES=4096
REQUEST_TIMEOUT_SECONDS=120
POLL_INTERVAL_SECONDS=0.1

# System
LOG_LEVEL=INFO
```

## Validation

The configuration system validates all values and provides helpful error messages:

```python
# Example validation errors:
API_PORT=99999  # Error: API_PORT must be between 1 and 65535
NUM_DEVICES=0   # Error: NUM_DEVICES must be positive
LOG_LEVEL=BAD   # Error: LOG_LEVEL must be one of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
```

## Testing

Run the configuration test suite to verify everything works:

```bash
python3 test_config.py
```

This will test:
1. Default configuration loading
2. Environment variable overrides
3. Validation with invalid values
4. Loading from `.env` files

## API Reference

### `load_config(env_file: Optional[str] = None) -> Config`

Load configuration from environment variables and optional .env file.

**Parameters:**
- `env_file`: Path to .env file (default: `.env` in current directory)

**Returns:**
- Validated `Config` object

**Raises:**
- `ConfigValidationError`: If any configuration value is invalid

**Example:**
```python
from config import load_config, ConfigValidationError

try:
    config = load_config()
    print(f"API will run on {config.network.api_host}:{config.network.api_port}")
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
```

## Notes

- The `.env` file is gitignored by default to prevent committing sensitive settings
- If `python-dotenv` is not installed, the system falls back to OS environment variables
- All configuration values have sensible defaults
- The `${USER}` variable in paths is automatically expanded
