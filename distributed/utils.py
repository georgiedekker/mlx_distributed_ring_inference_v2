"""
Distributed utilities for model loading and communication.

This module provides functions for:
- Broadcasting prompts across distributed ranks
- Loading and sharding models for distributed inference
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.core.distributed as dist
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model, load_tokenizer

logger = logging.getLogger(__name__)


def broadcast_prompt(
    prompt: Optional[str], rank: int, max_tokens: int, group, max_len: int = 1000
) -> Tuple[bool, str, int]:
    """
    Broadcast prompt and parameters from rank 0 to all other ranks.

    This function implements a two-phase broadcast:
    1. First, it broadcasts whether there's work to do (has_work flag)
    2. If there's work, it broadcasts the prompt length, max_tokens, and prompt text

    The prompt is encoded as bytes, padded to a fixed size, converted to floats,
    and broadcasted using mx.distributed.all_sum. This ensures all ranks receive
    the same prompt and parameters.

    Args:
        prompt: The prompt to broadcast. On rank 0, this should be the actual prompt
               string (or None if no work). On other ranks, this parameter is ignored.
        rank: The current rank in the distributed group.
        max_tokens: Maximum tokens to generate. Only used on rank 0, broadcasted to
                   all other ranks.
        group: The distributed group for communication.
        max_len: Maximum length for the prompt buffer in bytes (default: 1000).
                This is the buffer size used for broadcasting.

    Returns:
        A tuple of (has_work, prompt, max_tokens) where:
            - has_work: Boolean indicating whether there's work to process
            - prompt: The broadcasted prompt string (same on all ranks)
            - max_tokens: The broadcasted max_tokens value (same on all ranks)

    Example:
        >>> # On rank 0
        >>> has_work, prompt, max_tokens = broadcast_prompt(
        ...     "Hello world", rank=0, max_tokens=50, group=group
        ... )
        >>> # On rank 1+
        >>> has_work, prompt, max_tokens = broadcast_prompt(
        ...     None, rank=1, max_tokens=0, group=group
        ... )
        >>> # Both ranks now have the same prompt and max_tokens
    """
    # Broadcast whether we have work
    if rank == 0:
        has_work = mx.array([1.0 if prompt else 0.0])
    else:
        has_work = mx.array([0.0])

    has_work = dist.all_sum(has_work, group=group)
    mx.eval(has_work)

    if has_work.item() > 0:
        # Broadcast prompt length
        if rank == 0:
            prompt_bytes = prompt.encode("utf-8")
            prompt_len = len(prompt_bytes)
            params = mx.array([float(prompt_len), float(max_tokens)])
        else:
            params = mx.array([0.0, 0.0])

        params = dist.all_sum(params, group=group)
        mx.eval(params)

        prompt_len = int(params[0].item())
        max_tokens = int(params[1].item())

        # Broadcast prompt text
        if rank == 0:
            # Pad prompt to fixed size for simplicity
            padded = prompt_bytes[:max_len].ljust(max_len, b"\0")
            prompt_array = mx.array([float(b) for b in padded])
        else:
            prompt_array = mx.zeros(max_len)

        prompt_array = dist.all_sum(prompt_array, group=group)
        mx.eval(prompt_array)

        # Reconstruct prompt
        prompt_bytes = bytes([int(v.item()) for v in prompt_array[:prompt_len]])
        prompt = prompt_bytes.decode("utf-8")

        return True, prompt, max_tokens
    else:
        return False, "", 0


def shard_and_load(repo: str, config: Any = None, adapter_path: str = "") -> Tuple:
    """
    Load and shard a model for distributed inference using pipeline parallelism.

    This function:
    1. Loads model metadata and configuration
    2. Initializes the distributed group
    3. Applies pipeline sharding to distribute layers across ranks
    4. Determines which weight files each rank needs based on its parameters
    5. Downloads only the necessary weight files for this rank
    6. Loads the tokenizer
    7. Loads and shards the model with weights
    8. Synchronizes all ranks

    The function uses MLX's pipeline() method to automatically distribute model
    layers across ranks. Each rank only loads the weights it needs for its layers,
    reducing memory usage.

    Args:
        repo: Model repository name on HuggingFace Hub (e.g., "mlx-community/...")
        config: Optional configuration object with model settings. If None, uses defaults.
        adapter_path: Optional path to LoRA adapter weights (currently unused,
                     reserved for future functionality).

    Returns:
        A tuple of (model, tokenizer, rank, world_size, group) where:
            - model: The loaded and sharded model
            - tokenizer: The loaded tokenizer
            - rank: This process's rank in the distributed group
            - world_size: Total number of ranks
            - group: The distributed group object

    Raises:
        FileNotFoundError: If the model cache directory or snapshots not found
        ValueError: If model loading or sharding fails
        RuntimeError: If distributed initialization or synchronization fails

    Example:
        >>> from config import get_config
        >>> config = get_config()
        >>> model, tokenizer, rank, world_size, group = shard_and_load(
        ...     "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit",
        ...     config=config
        ... )
        >>> logger.info(f"Rank {rank}/{world_size} loaded successfully")

    Note:
        This function expects to be called from within an MLX distributed context.
        All ranks must call this function collectively.
    """
    # Use local cached model instead of downloading
    username = os.getenv("USER", "mini1")  # Get current user (mini1 or mini2)

    if config and hasattr(config, "model") and hasattr(config.model, "cache_base_path"):
        cache_path_template = config.model.cache_base_path.format(username=username)
        model_path = Path(cache_path_template) / f"models--{repo.replace('/', '--')}"
    else:
        # Fallback to default path
        model_path = Path(
            f"/Users/{username}/.cache/huggingface/hub/models--{repo.replace('/', '--')}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model cache directory not found: {model_path}. Please download the model first."
        )

    # Find the snapshot directory
    snapshots = list(model_path.glob("snapshots/*"))
    if not snapshots:
        raise FileNotFoundError(f"No model snapshots found in {model_path}")
    model_path = snapshots[0]  # Use first (and likely only) snapshot

    logger.info(f"Using cached model from: {model_path}")

    try:
        # Load and shard model
        model, model_config = load_model(model_path, lazy=True, strict=False)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    try:
        group = dist.init()
        rank = group.rank()
        world_size = group.size()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed group: {e}")

    logger.info(
        f"[Rank {rank}] Before sharding: model has {len(list(model.parameters()))} parameters"
    )

    # Apply pipeline sharding - this MUST distribute layers evenly
    try:
        model.model.pipeline(group)
    except Exception as e:
        raise RuntimeError(f"Failed to apply pipeline sharding: {e}")

    logger.info(
        f"[Rank {rank}] After sharding: model has {len(list(model.parameters()))} parameters"
    )

    # Figure out which files we need
    try:
        with open(model_path / "model.safetensors.index.json", "r") as fid:
            weight_index = json.load(fid)["weight_map"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to load weight index: {e}")

    local_files = set()
    for k, _ in tree_flatten(model.parameters()):
        if k in weight_index:
            local_files.add(weight_index[k])

    # Download weights
    try:
        _download(repo, allow_patterns=list(local_files))
    except Exception as e:
        logger.warning(f"Failed to download additional weights: {e}")

    # Load tokenizer
    try:
        tokenizer = load_tokenizer(
            model_path,
            {"trust_remote_code": True},
            eos_token_ids=model_config.get("eos_token_id", None),
        )
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer: {e}")

    # Load and shard the model with weights
    try:
        model, _ = load_model(model_path, lazy=True, strict=False)
        model.model.pipeline(group)
        mx.eval(model.parameters())
    except Exception as e:
        raise RuntimeError(f"Failed to load and shard model with weights: {e}")

    # Synchronize all processes
    try:
        mx.eval(dist.all_sum(mx.array(1.0), stream=mx.cpu))
    except Exception as e:
        raise RuntimeError(f"Failed to synchronize processes: {e}")

    return model, tokenizer, rank, world_size, group


def _download(repo: str, allow_patterns: list[str]) -> Path:
    """
    Download model files from HuggingFace Hub.

    Args:
        repo: Repository ID on HuggingFace Hub
        allow_patterns: List of file patterns to download

    Returns:
        Path to the downloaded model directory
    """
    return Path(snapshot_download(repo, allow_patterns=allow_patterns))
