"""
Enhanced support classes for distributed inference with MLX native caching.

This module provides an enhanced key/value cache implementation that supports
MLX's native caching features including offset tracking, cache management,
and serialization. The implementation addresses common issues like "IndexError:
list index out of range" by properly managing cache state and offsets.
"""

from typing import Optional, Union, List, Tuple, Dict, Any
import mlx.core as mx
import mlx.nn as nn
import pickle
import logging

logger = logging.getLogger(__name__)


class KVCache:
    """Enhanced key/value cache for attention layers with MLX native support.

    This cache implementation provides:
    - Proper offset tracking to avoid index errors
    - Cache truncation and management methods
    - Support for both list-based and tensor-based cache structures
    - Cache serialization/deserialization for persistence
    - Compatibility with MLX's make_kv_caches function
    - Support for both single-node and distributed cache scenarios
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        step_size: int = 1,
        offset: int = 0,
        keys: Optional[Union[mx.array, List[mx.array]]] = None,
        values: Optional[Union[mx.array, List[mx.array]]] = None,
    ) -> None:
        """Initialize KVCache with optional size limits and offset tracking.

        Args:
            max_size: Maximum cache size (None for unlimited)
            step_size: Step size for cache expansion (default: 1)
            offset: Current offset in the cache (default: 0)
            keys: Initial keys (optional)
            values: Initial values (optional)
        """
        self.max_size = max_size
        self.step_size = step_size
        self.offset = offset
        self.keys = keys
        self.values = values
        self._version = 0  # For cache invalidation tracking

    @property
    def size(self) -> int:
        """Get current cache size."""
        if self.keys is None:
            return 0
        if isinstance(self.keys, list):
            return sum(k.shape[2] if k is not None else 0 for k in self.keys)
        else:
            return self.keys.shape[2] if len(self.keys.shape) > 2 else 0

    @property
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return self.keys is None or self.values is None

    def append(
        self, new_keys: Union[mx.array, List[mx.array]], new_values: Union[mx.array, List[mx.array]]
    ) -> None:
        """Append new key-value pairs to the cache.

        Args:
            new_keys: New keys to append
            new_values: New values to append
        """
        if self.is_empty:
            self.keys = new_keys
            self.values = new_values
        else:
            if isinstance(self.keys, list) and isinstance(new_keys, list):
                # List-based cache (for multi-layer models)
                for i, (k, v) in enumerate(zip(new_keys, new_values)):
                    if k is not None and v is not None:
                        if i < len(self.keys) and self.keys[i] is not None:
                            self.keys[i] = mx.concatenate([self.keys[i], k], axis=2)
                            self.values[i] = mx.concatenate([self.values[i], v], axis=2)
                        else:
                            if i >= len(self.keys):
                                # Extend lists if needed
                                self.keys.extend([None] * (i - len(self.keys) + 1))
                                self.values.extend([None] * (i - len(self.values) + 1))
                            self.keys[i] = k
                            self.values[i] = v
            else:
                # Tensor-based cache
                if isinstance(new_keys, list):
                    new_keys = new_keys[0] if new_keys else None
                    new_values = new_values[0] if new_values else None

                if new_keys is not None and new_values is not None:
                    self.keys = mx.concatenate([self.keys, new_keys], axis=2)
                    self.values = mx.concatenate([self.values, new_values], axis=2)

        # Update offset
        if isinstance(new_keys, list):
            if new_keys and new_keys[0] is not None:
                self.offset += new_keys[0].shape[2]
        else:
            if new_keys is not None:
                self.offset += new_keys.shape[2]

        # Handle max_size constraint
        if self.max_size is not None and self.size > self.max_size:
            self._truncate_to_size(self.max_size)

        self._version += 1

    def truncate(self, length: int) -> None:
        """Truncate cache to specified length.

        Args:
            length: Target length for the cache
        """
        if self.is_empty or length <= 0:
            self.reset()
            return

        if isinstance(self.keys, list):
            # List-based cache
            for i in range(len(self.keys)):
                if self.keys[i] is not None:
                    current_size = self.keys[i].shape[2]
                    if length < current_size:
                        self.keys[i] = self.keys[i][:, :, :length, :]
                        self.values[i] = self.values[i][:, :, :length, :]
        else:
            # Tensor-based cache
            if self.keys is not None:
                current_size = self.keys.shape[2]
                if length < current_size:
                    self.keys = self.keys[:, :, :length, :]
                    self.values = self.values[:, :, :length, :]

        self.offset = min(self.offset, length)
        self._version += 1

    def _truncate_to_size(self, target_size: int) -> None:
        """Internal method to truncate cache to target size."""
        if target_size <= 0:
            self.reset()
            return

        # Calculate how much to remove from the beginning (FIFO)
        current_size = self.size
        if current_size <= target_size:
            return

        remove_count = current_size - target_size

        if isinstance(self.keys, list):
            # List-based cache
            for i in range(len(self.keys)):
                if self.keys[i] is not None:
                    self.keys[i] = self.keys[i][:, :, remove_count:, :]
                    self.values[i] = self.values[i][:, :, remove_count:, :]
        else:
            # Tensor-based cache
            if self.keys is not None:
                self.keys = self.keys[:, :, remove_count:, :]
                self.values = self.values[:, :, remove_count:, :]

        self.offset = max(0, self.offset - remove_count)
        self._version += 1

    def reset(self) -> None:
        """Reset cache to empty state."""
        self.keys = None
        self.values = None
        self.offset = 0
        self._version += 1

    def clone(self) -> "KVCache":
        """Create a deep copy of the cache."""
        new_cache = KVCache(max_size=self.max_size, step_size=self.step_size, offset=self.offset)

        if not self.is_empty:
            if isinstance(self.keys, list):
                new_cache.keys = [mx.array(k) if k is not None else None for k in self.keys]
                new_cache.values = [mx.array(v) if v is not None else None for v in self.values]
            else:
                new_cache.keys = mx.array(self.keys)
                new_cache.values = mx.array(self.values)

        new_cache._version = self._version
        return new_cache

    def get_state(self) -> Dict[str, Any]:
        """Get cache state for serialization."""
        state = {
            "max_size": self.max_size,
            "step_size": self.step_size,
            "offset": self.offset,
            "version": self._version,
            "keys": None,
            "values": None,
        }

        if not self.is_empty:
            if isinstance(self.keys, list):
                # Convert arrays to lists for serialization
                state["keys"] = [k.tolist() if k is not None else None for k in self.keys]
                state["values"] = [v.tolist() if v is not None else None for v in self.values]
                state["is_list_cache"] = True
            else:
                state["keys"] = self.keys.tolist()
                state["values"] = self.values.tolist()
                state["is_list_cache"] = False

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore cache state from serialization."""
        self.max_size = state.get("max_size")
        self.step_size = state.get("step_size", 1)
        self.offset = state.get("offset", 0)
        self._version = state.get("version", 0)

        if state.get("keys") is not None and state.get("values") is not None:
            if state.get("is_list_cache", False):
                # Restore list-based cache
                self.keys = [mx.array(k) if k is not None else None for k in state["keys"]]
                self.values = [mx.array(v) if v is not None else None for v in state["values"]]
            else:
                # Restore tensor-based cache
                self.keys = mx.array(state["keys"])
                self.values = mx.array(state["values"])
        else:
            self.keys = None
            self.values = None

    def save(self, path: str) -> None:
        """Save cache state to file."""
        try:
            with open(path, "wb") as f:
                pickle.dump(self.get_state(), f)
            logger.debug(f"Cache saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save cache to {path}: {e}")
            raise

    def load(self, path: str) -> None:
        """Load cache state from file."""
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.set_state(state)
            logger.debug(f"Cache loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load cache from {path}: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of cache."""
        if self.is_empty:
            return f"KVCache(empty, offset={self.offset})"

        size = self.size
        cache_type = "list" if isinstance(self.keys, list) else "tensor"
        return f"KVCache({cache_type}, size={size}, offset={self.offset}, max_size={self.max_size})"


class IdentityBlock(nn.Module):
    """A no‑op neural network block.

    This module simply returns its input. It accepts optional mask and
    cache parameters for API compatibility with more complex blocks.
    """

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        return x


# Utility functions for cache creation and management


def make_kv_caches(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
    step_size: int = 1,
    dtype: mx.Dtype = mx.float16,
) -> List[Optional[KVCache]]:
    """Create KV caches compatible with MLX's native caching.

    This function creates a list of KVCache instances for each layer in the model,
    compatible with MLX's make_kv_caches function but with enhanced features.

    Args:
        model: The model to create caches for
        max_kv_size: Maximum cache size per layer (None for unlimited)
        step_size: Step size for cache expansion
        dtype: Data type for cache tensors

    Returns:
        List of KVCache instances, one per layer
    """
    caches = []

    # Determine number of layers
    if hasattr(model, "layers"):
        num_layers = len(model.layers)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    else:
        # Fallback: try to count transformer layers
        num_layers = 0
        for name, _ in model.named_modules():
            if "layer" in name.lower() and "transformer" in name.lower():
                try:
                    layer_num = int(name.split(".")[-1])
                    num_layers = max(num_layers, layer_num + 1)
                except:
                    pass

        if num_layers == 0:
            logger.warning("Could not determine number of layers, defaulting to 32")
            num_layers = 32

    # Create cache for each layer
    for i in range(num_layers):
        cache = KVCache(max_size=max_kv_size, step_size=step_size, offset=0)
        caches.append(cache)

    logger.debug(f"Created {num_layers} KV caches with max_size={max_kv_size}")
    return caches


def create_rotating_cache(
    model: nn.Module, max_kv_size: int, step_size: int = 1, dtype: mx.Dtype = mx.float16
) -> List[Optional[KVCache]]:
    """Create rotating KV caches with fixed size limit.

    This creates caches that automatically truncate old entries when they
    exceed the maximum size, implementing a FIFO rotation strategy.

    Args:
        model: The model to create caches for
        max_kv_size: Maximum cache size (required for rotating cache)
        step_size: Step size for cache expansion
        dtype: Data type for cache tensors

    Returns:
        List of rotating KVCache instances
    """
    if max_kv_size <= 0:
        raise ValueError("max_kv_size must be positive for rotating cache")

    return make_kv_caches(model=model, max_kv_size=max_kv_size, step_size=step_size, dtype=dtype)


def truncate_caches_to_length(
    caches: List[Optional[KVCache]], length: int
) -> List[Optional[KVCache]]:
    """Truncate all caches to specified length.

    This is useful for implementing cache reuse where you want to truncate
    caches to a common prefix length.

    Args:
        caches: List of cache instances
        length: Target length for truncation

    Returns:
        List of truncated caches
    """
    if not caches:
        return caches

    truncated_caches = []
    for cache in caches:
        if cache is not None:
            new_cache = cache.clone()
            new_cache.truncate(length)
            truncated_caches.append(new_cache)
        else:
            truncated_caches.append(None)

    return truncated_caches


def extract_cache_from_generate_step(cache_history) -> List[Optional[KVCache]]:
    """Extract KVCache instances from MLX generate_step cache history.

    This function converts the cache history returned by MLX's generate_step
    into our enhanced KVCache format for better management.

    Args:
        cache_history: Cache history from generate_step (typically tuple of keys/values)

    Returns:
        List of KVCache instances
    """
    if cache_history is None:
        return []

    caches = []

    if isinstance(cache_history, tuple) and len(cache_history) == 2:
        keys, values = cache_history

        if isinstance(keys, list) and isinstance(values, list):
            # Multi-layer cache
            for k, v in zip(keys, values):
                if k is not None and v is not None:
                    cache = KVCache()
                    cache.keys = k
                    cache.values = v
                    cache.offset = k.shape[2] if len(k.shape) > 2 else 0
                    caches.append(cache)
                else:
                    caches.append(None)
        else:
            # Single-layer cache
            if keys is not None and values is not None:
                cache = KVCache()
                cache.keys = keys
                cache.values = values
                cache.offset = keys.shape[2] if len(keys.shape) > 2 else 0
                caches.append(cache)

    return caches


def convert_to_mlx_cache_format(
    caches: List[Optional[KVCache]],
) -> Tuple[List[Optional[mx.array]], List[Optional[mx.array]]]:
    """Convert KVCache instances back to MLX's expected cache format.

    This function converts our enhanced KVCache instances back to the tuple
    format expected by MLX's generate_step and other functions.

    Args:
        caches: List of KVCache instances

    Returns:
        Tuple of (keys_list, values_list) compatible with MLX
    """
    if not caches:
        return [], []

    keys_list = []
    values_list = []

    for cache in caches:
        if cache is not None and not cache.is_empty:
            if isinstance(cache.keys, list):
                # Multi-layer cache - take first layer
                keys_list.append(cache.keys[0] if cache.keys else None)
                values_list.append(cache.values[0] if cache.values else None)
            else:
                keys_list.append(cache.keys)
                values_list.append(cache.values)
        else:
            keys_list.append(None)
            values_list.append(None)

    return keys_list, values_list


def save_caches(caches: List[Optional[KVCache]], base_path: str) -> None:
    """Save all caches to disk with a common base path.

    Args:
        caches: List of cache instances to save
        base_path: Base path for saving (will append layer indices)
    """
    for i, cache in enumerate(caches):
        if cache is not None:
            cache_path = f"{base_path}_layer_{i}.cache"
            cache.save(cache_path)


def load_caches(base_path: str, num_layers: int) -> List[Optional[KVCache]]:
    """Load caches from disk.

    Args:
        base_path: Base path for loading (will append layer indices)
        num_layers: Number of layers to load

    Returns:
        List of loaded cache instances
    """
    caches = []
    for i in range(num_layers):
        cache_path = f"{base_path}_layer_{i}.cache"
        try:
            cache = KVCache()
            cache.load(cache_path)
            caches.append(cache)
        except (FileNotFoundError, Exception) as e:
            logger.debug(f"Could not load cache for layer {i}: {e}")
            caches.append(None)

    return caches


def get_cache_stats(caches: List[Optional[KVCache]]) -> Dict[str, Any]:
    """Get statistics about cache usage.

    Args:
        caches: List of cache instances

    Returns:
        Dictionary with cache statistics
    """
    stats = {
        "total_layers": len(caches),
        "active_layers": 0,
        "total_size": 0,
        "max_size": 0,
        "min_size": float("inf"),
        "total_offset": 0,
        "memory_usage_mb": 0.0,
    }

    for cache in caches:
        if cache is not None and not cache.is_empty:
            stats["active_layers"] += 1
            cache_size = cache.size
            stats["total_size"] += cache_size
            stats["max_size"] = max(stats["max_size"], cache_size)
            stats["min_size"] = min(stats["min_size"], cache_size)
            stats["total_offset"] += cache.offset

            # Rough memory estimation (keys + values, assuming float16)
            if isinstance(cache.keys, list):
                for k, v in zip(cache.keys or [], cache.values or []):
                    if k is not None and v is not None:
                        stats["memory_usage_mb"] += (k.nbytes + v.nbytes) / (1024 * 1024)
            else:
                if cache.keys is not None and cache.values is not None:
                    stats["memory_usage_mb"] += (cache.keys.nbytes + cache.values.nbytes) / (
                        1024 * 1024
                    )

    if stats["min_size"] == float("inf"):
        stats["min_size"] = 0

    if stats["active_layers"] > 0:
        stats["avg_size"] = stats["total_size"] / stats["active_layers"]
        stats["avg_offset"] = stats["total_offset"] / stats["active_layers"]
    else:
        stats["avg_size"] = 0
        stats["avg_offset"] = 0

    return stats
