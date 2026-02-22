"""
Centralized configuration management for MLX Distributed Inference.
"""

from typing import Optional

from .manager import (
    Config,
    ModelConfig,
    NetworkConfig,
    KVCacheConfig,
    PerformanceConfig,
    DistributedConfig,
    FilePathsConfig,
    SystemConfig,
    load_config,
    ConfigValidationError,
)

_config: Optional[Config] = None


def get_config() -> Config:
    """Return the singleton Config, loading from env on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


__all__ = [
    "Config",
    "ModelConfig",
    "NetworkConfig",
    "KVCacheConfig",
    "PerformanceConfig",
    "DistributedConfig",
    "FilePathsConfig",
    "SystemConfig",
    "load_config",
    "get_config",
    "ConfigValidationError",
]
