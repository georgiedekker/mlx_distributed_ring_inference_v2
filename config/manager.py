"""
Configuration manager for MLX Distributed Inference.

This module provides centralized configuration management with:
- Environment variable loading from .env files
- Type-safe configuration dataclasses
- Validation with helpful error messages
- Graceful fallback to defaults
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass
class ModelConfig:
    """Model configuration settings."""

    repo: str = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit"
    cache_dir: str = ""  # Empty means use HuggingFace default

    def __post_init__(self):
        """Validate model configuration."""
        if not self.repo:
            raise ConfigValidationError("MODEL_REPO cannot be empty")

        # Expand ${USER} in cache_dir
        if self.cache_dir and "${USER}" in self.cache_dir:
            username = os.getenv("USER", "user")
            self.cache_dir = self.cache_dir.replace("${USER}", username)

        logger.debug(f"Model config: repo={self.repo}, cache_dir={self.cache_dir or 'default'}")


@dataclass
class NetworkConfig:
    """Network configuration settings."""

    api_host: str = "192.168.5.1"
    api_port: int = 8100
    worker_hosts: List[str] = field(default_factory=lambda: ["192.168.5.2"])
    worker_ssh: List[str] = field(default_factory=lambda: ["mini2@192.168.5.2"])

    def __post_init__(self):
        """Validate network configuration."""
        # Validate port number
        if not (1 <= self.api_port <= 65535):
            raise ConfigValidationError(
                f"API_PORT must be between 1 and 65535, got {self.api_port}"
            )

        # Validate host format (basic check)
        if not self.api_host:
            raise ConfigValidationError("API_HOST cannot be empty")

        if not self.worker_hosts:
            raise ConfigValidationError("WORKER_HOSTS cannot be empty")

        if not self.worker_ssh:
            raise ConfigValidationError("WORKER_SSH cannot be empty")

        # Ensure worker_hosts and worker_ssh have same length
        if len(self.worker_hosts) != len(self.worker_ssh):
            raise ConfigValidationError(
                f"WORKER_HOSTS and WORKER_SSH must have same number of entries. "
                f"Got {len(self.worker_hosts)} hosts and {len(self.worker_ssh)} SSH entries"
            )

        logger.debug(
            f"Network config: api={self.api_host}:{self.api_port}, workers={len(self.worker_hosts)}"
        )


@dataclass
class KVCacheConfig:
    """KV-Cache configuration settings."""

    max_size: Optional[int] = None  # None means no limit
    reserved_memory_mb: int = 2048
    max_sequence_length: int = 4096

    def __post_init__(self):
        """Validate KV-cache configuration."""
        if self.max_size is not None and self.max_size < 1:
            raise ConfigValidationError(
                f"KV_CACHE_MAX_SIZE must be positive or None, got {self.max_size}"
            )

        if self.reserved_memory_mb < 0:
            raise ConfigValidationError(
                f"KV_CACHE_RESERVED_MEMORY_MB must be non-negative, got {self.reserved_memory_mb}"
            )

        if self.max_sequence_length < 1:
            raise ConfigValidationError(
                f"MAX_SEQUENCE_LENGTH must be positive, got {self.max_sequence_length}"
            )

        logger.debug(
            f"KV-Cache config: max_size={self.max_size}, "
            f"reserved_memory={self.reserved_memory_mb}MB, "
            f"max_seq_len={self.max_sequence_length}"
        )


@dataclass
class PerformanceConfig:
    """Performance tuning settings."""

    max_prompt_len_bytes: int = 4096
    request_timeout_seconds: int = 120
    poll_interval_seconds: float = 0.1
    default_max_tokens: int = 50

    def __post_init__(self):
        """Validate performance configuration."""
        if self.max_prompt_len_bytes < 1:
            raise ConfigValidationError(
                f"MAX_PROMPT_LEN_BYTES must be positive, got {self.max_prompt_len_bytes}"
            )

        if self.request_timeout_seconds < 1:
            raise ConfigValidationError(
                f"REQUEST_TIMEOUT_SECONDS must be positive, got {self.request_timeout_seconds}"
            )

        if not (0 < self.poll_interval_seconds <= 10):
            raise ConfigValidationError(
                f"POLL_INTERVAL_SECONDS must be between 0 and 10, got {self.poll_interval_seconds}"
            )

        if self.default_max_tokens < 1:
            raise ConfigValidationError(
                f"DEFAULT_MAX_TOKENS must be positive, got {self.default_max_tokens}"
            )

        logger.debug(
            f"Performance config: max_prompt={self.max_prompt_len_bytes}B, "
            f"timeout={self.request_timeout_seconds}s, "
            f"poll={self.poll_interval_seconds}s"
        )


@dataclass
class DistributedConfig:
    """Distributed computing settings."""

    num_devices: int = 2
    backend: str = "ring"

    def __post_init__(self):
        """Validate distributed configuration."""
        if self.num_devices < 1:
            raise ConfigValidationError(f"NUM_DEVICES must be positive, got {self.num_devices}")

        valid_backends = ["ring", "nccl", "gloo"]
        if self.backend not in valid_backends:
            logger.warning(
                f"DISTRIBUTED_BACKEND '{self.backend}' not in known backends {valid_backends}. "
                f"Proceeding anyway..."
            )

        logger.debug(f"Distributed config: {self.num_devices} devices, backend={self.backend}")


@dataclass
class FilePathsConfig:
    """File paths for IPC and logging."""

    request_file_path: str = "/tmp/mlx_request.json"
    response_file_path: str = "/tmp/mlx_response.json"
    server_log_path: str = "server.log"
    api_log_path: str = "api.log"

    def __post_init__(self):
        """Validate file paths."""
        # Check that request and response files are different
        if self.request_file_path == self.response_file_path:
            raise ConfigValidationError(
                "REQUEST_FILE_PATH and RESPONSE_FILE_PATH must be different"
            )

        # Ensure all paths are non-empty
        for name, path in [
            ("REQUEST_FILE_PATH", self.request_file_path),
            ("RESPONSE_FILE_PATH", self.response_file_path),
            ("SERVER_LOG_PATH", self.server_log_path),
            ("API_LOG_PATH", self.api_log_path),
        ]:
            if not path:
                raise ConfigValidationError(f"{name} cannot be empty")

        logger.debug(
            f"File paths: request={self.request_file_path}, response={self.response_file_path}"
        )


@dataclass
class SystemConfig:
    """System-level configuration."""

    file_descriptor_soft_limit: int = 2048
    file_descriptor_hard_limit: int = 4096
    model_load_wait_seconds: int = 15
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate system configuration."""
        if self.file_descriptor_soft_limit < 1:
            raise ConfigValidationError(
                f"FILE_DESCRIPTOR_SOFT_LIMIT must be positive, got {self.file_descriptor_soft_limit}"
            )

        if self.file_descriptor_hard_limit < self.file_descriptor_soft_limit:
            raise ConfigValidationError(
                f"FILE_DESCRIPTOR_HARD_LIMIT ({self.file_descriptor_hard_limit}) must be >= "
                f"FILE_DESCRIPTOR_SOFT_LIMIT ({self.file_descriptor_soft_limit})"
            )

        if self.model_load_wait_seconds < 0:
            raise ConfigValidationError(
                f"MODEL_LOAD_WAIT_SECONDS must be non-negative, got {self.model_load_wait_seconds}"
            )

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigValidationError(
                f"LOG_LEVEL must be one of {valid_log_levels}, got {self.log_level}"
            )
        self.log_level = self.log_level.upper()

        logger.debug(
            f"System config: fd_limits=({self.file_descriptor_soft_limit}, {self.file_descriptor_hard_limit}), "
            f"log_level={self.log_level}"
        )


@dataclass
class Config:
    """Main configuration object containing all settings."""

    model: ModelConfig
    network: NetworkConfig
    kv_cache: KVCacheConfig
    performance: PerformanceConfig
    distributed: DistributedConfig
    file_paths: FilePathsConfig
    system: SystemConfig

    @classmethod
    def from_env(cls) -> "Config":
        """
        Create configuration from environment variables.

        Returns:
            Config: Configuration object with all settings

        Raises:
            ConfigValidationError: If any configuration value is invalid
        """

        def get_env_int(key: str, default: int) -> int:
            """Get integer from environment with validation."""
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                raise ConfigValidationError(f"{key} must be an integer, got '{value}'")

        def get_env_float(key: str, default: float) -> float:
            """Get float from environment with validation."""
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                raise ConfigValidationError(f"{key} must be a number, got '{value}'")

        def get_env_list(key: str, default: List[str]) -> List[str]:
            """Get comma-separated list from environment."""
            value = os.getenv(key)
            if value is None:
                return default
            # Split by comma and strip whitespace
            return [item.strip() for item in value.split(",") if item.strip()]

        def get_env_optional_int(key: str, default: Optional[int]) -> Optional[int]:
            """Get optional integer from environment."""
            value = os.getenv(key)
            if value is None or value == "":
                return default
            try:
                return int(value)
            except ValueError:
                raise ConfigValidationError(f"{key} must be an integer or empty, got '{value}'")

        # Create configuration sections
        model_config = ModelConfig(
            repo=os.getenv("MODEL_REPO", "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit"),
            cache_dir=os.getenv("MODEL_CACHE_DIR", "/Users/${USER}/.cache/huggingface/hub"),
        )

        network_config = NetworkConfig(
            api_host=os.getenv("API_HOST", "192.168.5.1"),
            api_port=get_env_int("API_PORT", 8100),
            worker_hosts=get_env_list("WORKER_HOSTS", ["192.168.5.2"]),
            worker_ssh=get_env_list("WORKER_SSH", ["mini2@192.168.5.2"]),
        )

        kv_cache_config = KVCacheConfig(
            max_size=get_env_optional_int("KV_CACHE_MAX_SIZE", None),
            reserved_memory_mb=get_env_int("KV_CACHE_RESERVED_MEMORY_MB", 2048),
            max_sequence_length=get_env_int("MAX_SEQUENCE_LENGTH", 4096),
        )

        performance_config = PerformanceConfig(
            max_prompt_len_bytes=get_env_int("MAX_PROMPT_LEN_BYTES", 4096),
            request_timeout_seconds=get_env_int("REQUEST_TIMEOUT_SECONDS", 120),
            poll_interval_seconds=get_env_float("POLL_INTERVAL_SECONDS", 0.1),
            default_max_tokens=get_env_int("DEFAULT_MAX_TOKENS", 50),
        )

        distributed_config = DistributedConfig(
            num_devices=get_env_int("NUM_DEVICES", 2),
            backend=os.getenv("DISTRIBUTED_BACKEND", "ring"),
        )

        file_paths_config = FilePathsConfig(
            request_file_path=os.getenv("REQUEST_FILE_PATH", "/tmp/mlx_request.json"),
            response_file_path=os.getenv("RESPONSE_FILE_PATH", "/tmp/mlx_response.json"),
            server_log_path=os.getenv("SERVER_LOG_PATH", "server.log"),
            api_log_path=os.getenv("API_LOG_PATH", "api.log"),
        )

        system_config = SystemConfig(
            file_descriptor_soft_limit=get_env_int("FILE_DESCRIPTOR_SOFT_LIMIT", 2048),
            file_descriptor_hard_limit=get_env_int("FILE_DESCRIPTOR_HARD_LIMIT", 4096),
            model_load_wait_seconds=get_env_int("MODEL_LOAD_WAIT_SECONDS", 15),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

        return cls(
            model=model_config,
            network=network_config,
            kv_cache=kv_cache_config,
            performance=performance_config,
            distributed=distributed_config,
            file_paths=file_paths_config,
            system=system_config,
        )


def load_config(env_file: Optional[str] = None) -> Config:
    """
    Load configuration from environment variables and optional .env file.

    Args:
        env_file: Path to .env file. If None, looks for .env in current directory.
                 If .env doesn't exist, uses environment variables only.

    Returns:
        Config: Validated configuration object

    Raises:
        ConfigValidationError: If any configuration value is invalid

    Example:
        >>> config = load_config()
        >>> print(config.network.api_port)
        8100
        >>> print(config.model.repo)
        mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit
    """
    # Determine .env file path
    if env_file is None:
        env_file = ".env"

    env_path = Path(env_file)

    # Load .env file if it exists
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
            logger.info(f"Loaded configuration from {env_path}")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Install it with: pip install python-dotenv"
            )
            logger.info("Using environment variables only")
    else:
        logger.info(f"No {env_file} file found, using environment variables and defaults")

    # Create and validate configuration
    try:
        config = Config.from_env()
        logger.info("Configuration loaded successfully")
        return config
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise ConfigValidationError(f"Failed to load configuration: {e}") from e
