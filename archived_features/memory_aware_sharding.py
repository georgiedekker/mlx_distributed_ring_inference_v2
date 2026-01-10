"""
Memory-aware pipeline sharding for mixed-memory distributed clusters.

This module provides utilities for distributing model layers across ranks
based on available memory rather than equal layer splitting. It supports
heterogeneous memory configurations like mixing Mac Studio (192GB) with
Mac mini (16GB) machines.

Key features:
- Per-rank memory detection and configuration
- Layer size estimation for accurate memory distribution
- Memory-aware layer allocation algorithm
- Environment variable configuration for manual overrides
- Comprehensive logging of sharding decisions
- OOM prevention and warnings
"""

import os
import subprocess
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)


@dataclass
class MemoryInfo:
    """Memory information for a specific rank."""

    rank: int
    total_memory_mb: int
    available_memory_mb: int
    reserved_memory_mb: int = 1024  # Reserve 1GB by default
    usable_memory_mb: int = 0

    def __post_init__(self):
        self.usable_memory_mb = max(0, self.available_memory_mb - self.reserved_memory_mb)


@dataclass
class LayerEstimate:
    """Memory estimation for a model layer."""

    layer_idx: int
    parameters_mb: float
    activations_mb: float
    cache_mb: float
    total_mb: float

    def __post_init__(self):
        self.total_mb = self.parameters_mb + self.activations_mb + self.cache_mb


@dataclass
class ShardingPlan:
    """Complete sharding plan for memory-aware distribution."""

    rank: int
    start_layer: int
    end_layer: int
    num_layers: int
    estimated_memory_mb: float
    available_memory_mb: int
    memory_utilization: float = 0.0

    def __post_init__(self):
        self.memory_utilization = (
            self.estimated_memory_mb / self.available_memory_mb
            if self.available_memory_mb > 0
            else float("inf")
        )


def get_system_memory_mb() -> Optional[int]:
    """Get total system memory in MB for the current machine."""
    try:
        # macOS specific - use sysctl
        result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
        if result.returncode == 0:
            mem_bytes = int(result.stdout.split(":")[1].strip())
            mem_mb = mem_bytes // (1024 * 1024)
            return mem_mb
    except Exception as e:
        logger.warning(f"Could not determine system memory via sysctl: {e}")

    try:
        # Fallback: try vm_stat for available memory
        result = subprocess.run(["vm_stat"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            page_size = 4096  # Default page size
            free_pages = 0

            for line in lines:
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                elif "Pages free:" in line:
                    free_pages += int(line.split()[-1].rstrip("."))
                elif "Pages speculative:" in line:
                    free_pages += int(line.split()[-1].rstrip("."))

            if free_pages > 0:
                free_mb = (free_pages * page_size) // (1024 * 1024)
                # Estimate total as free + some used (rough approximation)
                total_mb = free_mb * 4  # Assume 25% free
                return total_mb
    except Exception as e:
        logger.warning(f"Could not determine memory via vm_stat: {e}")

    # Final fallback - use commonly available values for macOS
    logger.warning("Using fallback memory detection")
    return None


def get_available_memory_mb() -> int:
    """Get currently available memory in MB."""
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            page_size = 4096
            free_pages = 0

            for line in lines:
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                elif "Pages free:" in line:
                    free_pages += int(line.split()[-1].rstrip("."))
                elif "Pages speculative:" in line:
                    free_pages += int(line.split()[-1].rstrip("."))

            available_mb = (free_pages * page_size) // (1024 * 1024)
            return available_mb
    except Exception as e:
        logger.warning(f"Could not determine available memory: {e}")

    return 8192  # 8GB fallback


def parse_memory_config() -> Dict[int, int]:
    """Parse per-rank memory configuration from environment variables.

    Environment variables:
    - RANK_MEMORY_MB_<rank>: Memory limit for specific rank (e.g., RANK_MEMORY_MB_0=192000)
    - DEFAULT_RANK_MEMORY_MB: Default memory limit for ranks not explicitly configured

    Returns:
        Dictionary mapping rank to memory limit in MB
    """
    memory_config = {}

    # Check for rank-specific configurations
    for key, value in os.environ.items():
        if key.startswith("RANK_MEMORY_MB_"):
            try:
                rank = int(key.split("_")[-1])
                memory_mb = int(value)
                memory_config[rank] = memory_mb
                logger.info(f"Configured rank {rank} with {memory_mb} MB memory limit")
            except ValueError as e:
                logger.warning(f"Invalid memory config {key}={value}: {e}")

    # Check for default configuration
    default_memory = os.environ.get("DEFAULT_RANK_MEMORY_MB")
    if default_memory:
        try:
            default_mb = int(default_memory)
            memory_config["default"] = default_mb
            logger.info(f"Default rank memory limit: {default_mb} MB")
        except ValueError as e:
            logger.warning(f"Invalid default memory config: {e}")

    return memory_config


def get_rank_memory_info(rank: int, world_size: int, memory_config: Dict[int, int]) -> MemoryInfo:
    """Get memory information for a specific rank.

    Args:
        rank: The rank to get memory info for
        world_size: Total number of ranks
        memory_config: Dictionary of rank-specific memory configurations

    Returns:
        MemoryInfo object with memory details for the rank
    """
    # Check for rank-specific configuration
    if rank in memory_config:
        configured_mb = memory_config[rank]
        logger.info(f"Rank {rank}: Using configured memory limit {configured_mb} MB")
        return MemoryInfo(
            rank=rank,
            total_memory_mb=configured_mb,
            available_memory_mb=configured_mb,
            reserved_memory_mb=max(1024, configured_mb // 20),  # Reserve 5% or 1GB minimum
        )

    # Check for default configuration
    if "default" in memory_config:
        default_mb = memory_config["default"]
        logger.info(f"Rank {rank}: Using default memory limit {default_mb} MB")
        return MemoryInfo(
            rank=rank,
            total_memory_mb=default_mb,
            available_memory_mb=default_mb,
            reserved_memory_mb=max(1024, default_mb // 20),
        )

    # Auto-detect system memory
    system_memory = get_system_memory_mb()
    available_memory = get_available_memory_mb()

    if system_memory is None:
        # Fallback based on typical configurations
        if available_memory > 150000:  # Likely 192GB Mac Studio
            total_mb = 196608  # 192GB
            available_mb = min(available_memory, int(total_mb * 0.9))
        elif available_memory > 60000:  # Likely 64GB Mac Studio
            total_mb = 65536  # 64GB
            available_mb = min(available_memory, int(total_mb * 0.8))
        elif available_memory > 30000:  # Likely 32GB machine
            total_mb = 32768  # 32GB
            available_mb = min(available_memory, int(total_mb * 0.7))
        else:  # Likely 16GB Mac mini or similar
            total_mb = 16384  # 16GB
            available_mb = min(available_memory, int(total_mb * 0.6))
    else:
        total_mb = system_memory
        available_mb = min(available_memory, int(total_mb * 0.8))  # Use up to 80% of total

    reserved_mb = max(2048, total_mb // 16)  # Reserve at least 2GB or 1/16th of total

    logger.info(f"Rank {rank}: Auto-detected {total_mb} MB total, {available_mb} MB available")

    return MemoryInfo(
        rank=rank,
        total_memory_mb=total_mb,
        available_memory_mb=available_mb,
        reserved_memory_mb=reserved_mb,
    )


def estimate_layer_params_from_config(config) -> int:
    """Estimate parameter count from model configuration as fallback."""
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    n_experts = config.n_routed_experts
    n_shared = config.n_shared_experts
    moe_intermediate = config.moe_intermediate_size
    shared_intermediate = config.shared_expert_intermediate_size

    # Self-attention parameters
    attn_params = (
        hidden_size * hidden_size * 3  # Q, K, V projections
        + hidden_size * hidden_size  # Output projection
    )

    # MoE parameters
    # Shared experts
    shared_params = n_shared * (
        hidden_size * shared_intermediate  # Up projection
        + shared_intermediate * hidden_size  # Down projection
    )

    # Routed experts
    routed_params = n_experts * (
        hidden_size * moe_intermediate  # Up projection
        + moe_intermediate * hidden_size  # Down projection
    )

    # Router
    router_params = hidden_size * n_experts

    # Layer norms
    norm_params = hidden_size * 2  # Two layer norms

    total_params = attn_params + shared_params + routed_params + router_params + norm_params
    return total_params


def estimate_layer_memory(
    layer: nn.Module, config, batch_size: int = 1, seq_len: int = 2048, dtype_size: int = 2
) -> LayerEstimate:
    """Estimate memory usage for a single transformer layer.

    Args:
        layer: The transformer layer module
        config: Model configuration
        batch_size: Batch size for activation estimation
        seq_len: Sequence length for activation estimation
        dtype_size: Size of data type in bytes (2 for float16, 4 for float32)

    Returns:
        LayerEstimate with memory breakdown
    """
    # Estimate parameter memory using MLX tree_flatten
    from mlx.utils import tree_flatten

    param_count = 0

    try:
        # Use MLX's tree_flatten to get all parameters
        flattened_params = tree_flatten(layer.parameters())
        for name, param in flattened_params:
            if hasattr(param, "size"):
                param_count += param.size
            elif hasattr(param, "shape"):
                import math

                param_count += math.prod(param.shape)
            else:
                # Try to estimate from shape info if available
                param_str = str(param)
                if "shape" in param_str:
                    logger.debug(f"Could not determine exact size for {name}, using approximation")
                    # This is a rough approximation for unknown parameter types
                    param_count += 1000  # Rough fallback
    except Exception as e:
        logger.warning(f"Could not access layer parameters: {e}")
        # Fallback estimation based on model config
        param_count = estimate_layer_params_from_config(config)

    param_mb = (param_count * dtype_size) / (1024 * 1024)

    # Estimate activation memory for MoE layer
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    # Self-attention activations
    # Q, K, V projections + attention scores + output
    attn_activations = (
        3 * batch_size * seq_len * hidden_size  # Q, K, V
        + batch_size * config.num_attention_heads * seq_len * seq_len  # Attention scores
        + batch_size * seq_len * hidden_size  # Attention output
    )

    # MoE activations (more complex due to expert routing)
    n_experts = config.n_routed_experts
    experts_per_token = config.num_experts_per_tok
    moe_intermediate = config.moe_intermediate_size
    shared_intermediate = config.shared_expert_intermediate_size

    # Shared experts activations
    shared_activations = batch_size * seq_len * shared_intermediate

    # Routed experts activations (only active experts)
    # Routing computation
    routing_activations = batch_size * seq_len * n_experts

    # Active expert computation (approximation)
    active_expert_activations = batch_size * seq_len * experts_per_token * moe_intermediate

    moe_activations = shared_activations + routing_activations + active_expert_activations

    # Layer norm activations
    norm_activations = 2 * batch_size * seq_len * hidden_size

    total_activations = attn_activations + moe_activations + norm_activations
    activation_mb = (total_activations * dtype_size) / (1024 * 1024)

    # Estimate KV cache memory
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = hidden_size // config.num_attention_heads

    # KV cache: keys + values for this layer
    kv_cache_size = 2 * batch_size * num_kv_heads * seq_len * head_dim
    cache_mb = (kv_cache_size * dtype_size) / (1024 * 1024)

    return LayerEstimate(
        layer_idx=-1,  # Will be set by caller
        parameters_mb=param_mb,
        activations_mb=activation_mb,
        cache_mb=cache_mb,
        total_mb=param_mb + activation_mb + cache_mb,
    )


def estimate_model_memory(
    model, config, batch_size: int = 1, seq_len: int = 2048
) -> List[LayerEstimate]:
    """Estimate memory usage for all model layers.

    Args:
        model: The model to estimate
        config: Model configuration
        batch_size: Batch size for estimation
        seq_len: Sequence length for estimation

    Returns:
        List of LayerEstimate objects, one per layer
    """
    layer_estimates = []

    # Get the actual transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        logger.warning("Could not find model layers for memory estimation")
        return []

    logger.info(f"Estimating memory for {len(layers)} layers")

    for i, layer in enumerate(layers):
        estimate = estimate_layer_memory(layer, config, batch_size, seq_len)
        estimate.layer_idx = i
        layer_estimates.append(estimate)

        logger.debug(
            f"Layer {i}: {estimate.total_mb:.1f} MB "
            f"(params: {estimate.parameters_mb:.1f}, "
            f"activations: {estimate.activations_mb:.1f}, "
            f"cache: {estimate.cache_mb:.1f})"
        )

    total_memory = sum(est.total_mb for est in layer_estimates)
    logger.info(f"Total estimated memory: {total_memory:.1f} MB ({total_memory / 1024:.1f} GB)")

    return layer_estimates


def create_memory_aware_sharding_plan(
    memory_infos: List[MemoryInfo],
    layer_estimates: List[LayerEstimate],
    safety_margin: float = 0.85,
) -> List[ShardingPlan]:
    """Create a memory-aware sharding plan for distributing layers across ranks.

    Args:
        memory_infos: Memory information for each rank
        layer_estimates: Memory estimates for each layer
        safety_margin: Use only this fraction of available memory (default: 85%)

    Returns:
        List of ShardingPlan objects, one per rank
    """
    if not memory_infos or not layer_estimates:
        logger.error("Cannot create sharding plan: missing memory info or layer estimates")
        return []

    num_ranks = len(memory_infos)
    num_layers = len(layer_estimates)

    logger.info(f"Creating memory-aware sharding plan for {num_ranks} ranks, {num_layers} layers")

    # Sort ranks by available memory (descending)
    sorted_ranks = sorted(memory_infos, key=lambda x: x.usable_memory_mb, reverse=True)

    # Calculate effective memory for each rank
    effective_memory = []
    for info in sorted_ranks:
        effective_mb = info.usable_memory_mb * safety_margin
        effective_memory.append(effective_mb)
        logger.info(
            f"Rank {info.rank}: {effective_mb:.1f} MB effective memory "
            f"({info.usable_memory_mb} MB usable * {safety_margin} safety margin)"
        )

    # Greedy allocation algorithm
    sharding_plans = []
    layer_assignments = {}  # rank -> list of layer indices
    memory_usage = {}  # rank -> current memory usage

    # Initialize
    for info in sorted_ranks:
        layer_assignments[info.rank] = []
        memory_usage[info.rank] = 0.0

    # Use a more balanced allocation strategy
    # Sort layers by index (for contiguous assignment)
    sorted_layers = list(enumerate(layer_estimates))

    # First, try to allocate layers sequentially for better pipelining
    current_rank_idx = 0
    layers_per_rank_target = num_layers // num_ranks
    extra_layers = num_layers % num_ranks

    for layer_idx, layer_est in sorted_layers:
        # Determine which rank should get this layer based on sequential assignment
        target_rank_idx = min(current_rank_idx, len(sorted_ranks) - 1)
        target_rank = sorted_ranks[target_rank_idx].rank

        # Check if this rank can fit the layer
        current_usage = memory_usage[target_rank]
        available = effective_memory[target_rank_idx] - current_usage

        if available >= layer_est.total_mb:
            # Assign to target rank
            layer_assignments[target_rank].append(layer_idx)
            memory_usage[target_rank] += layer_est.total_mb
            logger.debug(
                f"Assigned layer {layer_idx} ({layer_est.total_mb:.1f} MB) to rank {target_rank}"
            )
        else:
            # Try to find another rank that can fit this layer
            best_rank = None
            best_remaining = -1

            for i, info in enumerate(sorted_ranks):
                rank = info.rank
                current_usage = memory_usage[rank]
                available_mem = effective_memory[i] - current_usage

                if available_mem >= layer_est.total_mb and available_mem > best_remaining:
                    best_rank = rank
                    best_remaining = available_mem

            if best_rank is not None:
                layer_assignments[best_rank].append(layer_idx)
                memory_usage[best_rank] += layer_est.total_mb
                logger.debug(
                    f"Assigned layer {layer_idx} ({layer_est.total_mb:.1f} MB) to rank {best_rank} (fallback)"
                )
            else:
                # No rank can fit this layer - issue warning and assign to highest capacity rank
                logger.warning(
                    f"Layer {layer_idx} ({layer_est.total_mb:.1f} MB) cannot fit on any rank!"
                )
                best_rank = sorted_ranks[0].rank
                layer_assignments[best_rank].append(layer_idx)
                memory_usage[best_rank] += layer_est.total_mb

        # Move to next rank when we've assigned enough layers
        layers_assigned_to_current = len(layer_assignments[sorted_ranks[current_rank_idx].rank])
        layers_target_for_current = layers_per_rank_target + (
            1 if current_rank_idx < extra_layers else 0
        )

        if (
            layers_assigned_to_current >= layers_target_for_current
            and current_rank_idx < len(sorted_ranks) - 1
        ):
            current_rank_idx += 1

    # Create sharding plans
    for info in memory_infos:
        rank = info.rank
        assigned_layers = sorted(layer_assignments[rank])

        if assigned_layers:
            start_layer = min(assigned_layers)
            end_layer = max(assigned_layers)
            num_layers_assigned = len(assigned_layers)

            # Check for non-contiguous assignment
            if num_layers_assigned != (end_layer - start_layer + 1):
                logger.warning(
                    f"Rank {rank} has non-contiguous layer assignment: {assigned_layers}"
                )
                # For pipeline parallelism, we need contiguous layers
                # Reassign to make contiguous
                start_layer = assigned_layers[0]
                end_layer = assigned_layers[-1]
                num_layers_assigned = end_layer - start_layer + 1
        else:
            # No layers assigned
            start_layer = end_layer = 0
            num_layers_assigned = 0

        plan = ShardingPlan(
            rank=rank,
            start_layer=start_layer,
            end_layer=end_layer,
            num_layers=num_layers_assigned,
            estimated_memory_mb=memory_usage[rank],
            available_memory_mb=info.usable_memory_mb,
        )

        sharding_plans.append(plan)

        logger.info(
            f"Rank {rank}: layers {start_layer}-{end_layer} ({num_layers_assigned} layers), "
            f"{plan.estimated_memory_mb:.1f} MB estimated, "
            f"{plan.memory_utilization:.1%} utilization"
        )

    # Validate that all layers are assigned
    total_assigned = sum(plan.num_layers for plan in sharding_plans)
    if total_assigned != num_layers:
        logger.warning(
            f"Layer assignment mismatch: {total_assigned} assigned vs {num_layers} total"
        )

    return sharding_plans


def validate_sharding_plan(
    sharding_plans: List[ShardingPlan], num_layers: int
) -> Tuple[bool, List[str]]:
    """Validate a sharding plan for correctness.

    Args:
        sharding_plans: List of sharding plans to validate
        num_layers: Expected total number of layers

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check that all layers are covered
    covered_layers = set()
    for plan in sharding_plans:
        if plan.num_layers > 0:
            for layer_idx in range(plan.start_layer, plan.end_layer + 1):
                if layer_idx in covered_layers:
                    issues.append(f"Layer {layer_idx} assigned to multiple ranks")
                covered_layers.add(layer_idx)

    missing_layers = set(range(num_layers)) - covered_layers
    if missing_layers:
        issues.append(f"Missing layers: {sorted(missing_layers)}")

    # Check for memory overcommitment
    for plan in sharding_plans:
        if plan.memory_utilization > 1.0:
            issues.append(f"Rank {plan.rank} memory overcommitted: {plan.memory_utilization:.1%}")

    # Check for empty ranks
    empty_ranks = [plan.rank for plan in sharding_plans if plan.num_layers == 0]
    if empty_ranks:
        issues.append(f"Empty ranks (no layers assigned): {empty_ranks}")

    return len(issues) == 0, issues


def log_sharding_summary(sharding_plans: List[ShardingPlan], layer_estimates: List[LayerEstimate]):
    """Log a comprehensive summary of the sharding plan."""
    logger.info("=" * 60)
    logger.info("MEMORY-AWARE SHARDING PLAN SUMMARY")
    logger.info("=" * 60)

    total_layers = len(layer_estimates)
    total_memory = sum(est.total_mb for est in layer_estimates)

    logger.info(f"Total layers: {total_layers}")
    logger.info(f"Total estimated memory: {total_memory:.1f} MB ({total_memory / 1024:.1f} GB)")
    logger.info("")

    # Per-rank breakdown
    for plan in sorted(sharding_plans, key=lambda x: x.rank):
        if plan.num_layers > 0:
            utilization_color = (
                "HIGH"
                if plan.memory_utilization > 0.9
                else "MEDIUM"
                if plan.memory_utilization > 0.7
                else "LOW"
            )
            logger.info(
                f"Rank {plan.rank:2d}: Layers {plan.start_layer:2d}-{plan.end_layer:2d} "
                f"({plan.num_layers:2d} layers) | "
                f"Memory: {plan.estimated_memory_mb:8.1f} MB / {plan.available_memory_mb:8.1f} MB "
                f"({plan.memory_utilization:6.1%}) [{utilization_color}]"
            )
        else:
            logger.info(f"Rank {plan.rank:2d}: No layers assigned")

    logger.info("")

    # Memory distribution summary
    memory_used = sum(plan.estimated_memory_mb for plan in sharding_plans)
    memory_available = sum(plan.available_memory_mb for plan in sharding_plans)
    avg_utilization = memory_used / memory_available if memory_available > 0 else 0

    logger.info(
        f"Total memory utilization: {memory_used:.1f} MB / {memory_available:.1f} MB ({avg_utilization:.1%})"
    )

    # Efficiency metrics
    max_utilization = max(plan.memory_utilization for plan in sharding_plans if plan.num_layers > 0)
    min_utilization = min(plan.memory_utilization for plan in sharding_plans if plan.num_layers > 0)
    utilization_range = max_utilization - min_utilization

    logger.info(
        f"Utilization range: {min_utilization:.1%} - {max_utilization:.1%} (spread: {utilization_range:.1%})"
    )

    logger.info("=" * 60)
