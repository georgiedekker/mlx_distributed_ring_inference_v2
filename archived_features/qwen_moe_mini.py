"""
Qwen‑MoE‑Mini: A lightweight MoE model with pipeline support for 16 GB Mac minis.

This file extends the original Qwen‑MoE‑Mini implementation by adding
support for **pipeline parallelism**.  The implementation follows the
same strategy used in Apple’s DeepSeek‑V3 model: each process in
`mx.distributed` is assigned a slice of the transformer layers and
cooperates with its neighbours via point‑to‑point communication and a
final `all_gather` to assemble the full hidden state.  The low‑rank
processes compute the later layers while the high‑rank processes compute
the earlier layers, so that rank 0 holds the final layers and returns
the normalized hidden state for the language‑model head.

**Important caveats**
---------------------

1. The pipeline is implemented using `mx.distributed.recv_like`,
   `mx.distributed.send` and `mx.distributed.all_gather`.  These
   primitives are identical to those used in the DeepSeek implementation,
   but they may hang or trigger GPU timeouts on some versions of MLX
   when used on Apple hardware.  You should test this implementation
   thoroughly on your machines and fall back to an `all_gather`‑only
   strategy if necessary.

2. The pipeline splits layers in **reverse order** so that rank 0
   processes the **last** layers and rank (pipeline_size − 1) processes
   the **first** layers.  This mirrors the DeepSeek behaviour【751894194149196†L419-L465】.  If you
   prefer to process the first layers on rank 0, adjust `start_idx`
   accordingly.

3. The `ModelArgs.shard` field is ignored during pipeline execution; it
   is only used when running single‑device inference.  The pipeline
   method computes its own layer ranges based on the communicator size.

4. During distributed inference you should always call
   `model.pipeline(group)` immediately after `mx.distributed.init()` and
   before downloading weight files or evaluating parameters.  See
   Apple’s `pipeline_generate.py` example for guidance【304817515359609†L54-L83】.

"""

from dataclasses import dataclass, field
from typing import Optional, List
import math
import os

import mlx.core as mx
import mlx.nn as nn

from base import IdentityBlock, KVCache  # Import from our base module
from shard import Shard
from memory_aware_sharding import (
    get_rank_memory_info,
    parse_memory_config,
    estimate_model_memory,
    create_memory_aware_sharding_plan,
    validate_sharding_plan,
    log_sharding_summary,
)


@dataclass
class ModelArgs:
    """Configuration for Qwen‑MoE‑Mini.

    The `shard` field is still present for backward compatibility, but
    when using pipeline parallelism its values are ignored.
    """

    model_type: str = "qwen_moe_mini"
    vocab_size: int = 32_000
    hidden_size: int = 1_024  # Smaller than standard models
    intermediate_size: int = 2_816  # Reduced for memory efficiency
    num_hidden_layers: int = 16  # Fewer layers for 16GB constraint
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA for memory efficiency

    # MoE specific parameters
    n_routed_experts: int = 8  # Total number of experts
    n_shared_experts: int = 2  # Always‑active experts
    num_experts_per_tok: int = 2  # Top‑k routing
    moe_intermediate_size: int = 1_408  # Expert FFN size
    shared_expert_intermediate_size: int = 2_816

    # Standard parameters
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 8_192
    tie_word_embeddings: bool = False

    # Sharding support (unused in pipeline mode)
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

    def __post_init__(self):
        # Ensure shard is a Shard instance
        if isinstance(self.shard, Shard):
            return
        if not isinstance(self.shard, dict):
            raise TypeError(
                f"Expected shard to be a Shard instance or a dict, got {type(self.shard)}"
            )
        self.shard = Shard(**self.shard)


class MoEGate(nn.Module):
    """Router for selecting experts."""

    def __init__(self, hidden_size: int, n_routed_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(hidden_size, n_routed_experts, bias=False)

    def __call__(self, x: mx.array) -> tuple:
        # x shape: [batch, seq_len, hidden_size]
        logits = self.gate(x)  # [batch, seq_len, n_experts]

        # Get top‑k experts
        scores = mx.softmax(logits, axis=-1)
        expert_indices = mx.argpartition(-scores, kth=self.num_experts_per_tok - 1, axis=-1)
        expert_indices = expert_indices[..., : self.num_experts_per_tok]

        # Get weights for selected experts
        expert_weights = mx.take_along_axis(scores, expert_indices, axis=-1)
        expert_weights = expert_weights / mx.sum(expert_weights, axis=-1, keepdims=True)

        return expert_indices, expert_weights


class MoEBlock(nn.Module):
    """Mixture of Experts feed‑forward block."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Router
        self.gate = MoEGate(
            config.hidden_size,
            config.n_routed_experts,
            config.num_experts_per_tok,
        )

        # Shared experts (always active)
        self.shared_experts = nn.Sequential(
            nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.shared_expert_intermediate_size, config.hidden_size, bias=False),
        )

        # Routed experts
        self.experts: List[nn.Sequential] = []
        for _ in range(config.n_routed_experts):
            expert = nn.Sequential(
                nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False),
            )
            self.experts.append(expert)

    def __call__(self, x: mx.array) -> mx.array:
        # Shared experts contribution (always active)
        shared_output = self.shared_experts(x)

        # Get routing decisions
        expert_indices, expert_weights = self.gate(x)

        # Process through selected experts
        expert_output = mx.zeros_like(x)
        for i in range(self.num_experts_per_tok):
            for expert_id in range(self.n_routed_experts):
                # Create mask for tokens routed to this expert
                mask = expert_indices[..., i] == expert_id
                if mx.any(mask):
                    expert_input = x * mask[..., None]
                    output = self.experts[expert_id](expert_input)
                    weight = expert_weights[..., i : i + 1] * mask[..., None]
                    expert_output = expert_output + output * weight

        # Combine shared and routed expert outputs
        return shared_output + expert_output


class QwenMoEMiniDecoderLayer(nn.Module):
    """Transformer decoder layer with MoE feed‑forward."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Self‑attention
        self.self_attn = nn.MultiHeadAttention(
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            bias=False,
        )
        # MoE FFN
        self.mlp = MoEBlock(config)
        # Layer norms
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        # Self‑attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, x, x, mask=mask)
        x = residual + x

        # MoE FFN with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class QwenMoEMiniModel(nn.Module):
    """Decoder‑only model with pipeline parallelism support.

    The pipeline is implemented in a similar fashion to the DeepSeek‑V3
    pipeline: ranks are assigned contiguous slices of transformer
    layers in **reverse** order so that rank 0 holds the last layers.  At
    inference time each rank receives hidden states from the next rank,
    applies its local layers, sends the result to the previous rank and
    finally participates in an `all_gather` to broadcast the hidden
    state for the language‑model head.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Embeddings (present on all ranks; only used on first stage)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers (initially all present; will be pruned in pipeline)
        self.layers: List[nn.Module] = []
        for i in range(self.num_hidden_layers):
            self.layers.append(QwenMoEMiniDecoderLayer(config))

        # Final norm (always present; applied on final rank)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Pipeline state
        self.pipeline_rank: int = 0
        self.pipeline_size: int = 1
        self.start_idx: int = 0
        self.end_idx: int = self.num_hidden_layers
        self.num_layers: int = self.num_hidden_layers

    # --- Pipeline API ---
    def pipeline(self, group, use_memory_aware: bool = None):
        """Split layers across ranks and prepare for pipelined inference.

        Each rank receives a contiguous slice of `self.layers` in reverse
        order so that rank 0 gets the last layers.  The unused layers are
        replaced with `IdentityBlock` so that they do not consume memory
        during the forward pass.  After calling this method you should
        synchronise all ranks with a small collective (e.g. `all_sum`) to
        ensure the pipeline is set up on every process.
        """
        import logging

        logger = logging.getLogger(__name__)

        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        total_layers = len(self.layers)

        # Determine whether to use memory-aware sharding
        if use_memory_aware is None:
            use_memory_aware = os.environ.get("MEMORY_AWARE_SHARDING", "true").lower() == "true"

        if use_memory_aware and self.pipeline_size > 1:
            logger.info(f"Using memory-aware pipeline sharding for {self.pipeline_size} ranks")
            try:
                self._setup_memory_aware_pipeline(group)
            except Exception as e:
                logger.error(f"Memory-aware sharding failed: {e}")
                logger.info("Falling back to traditional equal layer distribution")
                self._setup_traditional_pipeline(group)
        else:
            if not use_memory_aware:
                logger.info("Using traditional equal layer distribution (memory-aware disabled)")
            else:
                logger.info("Using traditional equal layer distribution (single rank)")
            self._setup_traditional_pipeline(group)

        return self

    def _setup_memory_aware_pipeline(self, group):
        """Setup pipeline using memory-aware layer distribution."""
        import logging

        logger = logging.getLogger(__name__)

        total_layers = len(self.layers)

        # Parse memory configuration
        memory_config = parse_memory_config()

        # Get memory info for all ranks
        memory_infos = []
        for rank in range(self.pipeline_size):
            memory_info = get_rank_memory_info(rank, self.pipeline_size, memory_config)
            memory_infos.append(memory_info)

        # Estimate layer memory requirements
        layer_estimates = estimate_model_memory(self, self.args)

        if not layer_estimates:
            raise ValueError("Could not estimate layer memory requirements")

        # Create memory-aware sharding plan
        sharding_plans = create_memory_aware_sharding_plan(memory_infos, layer_estimates)

        if not sharding_plans:
            raise ValueError("Could not create memory-aware sharding plan")

        # Validate the plan
        is_valid, issues = validate_sharding_plan(sharding_plans, total_layers)
        if not is_valid:
            logger.warning("Sharding plan has issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        # Log the sharding summary
        log_sharding_summary(sharding_plans, layer_estimates)

        # Find our rank's plan
        our_plan = next((plan for plan in sharding_plans if plan.rank == self.pipeline_rank), None)
        if our_plan is None:
            raise ValueError(f"No sharding plan found for rank {self.pipeline_rank}")

        # Apply the sharding plan
        if our_plan.num_layers > 0:
            # Note: We need to maintain reverse ordering for pipeline compatibility
            # Convert forward indices to reverse indices
            forward_start = our_plan.start_layer
            forward_end = our_plan.end_layer

            # In reverse ordering, rank 0 gets the last layers
            # Map forward layer indices to reverse indices
            reverse_start = total_layers - forward_end - 1
            reverse_end = total_layers - forward_start - 1

            self.start_idx = reverse_start
            self.end_idx = reverse_end + 1  # end_idx is exclusive
            self.num_layers = our_plan.num_layers

            logger.info(
                f"Rank {self.pipeline_rank}: Forward layers {forward_start}-{forward_end} -> "
                f"Reverse indices {self.start_idx}-{self.end_idx - 1} ({self.num_layers} layers)"
            )
        else:
            # No layers assigned to this rank
            self.start_idx = 0
            self.end_idx = 0
            self.num_layers = 0
            logger.warning(f"Rank {self.pipeline_rank}: No layers assigned!")

        # Replace unused layers with IdentityBlock to free resources
        for idx in range(self.end_idx, total_layers):
            self.layers[idx] = IdentityBlock()
        for idx in range(self.start_idx):
            self.layers[idx] = IdentityBlock()

        logger.info(
            f"Rank {self.pipeline_rank}: Memory-aware sharding complete. "
            f"Using layers {self.start_idx}-{self.end_idx - 1} ({self.num_layers} layers)"
        )

    def _setup_traditional_pipeline(self, group):
        """Setup pipeline using traditional equal layer distribution."""
        import logging

        logger = logging.getLogger(__name__)

        total_layers = len(self.layers)
        base = total_layers // self.pipeline_size
        extra = total_layers % self.pipeline_size

        # compute number of layers for this rank (reverse ordering)
        layers_this = base + (1 if self.pipeline_rank < extra else 0)

        # start index in reversed allocation: rank 0 gets the last slice
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * base + min(
            extra, self.pipeline_size - self.pipeline_rank - 1
        )
        self.end_idx = self.start_idx + layers_this

        # protect against overflow
        self.start_idx = max(self.start_idx, 0)
        self.end_idx = min(self.end_idx, total_layers)

        # replace unused layers with IdentityBlock to free resources
        for idx in range(self.end_idx, total_layers):
            self.layers[idx] = IdentityBlock()
        for idx in range(self.start_idx):
            self.layers[idx] = IdentityBlock()

        # compute number of layers this rank will execute
        self.num_layers = max(self.end_idx - self.start_idx, 0)

        logger.info(
            f"Rank {self.pipeline_rank}: Traditional sharding complete. "
            f"Using layers {self.start_idx}-{self.end_idx - 1} ({self.num_layers} layers)"
        )

    # --- Forward ---
    def __call__(
        self,
        x: mx.array,
        cache: Optional[List[Optional[KVCache]]] = None,
    ) -> mx.array:
        """Forward pass with optional pipeline communication.

        If the model has been initialised with `pipeline()`, the forward
        call orchestrates communication across ranks.  Otherwise it acts
        like a standard decoder.  See DeepSeek‑V3’s implementation for
        inspiration【751894194149196†L419-L465】.
        """
        # Initial embedding – always computed on all ranks
        h = self.embed_tokens(x)

        # If not using pipeline, run all layers and return norm(h)
        if self.pipeline_size == 1:
            mask = None
            T = h.shape[1]
            if T > 1:
                mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
                mask = mask.astype(h.dtype)
            if cache is None:
                cache = [None] * len(self.layers)
            for i, layer in enumerate(self.layers):
                h = layer(h, mask, cache[i])
            return self.norm(h)

        # Use pipeline – ranks coordinate via send/recv/all_gather
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        # prepare mask and cache for local layers
        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)
        if cache is None:
            cache = [None] * self.num_layers

        # Receive hidden state from next rank (higher pipeline rank) unless we
        # are the first stage (highest rank)
        # Note: reversed ordering means rank (pipeline_size-1) is first stage.
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        # Run local layers
        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i])

        # Send hidden state to previous rank unless we are the last stage (rank=0)
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)

        # Gather outputs from all ranks – returns concatenated array
        all_h = mx.distributed.all_gather(h)
        # Extract only the first batch_size rows, which correspond to the
        # last stage’s output; this matches the DeepSeek implementation
        batch_size = x.shape[0]
        h = all_h[:batch_size]

        # Apply final norm on all ranks (the LM head will be applied only on rank 0)
        return self.norm(h)


class Model(nn.Module):
    """Top‑level Qwen‑MoE‑Mini model with LM head and pipeline support."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = QwenMoEMiniModel(config)
        # LM head defined on all ranks; applied on final stage only
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Pipeline state
        self.pipeline_rank: int = 0
        self.pipeline_size: int = 1

    def pipeline(self, group, use_memory_aware: bool = None):
        """Apply pipeline parallelism to the underlying model."""
        self.model.pipeline(group, use_memory_aware)
        self.pipeline_rank = self.model.pipeline_rank
        self.pipeline_size = self.model.pipeline_size
        return self

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Optional[KVCache]]] = None,
    ) -> mx.array:
        """Forward pass; apply LM head only on the final pipeline stage."""
        out = self.model(inputs, cache)
        # In pipeline mode apply lm_head only on rank 0
        if self.pipeline_size == 1 or self.pipeline_rank == 0:
            return self.lm_head(out)
        # Other ranks return zeros to keep shapes consistent
        return mx.zeros_like(out)

    def sanitize(self, weights):
        """Sanitise weights for sharded execution.

        In pipeline mode this method is unused, but it remains for
        compatibility with single‑device sharding.
        """
        shard_state_dict = {}
        for key, value in weights.items():
            # Handle layer weights
            if key.startswith("model.layers."):
                layer_num = int(key.split(".")[2])
                if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
                    shard_state_dict[key] = value
            # Handle embeddings
            elif self.args.shard.is_first_layer() and key.startswith("model.embed_tokens"):
                shard_state_dict[key] = value
            # Handle final norm and LM head
            elif self.args.shard.is_last_layer() and (
                key.startswith("model.norm") or key.startswith("lm_head")
            ):
                shard_state_dict[key] = value
        # Stack expert weights if needed
        for l in range(self.args.num_hidden_layers):
            if self.args.shard.start_layer <= l <= self.args.shard.end_layer:
                prefix = f"model.layers.{l}.mlp"
                for expert_type in ["experts"]:
                    for param in ["weight"]:
                        expert_keys = [
                            f"{prefix}.{expert_type}.{i}.0.{param}"
                            for i in range(self.args.n_routed_experts)
                        ]
                        if all(k in shard_state_dict for k in expert_keys):
                            stacked = mx.stack([shard_state_dict.pop(k) for k in expert_keys])
                            shard_state_dict[f"{prefix}.{expert_type}.{param}"] = stacked
        return shard_state_dict

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
