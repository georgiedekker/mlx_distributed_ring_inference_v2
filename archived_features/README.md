# Archived Features

This directory contains experimental and unused code that was developed for the MLX Distributed Ring Inference project but is no longer actively used in the main codebase.

**Archive Date:** 2025-11-18

## Archived Files

### 1. qwen_moe_mini.py (554 lines)
**Purpose:** Lightweight Mixture-of-Experts (MoE) model implementation with pipeline parallelism support

**Description:**
This file extends the Qwen-MoE-Mini implementation to add support for pipeline parallelism across distributed devices. It was designed to enable running MoE models on 16 GB Mac minis by distributing layers across multiple machines using `mx.distributed` primitives similar to Apple's DeepSeek-V3 strategy.

**Key Features:**
- Pipeline parallelism using point-to-point communication
- Reverse layer ordering (rank 0 processes last layers)
- Custom layer range allocation based on communicator size
- Integration with MLX distributed primitives (recv_like, send, all_gather)

**Why Archived:**
The project standardized on the DeepSeek model architecture rather than pursuing the Qwen-MoE-Mini variant. The pipeline approach implemented here was experimental and potentially unstable on some MLX versions.

---

### 2. memory_aware_sharding.py (633 lines)
**Purpose:** Memory-aware pipeline sharding for heterogeneous distributed clusters

**Description:**
This module provides sophisticated utilities for distributing model layers across ranks based on available memory rather than equal layer splitting. It was designed to support heterogeneous memory configurations, such as mixing Mac Studio machines (192GB) with Mac mini devices (16GB).

**Key Features:**
- Per-rank memory detection and configuration
- Layer size estimation for accurate memory distribution
- Memory-aware layer allocation algorithm
- Environment variable configuration for manual overrides
- Comprehensive logging and OOM prevention
- Support for mixed-memory cluster configurations

**Why Archived:**
The project moved to a simpler equal-distribution strategy that worked reliably across homogeneous clusters. The complexity of memory-aware sharding was deemed unnecessary for the primary use case of 2-3 similar Mac devices.

---

### 3. base.py (583 lines)
**Purpose:** Enhanced KV cache implementation with MLX native caching features

**Description:**
This module provides an enhanced key/value cache implementation that supports MLX's native caching features, including offset tracking, cache management, and serialization. It was designed to address common issues like "IndexError: list index out of range" by properly managing cache state.

**Key Features:**
- Proper offset tracking to avoid index errors
- Cache truncation and management methods
- Support for both list-based and tensor-based cache structures
- Cache serialization/deserialization for persistence
- Compatibility with MLX's make_kv_caches function
- Support for both single-node and distributed scenarios

**Why Archived:**
The MLX framework evolved to include more robust native caching mechanisms that made this custom implementation redundant. The project now uses the standard MLX caching utilities.

---

### 4. server_with_prompt_cache.py (387 lines)
**Purpose:** Alternative inference server implementation with advanced prompt caching

**Description:**
This is a distributed DeepSeek inference server implementation that includes advanced prompt caching based on Awni's approach for prefix reuse optimization. It was designed to optimize scenarios where the first half of prompts are frequently shared across requests.

**Key Features:**
- Advanced prompt caching system with prefix hashing
- In-memory cache store for frequently used prefixes
- Integration with MLX's prompt cache utilities
- File-based cache persistence
- Optimized for shared prompt prefix scenarios

**Why Archived:**
The main `server.py` implementation was chosen as the production version. This alternative implementation with prompt caching added complexity that wasn't needed for the primary use case of interactive inference across distributed devices.

---

### 5. shard.py (56 lines)
**Purpose:** Basic shard utilities for pipeline parallelism

**Description:**
This module defines a simple `Shard` dataclass used to describe sub-ranges of model layers. It includes convenience methods for detecting overlaps and computing the number of layers in a shard.

**Key Features:**
- Frozen dataclass for immutable shard definitions
- Methods to detect first/last layer positions
- Layer count computation
- Shard overlap detection
- Serialization to/from dictionaries

**Why Archived:**
The DeepSeek model's built-in `pipeline()` method handles sharding automatically, making this manual shard management unnecessary. The class was retained for compatibility but is no longer actively used.

---

## Restoration

If you need to restore any of these files:

1. Extract from the archive:
   ```bash
   tar -xzf ../archived_features_2025-11-18.tar.gz
   ```

2. Copy the needed file back to the project root:
   ```bash
   cp archived_features/<filename> ../
   ```

## Notes

- These files are preserved for reference and potential future use
- They represent valid experimental approaches that may be useful for different use cases
- The code is functional but may require updates to work with current MLX versions
- Consider reviewing the git history for the original implementation context

## Related Documentation

For the current active implementation, see:
- `server.py` - Production inference server
- `api.py` - HTTP API wrapper
- Main project README.md - Current architecture documentation
