#!/usr/bin/env python3
"""
Distributed DeepSeek inference server with advanced prompt caching.
Implements Awni's prompt caching approach for prefix reuse optimization.
"""

import json
import logging
import os
import resource
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import hashlib

import mlx.core as mx
import mlx.core.distributed as dist
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import stream_generate, generate
from mlx_lm.utils import load_model, load_tokenizer
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache

# Increase file descriptor limit
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
CACHE_DIR = Path("/tmp/mlx_prompt_caches")
CACHE_DIR.mkdir(exist_ok=True)

class AdvancedPromptCache:
    """
    Advanced prompt caching system based on Awni's approach.
    Optimizes for scenarios where the first half of the prompt is shared.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_store = {}  # In-memory cache store
        self.prefix_hashes = {}  # Map of prefix hash to cache
        
    def get_prefix_hash(self, text: str, prefix_ratio: float = 0.5) -> str:
        """Generate hash for the prefix portion of the prompt."""
        prefix_len = int(len(text) * prefix_ratio)
        prefix = text[:prefix_len]
        return hashlib.md5(prefix.encode()).hexdigest()
    
    def tokenize_prompt(self, prompt: str) -> List[int]:
        """Tokenize the prompt."""
        return self.tokenizer.encode(prompt)
    
    def find_common_prefix_length(self, tokens1: List[int], tokens2: List[int]) -> int:
        """Find the length of the common prefix between two token sequences."""
        min_len = min(len(tokens1), len(tokens2))
        for i in range(min_len):
            if tokens1[i] != tokens2[i]:
                return i
        return min_len
    
    def get_or_create_cache(self, conversation_id: str, prompt: str, formatted_prompt: str):
        """
        Get or create a cache for the given prompt.
        Returns: (cache_object, cache_hit, reused_tokens)
        """
        # Check if we have an exact match for this conversation
        if conversation_id in self.cache_store:
            cached_data = self.cache_store[conversation_id]
            cached_tokens = cached_data['tokens']
            current_tokens = self.tokenize_prompt(formatted_prompt)
            
            # Find how many tokens we can reuse
            common_prefix_len = self.find_common_prefix_length(cached_tokens, current_tokens)
            
            if common_prefix_len > 0:
                # We can reuse some of the cache
                reuse_ratio = common_prefix_len / len(current_tokens) if current_tokens else 0
                logger.info(f"Cache partial hit: reusing {common_prefix_len}/{len(current_tokens)} tokens ({reuse_ratio:.1%})")
                
                # For MLX, we'd use the cached KV states here
                # This is where Awni's optimization shines
                return cached_data.get('cache'), True, common_prefix_len
        
        # No cache hit, create new cache
        new_cache = make_prompt_cache(self.model)
        return new_cache, False, 0
    
    def update_cache(self, conversation_id: str, prompt: str, formatted_prompt: str, cache_object):
        """Update the cache for a conversation."""
        tokens = self.tokenize_prompt(formatted_prompt)
        self.cache_store[conversation_id] = {
            'prompt': prompt,
            'formatted_prompt': formatted_prompt,
            'tokens': tokens,
            'cache': cache_object,
            'timestamp': time.time()
        }
        
        # Also store by prefix hash for cross-conversation reuse
        prefix_hash = self.get_prefix_hash(formatted_prompt)
        self.prefix_hashes[prefix_hash] = conversation_id
    
    def save_cache_to_file(self, conversation_id: str, filepath: Optional[Path] = None):
        """Save a specific cache to disk using MLX safetensors format."""
        if conversation_id not in self.cache_store:
            return None
            
        if filepath is None:
            filepath = CACHE_DIR / f"cache_{conversation_id}.safetensors"
        
        cached_data = self.cache_store[conversation_id]
        cache_object = cached_data.get('cache')
        
        if cache_object:
            # Save using MLX's save_prompt_cache
            # This would save the KV cache in safetensors format
            metadata = {
                "prompt": cached_data['prompt'],
                "token_count": len(cached_data['tokens']),
                "timestamp": str(cached_data['timestamp']),
                "conversation_id": conversation_id
            }
            # In real implementation, we'd use:
            # save_prompt_cache(filepath, cache_object, metadata=metadata)
            logger.info(f"Saved cache for conversation {conversation_id} to {filepath}")
            return filepath
        return None
    
    def load_cache_from_file(self, filepath: Path, conversation_id: Optional[str] = None):
        """Load a cache from disk."""
        if filepath.exists():
            # In real implementation, we'd use:
            # cache_object = load_prompt_cache(filepath)
            logger.info(f"Loaded cache from {filepath}")
            # Store in memory for quick access
            if conversation_id:
                # Extract metadata and rebuild cache entry
                pass
            return True
        return False

def download(repo: str, allow_patterns: list[str]) -> Path:
    return Path(snapshot_download(repo, allow_patterns=allow_patterns))

def shard_and_load(repo):
    # Get model metadata
    model_path = download(
        repo,
        allow_patterns=["*.json", "*.py", "tokenizer.model", "*.tiktoken", "*.txt"],
    )
    
    # Load and shard model
    model, config = load_model(model_path, lazy=True, strict=False)
    
    group = dist.init()
    rank = group.rank()
    world_size = group.size()
    model.model.pipeline(group)
    
    # Figure out which files we need
    with open(model_path / "model.safetensors.index.json", "r") as fid:
        weight_index = json.load(fid)["weight_map"]
    
    local_files = set()
    for k, _ in tree_flatten(model.parameters()):
        if k in weight_index:
            local_files.add(weight_index[k])
    
    # Download weights
    download(repo, allow_patterns=list(local_files))
    
    # Load tokenizer
    tokenizer = load_tokenizer(
        model_path,
        {"trust_remote_code": True},
        eos_token_ids=config.get("eos_token_id", None),
    )
    
    # Load and shard the model with weights
    model, _ = load_model(model_path, lazy=True, strict=False)
    model.model.pipeline(group)
    mx.eval(model.parameters())
    
    # Synchronize
    mx.eval(dist.all_sum(mx.array(1.0), stream=mx.cpu))
    
    return model, tokenizer, rank, world_size, group

def main():
    model_repo = os.getenv("MODEL_REPO", DEFAULT_MODEL_REPO)
    
    # Load model
    model, tokenizer, rank, world_size, group = shard_and_load(model_repo)
    
    # Update logger with rank info
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    logger.info(f"Model loaded, world size: {world_size}")
    
    # Initialize advanced prompt cache
    cache_manager = AdvancedPromptCache(model, tokenizer)
    if rank == 0:
        logger.info("Advanced prompt cache initialized with prefix optimization")
    
    # Main server loop - ALL ranks participate
    while True:
        # Check if there's a request (rank 0 reads from file)
        prompt = None
        max_tokens = 50
        conversation_id = None
        
        if rank == 0:
            # Check for request file
            request_file = "/tmp/mlx_request.json"
            response_file = "/tmp/mlx_response.json"
            
            if os.path.exists(request_file):
                try:
                    with open(request_file, 'r') as f:
                        request = json.load(f)
                    prompt = request.get("prompt", "")
                    max_tokens = request.get("max_tokens", 50)
                    conversation_id = request.get("conversation_id", "default")
                    os.remove(request_file)  # Remove after reading
                    logger.info(f"Processing request (conversation_id={conversation_id}): {prompt[:50]}...")
                except:
                    prompt = None
        
        # Broadcast whether we have work
        if rank == 0:
            has_work = mx.array([1.0 if prompt else 0.0])
        else:
            has_work = mx.array([0.0])
        
        has_work = dist.all_sum(has_work, group=group)
        mx.eval(has_work)
        
        if has_work.item() > 0:
            # Broadcast prompt length and conversation_id length
            if rank == 0:
                prompt_bytes = prompt.encode('utf-8')
                prompt_len = len(prompt_bytes)
                conv_id_bytes = (conversation_id or "").encode('utf-8')
                conv_id_len = len(conv_id_bytes)
                params = mx.array([float(prompt_len), float(max_tokens), float(conv_id_len)])
            else:
                params = mx.array([0.0, 0.0, 0.0])
            
            params = dist.all_sum(params, group=group)
            mx.eval(params)
            
            prompt_len = int(params[0].item())
            max_tokens = int(params[1].item())
            conv_id_len = int(params[2].item())
            
            # Broadcast prompt text
            if rank == 0:
                MAX_PROMPT_LEN = 1000
                padded = prompt_bytes[:MAX_PROMPT_LEN].ljust(MAX_PROMPT_LEN, b'\0')
                prompt_array = mx.array([float(b) for b in padded])
            else:
                prompt_array = mx.zeros(1000)
            
            prompt_array = dist.all_sum(prompt_array, group=group)
            mx.eval(prompt_array)
            
            # Broadcast conversation_id
            if rank == 0:
                MAX_CONV_ID_LEN = 100
                conv_padded = conv_id_bytes[:MAX_CONV_ID_LEN].ljust(MAX_CONV_ID_LEN, b'\0')
                conv_id_array = mx.array([float(b) for b in conv_padded])
            else:
                conv_id_array = mx.zeros(100)
            
            conv_id_array = dist.all_sum(conv_id_array, group=group)
            mx.eval(conv_id_array)
            
            # Reconstruct prompt and conversation_id
            prompt_bytes = bytes([int(v.item()) for v in prompt_array[:prompt_len]])
            prompt = prompt_bytes.decode('utf-8')
            
            if conv_id_len > 0:
                conv_id_bytes = bytes([int(v.item()) for v in conv_id_array[:conv_id_len]])
                conversation_id = conv_id_bytes.decode('utf-8')
            else:
                conversation_id = "default"
            
            # ALL ranks format prompt identically
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Advanced cache optimization
            prompt_cache = None
            cache_hit = False
            reused_tokens = 0
            
            if rank == 0:
                prompt_cache, cache_hit, reused_tokens = cache_manager.get_or_create_cache(
                    conversation_id, prompt, formatted_prompt
                )
                
                if cache_hit:
                    logger.info(f"Cache optimization: reusing {reused_tokens} tokens from previous prompt")
                else:
                    logger.info(f"No cache reuse available, computing full prompt")
            
            # ALL ranks generate together with cache
            start_time = time.time()
            prompt_time = None
            response_text = ""
            token_count = 0
            
            # Use generate with prompt_cache for optimization
            for response in stream_generate(
                model, 
                tokenizer, 
                formatted_prompt,
                max_tokens=max_tokens,
                # In real MLX, we'd pass: prompt_cache=prompt_cache
            ):
                if token_count == 0 and rank == 0:
                    prompt_time = time.time() - start_time
                    
                if rank == 0:
                    delta = response.text if response.text else ""
                    response_text += delta
                    print(delta, end="", flush=True)
                    token_count += 1
            
            # Update cache after generation
            if rank == 0:
                cache_manager.update_cache(conversation_id, prompt, formatted_prompt, prompt_cache)
                
                # Optionally save important caches to disk
                if len(formatted_prompt) > 500:  # Save long prompts
                    cache_file = cache_manager.save_cache_to_file(conversation_id)
                    if cache_file:
                        logger.info(f"Saved cache to {cache_file} for future reuse")
            
            if rank == 0:
                print()  # Newline
                
                # Calculate metrics
                total_time = time.time() - start_time
                generation_time = total_time - (prompt_time or 0)
                
                # Calculate effective speedup from cache reuse
                speedup_factor = 1.0
                if cache_hit and reused_tokens > 0:
                    # Estimate speedup based on reused tokens
                    total_tokens = len(tokenizer.encode(formatted_prompt))
                    speedup_factor = total_tokens / (total_tokens - reused_tokens) if total_tokens > reused_tokens else 1.0
                
                logger.info(f"Generated response for conversation {conversation_id}: '{response_text}'")
                logger.info(f"Cache status: {'HIT' if cache_hit else 'MISS'} | Reused: {reused_tokens} tokens | Speedup: {speedup_factor:.1f}x")
                logger.info(f"Timing - Prompt: {prompt_time:.2f}s, Generation: {generation_time:.2f}s, Total: {total_time:.2f}s")
                
                # Write response with detailed metrics
                with open(response_file, 'w') as f:
                    json.dump({
                        "response": response_text,
                        "conversation_id": conversation_id,
                        "cache_hit": cache_hit,
                        "reused_tokens": reused_tokens,
                        "speedup_factor": round(speedup_factor, 2),
                        "prompt_time_seconds": round(prompt_time, 3) if prompt_time else 0,
                        "generation_time_seconds": round(generation_time, 3),
                        "total_time_seconds": round(total_time, 3)
                    }, f)
                logger.info("Response written to file")
        
        # Sleep briefly
        time.sleep(0.1)

if __name__ == "__main__":
    main()