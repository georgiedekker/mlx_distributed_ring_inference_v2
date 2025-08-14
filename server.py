#!/usr/bin/env python3
"""
Distributed DeepSeek inference server.
"""

import json
import logging
import os
import resource
import time
from pathlib import Path

import mlx.core as mx
import mlx.core.distributed as dist
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.utils import load_model, load_tokenizer

# Increase file descriptor limit
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit"

def download(repo: str, allow_patterns: list[str]) -> Path:
    return Path(snapshot_download(repo, allow_patterns=allow_patterns))

def shard_and_load(repo):
    # Use local cached model instead of downloading
    import os
    username = os.getenv('USER', 'mini1')  # Get current user (mini1 or mini2)
    model_path = Path(f"/Users/{username}/.cache/huggingface/hub/models--mlx-community--DeepSeek-Coder-V2-Lite-Instruct-8bit")
    
    # Find the snapshot directory
    snapshots = list(model_path.glob("snapshots/*"))
    if not snapshots:
        raise FileNotFoundError(f"No model snapshots found in {model_path}")
    model_path = snapshots[0]  # Use first (and likely only) snapshot
    
    logger.info(f"Using cached model from: {model_path}")
    
    # Load and shard model
    model, config = load_model(model_path, lazy=True, strict=False)
    
    group = dist.init()
    rank = group.rank()
    world_size = group.size()
    
    logger.info(f"[Rank {rank}] Before sharding: model has {len(list(model.parameters()))} parameters")
    
    # Apply pipeline sharding - this MUST distribute layers evenly
    model.model.pipeline(group)
    
    logger.info(f"[Rank {rank}] After sharding: model has {len(list(model.parameters()))} parameters")
    
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
    
    logger.info(f"[Rank {rank}] Model loaded, world size: {world_size}")
    
    # Simple conversation cache for rank 0
    cache = {} if rank == 0 else None
    
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
                    
                    # Check cache
                    cache_hit = False
                    if conversation_id in cache:
                        cached_prompt, _ = cache[conversation_id]
                        if prompt.startswith(cached_prompt):
                            cache_hit = True
                    
                    logger.info(f"[Rank {rank}] Processing: {prompt[:50]}... (cache: {'HIT' if cache_hit else 'MISS'})")
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
            # Broadcast prompt length
            if rank == 0:
                prompt_bytes = prompt.encode('utf-8')
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
                MAX_PROMPT_LEN = 1000
                padded = prompt_bytes[:MAX_PROMPT_LEN].ljust(MAX_PROMPT_LEN, b'\0')
                prompt_array = mx.array([float(b) for b in padded])
            else:
                prompt_array = mx.zeros(1000)
            
            prompt_array = dist.all_sum(prompt_array, group=group)
            mx.eval(prompt_array)
            
            # Reconstruct prompt
            prompt_bytes = bytes([int(v.item()) for v in prompt_array[:prompt_len]])
            prompt = prompt_bytes.decode('utf-8')
            
            # ALL ranks format prompt identically
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )
            
            # ALL ranks generate together
            logger.info(f"[Rank {rank}] Starting generation")
            response_text = ""
            last_response = None
            for response in stream_generate(
                model, 
                tokenizer, 
                formatted_prompt,
                max_tokens=max_tokens
            ):
                if rank == 0:
                    # stream_generate returns DELTAS (new tokens), not accumulated text!
                    # We need to accumulate them ourselves
                    delta = response.text if response.text else ""
                    response_text += delta
                    print(delta, end="", flush=True)
                    # Keep the last response to get metrics
                    last_response = response
            
            if rank == 0:
                print()  # Newline
                
                # Get actual metrics from stream_generate's final response
                prompt_tokens = getattr(last_response, 'prompt_tokens', 0) if last_response else 0
                prompt_tps = getattr(last_response, 'prompt_tps', 0) if last_response else 0
                generation_tokens = getattr(last_response, 'generation_tokens', 0) if last_response else 0
                generation_tps = getattr(last_response, 'generation_tps', 0) if last_response else 0
                
                logger.info(f"[Rank {rank}] Generated: '{response_text}'")
                logger.info(f"[Rank {rank}] Metrics - Prompt: {prompt_tokens} tokens @ {prompt_tps:.1f} tok/s, Generated: {generation_tokens} tokens @ {generation_tps:.1f} tok/s")
                
                # Cache the response
                cache[conversation_id] = (prompt, response_text)
                
                # Write response with metrics
                with open(response_file, 'w') as f:
                    json.dump({
                        "response": response_text,
                        "conversation_id": conversation_id,
                        "cache_hit": cache_hit,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generation_tokens,
                        "prompt_eval_tokens_per_second": round(prompt_tps, 2),
                        "eval_tokens_per_second": round(generation_tps, 2)
                    }, f)
                logger.info(f"[Rank {rank}] Response written to file")
            
            # ALL ranks log completion  
            logger.info(f"[Rank {rank}] Generation completed")
        
        # Sleep briefly
        time.sleep(0.1)

if __name__ == "__main__":
    main()