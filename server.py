#!/usr/bin/env python3
"""
Distributed DeepSeek inference server.

This server loads a sharded model across multiple processes using MLX distributed
and serves inference requests via file-based communication.
"""

import json
import logging
import os
import resource
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.core.distributed as dist
from mlx_lm import stream_generate

from config import get_config
from distributed.utils import broadcast_prompt, shard_and_load


# Initialize configuration
config = get_config()

# Set file descriptor limit
try:
    resource.setrlimit(
        resource.RLIMIT_NOFILE,
        (
            config.performance.file_descriptor_soft_limit,
            config.performance.file_descriptor_hard_limit,
        ),
    )
except (ValueError, OSError) as e:
    print(f"Warning: Could not set file descriptor limit: {e}")

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.level), format=config.logging.simple_format
)
logger = logging.getLogger(__name__)


def validate_request(request: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Validate the request format and extract fields.

    Args:
        request: Request dictionary from JSON file

    Returns:
        Tuple of (error_message, None) if validation fails, or
        (None, None) if validation succeeds. The request dict is validated in-place.

    Raises:
        None - returns error message instead
    """
    if not isinstance(request, dict):
        return "Request must be a JSON object", None

    if "prompt" not in request:
        return "Missing required field: prompt", None

    if not isinstance(request["prompt"], str):
        return "Field 'prompt' must be a string", None

    if len(request["prompt"]) == 0:
        return "Field 'prompt' cannot be empty", None

    if len(request["prompt"]) > config.model.max_prompt_length:
        return f"Prompt exceeds maximum length of {config.model.max_prompt_length}", None

    # Validate optional fields
    if "max_tokens" in request:
        if not isinstance(request["max_tokens"], (int, float)):
            return "Field 'max_tokens' must be a number", None
        if request["max_tokens"] <= 0:
            return "Field 'max_tokens' must be positive", None

    if "conversation_id" in request:
        if not isinstance(request["conversation_id"], str):
            return "Field 'conversation_id' must be a string", None

    return None, None


def read_request_file(request_file: str) -> Optional[Dict]:
    """Read and parse the request file.

    Args:
        request_file: Path to the request file

    Returns:
        Parsed request dictionary, or None if file doesn't exist or parsing fails
    """
    if not os.path.exists(request_file):
        return None

    try:
        with open(request_file, "r") as f:
            request = json.load(f)

        # Validate request format
        error, _ = validate_request(request)
        if error:
            logger.error(f"Invalid request format: {error}")
            return None

        return request
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse request JSON: {e}")
        return None
    except (IOError, OSError) as e:
        logger.error(f"Failed to read request file: {e}")
        return None
    finally:
        # Always try to remove the request file
        try:
            if os.path.exists(request_file):
                os.remove(request_file)
        except (IOError, OSError) as e:
            logger.warning(f"Failed to remove request file: {e}")


def write_response_file(
    response_file: str, response_data: Dict, request_file: Optional[str] = None
) -> bool:
    """Write the response to a file.

    Args:
        response_file: Path to write the response
        response_data: Response dictionary to write
        request_file: Optional path to request file to clean up on error

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(response_file, "w") as f:
            json.dump(response_data, f)
        logger.debug("Response written to file")
        return True
    except (IOError, OSError) as e:
        logger.error(f"Failed to write response file: {e}")

        # If we failed to write response, make sure request file is removed
        # so the client doesn't wait forever
        if request_file and os.path.exists(request_file):
            try:
                os.remove(request_file)
            except (IOError, OSError):
                pass

        return False


def main():
    """Main server loop.

    Loads the model and enters an infinite loop waiting for requests.
    All ranks participate in generation.
    """
    model_repo = os.getenv("MODEL_REPO", config.model.default_repo)

    logger.info(f"Loading model: {model_repo}")

    # Load model with error handling
    try:
        model, tokenizer, rank, world_size, group = shard_and_load(model_repo, config=config)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Update logger format to include rank
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)

    # Update logging format to include rank
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(config.logging.format))

    logger.info(f"Model loaded, world size: {world_size}")

    # Simple conversation cache for rank 0
    cache: Dict[str, Tuple[str, str]] = {} if rank == 0 else None

    # Main server loop - ALL ranks participate
    while True:
        # Check if there's a request (rank 0 reads from file)
        prompt = None
        max_tokens = config.model.default_max_tokens
        conversation_id = None
        cache_hit = False

        if rank == 0:
            # Read request from file
            request = read_request_file(config.paths.request_file)

            if request:
                prompt = request.get("prompt", "")
                max_tokens = request.get("max_tokens", config.model.default_max_tokens)
                conversation_id = request.get("conversation_id", "default")

                # Check cache
                if conversation_id in cache:
                    cached_prompt, _ = cache[conversation_id]
                    if prompt.startswith(cached_prompt):
                        cache_hit = True

                logger.info(
                    f"Processing: {prompt[:50]}... (cache: {'HIT' if cache_hit else 'MISS'})"
                )

        # Broadcast prompt and parameters from rank 0 to all other ranks
        has_work, prompt, max_tokens = broadcast_prompt(
            prompt, rank, max_tokens, group, max_len=config.model.max_prompt_length
        )

        if has_work:
            # ALL ranks format prompt identically
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # ALL ranks generate together
            logger.info("Starting generation")
            response_text = ""
            last_response = None

            try:
                for response in stream_generate(
                    model, tokenizer, formatted_prompt, max_tokens=max_tokens
                ):
                    if rank == 0:
                        # stream_generate returns DELTAS (new tokens), not accumulated text!
                        # We need to accumulate them ourselves
                        delta = response.text if response.text else ""
                        response_text += delta
                        print(delta, end="", flush=True)
                        # Keep the last response to get metrics
                        last_response = response
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                if rank == 0:
                    # Write error response
                    write_response_file(
                        config.paths.response_file,
                        {
                            "error": str(e),
                            "response": "",
                            "conversation_id": conversation_id or "default",
                        },
                    )
                continue

            if rank == 0:
                print()  # Newline

                # Get actual metrics from stream_generate's final response
                prompt_tokens = getattr(last_response, "prompt_tokens", 0) if last_response else 0
                prompt_tps = getattr(last_response, "prompt_tps", 0) if last_response else 0
                generation_tokens = (
                    getattr(last_response, "generation_tokens", 0) if last_response else 0
                )
                generation_tps = getattr(last_response, "generation_tps", 0) if last_response else 0

                logger.info(f"Generated: '{response_text}'")
                logger.info(
                    f"Metrics - Prompt: {prompt_tokens} tokens @ {prompt_tps:.1f} tok/s, "
                    f"Generated: {generation_tokens} tokens @ {generation_tps:.1f} tok/s"
                )

                # Cache the response
                cache[conversation_id] = (prompt, response_text)

                # Write response with metrics
                write_response_file(
                    config.paths.response_file,
                    {
                        "response": response_text,
                        "conversation_id": conversation_id,
                        "cache_hit": cache_hit,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generation_tokens,
                        "prompt_eval_tokens_per_second": round(prompt_tps, 2),
                        "eval_tokens_per_second": round(generation_tps, 2),
                    },
                )

            # ALL ranks log completion
            logger.info("Generation completed")

        # Sleep briefly before checking for next request
        time.sleep(config.performance.poll_interval)


if __name__ == "__main__":
    main()
