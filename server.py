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
import socket
import time
from typing import Dict, Optional, Tuple

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


def read_request_socket(
    server_sock: socket.socket,
) -> Tuple[Optional[Dict], Optional[socket.socket]]:
    """Non-blocking read from Unix domain socket.

    Args:
        server_sock: The listening server socket

    Returns:
        Tuple of (request_dict, client_connection) or (None, None) if no pending request.
    """
    try:
        conn, _ = server_sock.accept()
        data = b""
        conn.settimeout(5.0)  # per-connection read timeout
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        if data:
            request = json.loads(data.decode().strip())
            error, _ = validate_request(request)
            if error:
                logger.error(f"Invalid request format: {error}")
                # Send error back and close
                try:
                    err = json.dumps({"error": error}).encode() + b"\n"
                    conn.sendall(err)
                except Exception:
                    pass
                conn.close()
                return None, None
            return request, conn
        conn.close()
        return None, None
    except BlockingIOError:
        return None, None  # no pending connection
    except Exception as e:
        logger.error(f"Socket read error: {e}")
        return None, None


def write_response_socket(conn: socket.socket, response_data: Dict) -> bool:
    """Send response back through the client's socket connection.

    Args:
        conn: The client socket connection
        response_data: Response dictionary to send

    Returns:
        True if successful, False otherwise
    """
    try:
        data = json.dumps(response_data).encode() + b"\n"
        conn.sendall(data)
        conn.close()
        logger.debug("Response sent via socket")
        return True
    except Exception as e:
        logger.error(f"Socket write error: {e}")
        try:
            conn.close()
        except Exception:
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

    # --- Unix domain socket setup (rank 0 only) ---
    server_sock: Optional[socket.socket] = None
    sock_path = config.paths.socket_path

    if rank == 0:
        if os.path.exists(sock_path):
            os.remove(sock_path)
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(sock_path)
        server_sock.listen(8)
        server_sock.setblocking(False)
        logger.info(f"Listening on Unix socket: {sock_path}")

    # Simple conversation cache for rank 0
    cache: Optional[Dict[str, Tuple[str, str]]] = {} if rank == 0 else None

    try:
        # Main server loop - ALL ranks participate
        while True:
            prompt = None
            max_tokens = config.model.default_max_tokens
            conversation_id = None
            cache_hit = False
            client_conn: Optional[socket.socket] = None

            if rank == 0:
                assert server_sock is not None
                request, client_conn = read_request_socket(server_sock)

                if request:
                    prompt = request.get("prompt", "")
                    max_tokens = request.get("max_tokens", config.model.default_max_tokens)
                    conversation_id = request.get("conversation_id", "default")

                    # Check cache
                    assert cache is not None
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
                            delta = response.text if response.text else ""
                            response_text += delta
                            print(delta, end="", flush=True)
                            last_response = response
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    if rank == 0 and client_conn:
                        write_response_socket(
                            client_conn,
                            {
                                "error": str(e),
                                "response": "",
                                "conversation_id": conversation_id or "default",
                            },
                        )
                        client_conn = None
                    continue

                if rank == 0:
                    print()  # Newline

                    prompt_tokens = (
                        getattr(last_response, "prompt_tokens", 0) if last_response else 0
                    )
                    prompt_tps = getattr(last_response, "prompt_tps", 0) if last_response else 0
                    generation_tokens = (
                        getattr(last_response, "generation_tokens", 0) if last_response else 0
                    )
                    generation_tps = (
                        getattr(last_response, "generation_tps", 0) if last_response else 0
                    )

                    logger.info(f"Generated: '{response_text}'")
                    logger.info(
                        f"Metrics - Prompt: {prompt_tokens} tokens @ {prompt_tps:.1f} tok/s, "
                        f"Generated: {generation_tokens} tokens @ {generation_tps:.1f} tok/s"
                    )

                    # Cache the response
                    assert cache is not None
                    cache[str(conversation_id)] = (prompt, response_text)

                    # Send response via socket
                    if client_conn:
                        write_response_socket(
                            client_conn,
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
                        client_conn = None

                # ALL ranks log completion
                logger.info("Generation completed")

            # Sleep briefly before checking for next request
            time.sleep(config.performance.poll_interval)

    finally:
        # Cleanup socket on exit
        if rank == 0 and server_sock:
            server_sock.close()
            if os.path.exists(sock_path):
                os.remove(sock_path)
            logger.info("Socket cleaned up")


if __name__ == "__main__":
    main()
