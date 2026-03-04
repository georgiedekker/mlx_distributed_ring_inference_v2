#!/usr/bin/env python3
"""
API wrapper for the distributed MLX server.

Communicates via files with the distributed inference process.
Provides a REST API compatible with OpenAI's chat completion format.
"""

import asyncio
import json
import logging
import socket
import time
import uuid
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import get_config


# Initialize configuration
config = get_config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.level), format=config.logging.simple_format
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLX Distributed Inference API",
    description="REST API for distributed MLX inference",
    version="1.0.0",
)

# Add CORS middleware if enabled
if config.api.enable_cors:
    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS enabled for origins: {config.api.cors_origins}")


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    content: str = Field(..., description="Content of the message")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate that role is one of the allowed values."""
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be one of: user, assistant, system")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v


class ChatRequest(BaseModel):
    """Chat completion request following OpenAI format."""

    model: str = Field(default="DeepSeek", description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    max_tokens: int = Field(
        default=100, ge=1, le=4096, description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature (not currently used)"
    )
    stream: bool = Field(
        default=False, description="Whether to stream responses (not currently supported)"
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Optional conversation ID for caching"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List["ChatMessage"]) -> List["ChatMessage"]:
        """Validate that messages list is not empty."""
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v


class ChatResponse(BaseModel):
    """Chat completion response following OpenAI format."""

    id: str = Field(..., description="Unique response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict] = Field(..., description="List of completion choices")
    conversation_id: str = Field(..., description="Conversation ID")
    usage: Optional[Dict] = Field(default=None, description="Token usage statistics")
    performance: Optional[Dict] = Field(default=None, description="Performance metrics")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")


@app.get("/health")
async def health():
    """Health check endpoint.

    Returns:
        dict: Health status information
    """
    return {
        "status": "healthy",
        "api": "ready",
        "config": {"port": config.api.port, "cors_enabled": config.api.enable_cors},
    }


@app.get("/health/live")
async def health_live():
    """Kubernetes liveness probe endpoint.

    Returns:
        dict: Liveness status
    """
    return {"status": "alive"}


@app.get("/model/info")
async def model_info():
    """Model information endpoint for MLXModelRegistry compatibility.

    Returns:
        dict: Model info with top-level fields and a ``models`` list for registry compat
    """
    model_entry = {
        "id": config.model.repo,
        "model": config.model.repo,
        "max_context_length": config.kv_cache.max_sequence_length,
        "max_tokens": config.performance.default_max_tokens,
        "backend": "distributed_ring",
        "num_devices": config.distributed.num_devices,
    }
    return {
        **model_entry,
        "models": [model_entry],
    }


@app.get("/")
async def root():
    """Root endpoint with API information.

    Returns:
        dict: API information
    """
    return {
        "name": "MLX Distributed Inference API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "chat": "/v1/chat/completions",
        },
    }


async def send_request_socket(socket_path: str, request_data: Dict, timeout: int) -> Dict:
    """Send request via Unix domain socket and wait for response.

    Args:
        socket_path: Path to the Unix domain socket
        request_data: Request dictionary to send
        timeout: Maximum time to wait in seconds

    Returns:
        Response data dictionary

    Raises:
        HTTPException: If connection fails, timeout, or IPC error
    """
    loop = asyncio.get_event_loop()

    def _sync_send() -> Dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(socket_path)
        data = json.dumps(request_data).encode() + b"\n"
        sock.sendall(data)
        # Signal end of request
        sock.shutdown(socket.SHUT_WR)
        # Read response
        response_data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response_data += chunk
        sock.close()
        return json.loads(response_data.decode().strip())

    try:
        return await loop.run_in_executor(None, _sync_send)
    except socket.timeout:
        raise HTTPException(
            status_code=504,
            detail=f"Request timeout after {timeout}s. Server may be overloaded.",
        )
    except ConnectionRefusedError:
        raise HTTPException(
            status_code=503,
            detail="Inference server not running. Start with ./launch.sh start",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Socket {socket_path} not found. Inference server not running.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IPC error: {e}")


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    """Chat completions endpoint compatible with OpenAI API.

    Sends request to distributed server via file and waits for response.

    Args:
        req: Chat completion request

    Returns:
        ChatResponse: Formatted response with completion and metrics

    Raises:
        HTTPException: If request fails, timeout occurs, or server error
    """
    # Generate conversation_id if not provided
    conversation_id = req.conversation_id or str(uuid.uuid4())

    # Extract prompt from last user message
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = req.messages[-1].content
    logger.info(f"Received request: {prompt[:50]}... (conversation_id={conversation_id})")

    # Check if streaming is requested (not supported yet)
    if req.stream:
        raise HTTPException(status_code=501, detail="Streaming is not currently supported")

    # Send request via Unix domain socket
    request_data = {
        "prompt": prompt,
        "max_tokens": req.max_tokens,
        "conversation_id": conversation_id,
    }

    response_data = await send_request_socket(
        config.paths.socket_path,
        request_data,
        config.performance.request_timeout,
    )

    # Check for error in response
    if response_data is None:
        raise HTTPException(status_code=504, detail="No response from server")
    if "error" in response_data:
        error_msg = response_data.get("error", "Unknown error")
        logger.error(f"Server returned error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Server error: {error_msg}")

    # Format response following OpenAI format
    response = ChatResponse(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model=config.model.repo,
        conversation_id=conversation_id,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_data.get("response", "")},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": response_data.get("prompt_tokens", 0),
            "completion_tokens": response_data.get("generated_tokens", 0),
            "total_tokens": (
                response_data.get("prompt_tokens", 0) + response_data.get("generated_tokens", 0)
            ),
        },
        performance={
            "prompt_eval_tokens_per_second": response_data.get("prompt_eval_tokens_per_second", 0),
            "eval_tokens_per_second": response_data.get("eval_tokens_per_second", 0),
            "cache_hit": response_data.get("cache_hit", False),
        },
    )

    usage = response.usage or {}
    perf = response.performance or {}
    logger.info(
        f"Request completed: {usage.get('completion_tokens', 0)} tokens @ "
        f"{perf.get('eval_tokens_per_second', 0):.1f} tok/s"
    )

    return response


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors.

    Args:
        request: The request that caused the error
        exc: The ValueError exception

    Returns:
        JSONResponse with error details
    """
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(status_code=400, content={"error": "Validation error", "detail": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors.

    Args:
        request: The request that caused the error
        exc: The exception

    Returns:
        JSONResponse with error details
    """
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


def main():
    """Start the API server."""
    logger.info("=" * 60)
    logger.info("MLX Distributed Inference API Server")
    logger.info("=" * 60)
    logger.info(f"Starting API server on {config.api.host}:{config.api.port}")
    logger.info(f"Request timeout: {config.performance.request_timeout}s")
    logger.info(f"CORS enabled: {config.api.enable_cors}")
    logger.info("")
    logger.info("This API communicates with the distributed MLX server via Unix socket:")
    logger.info(f"  - Socket: {config.paths.socket_path}")
    logger.info("")
    logger.info("Make sure the distributed server is running with './launch.sh start'")
    logger.info("=" * 60)

    uvicorn.run(
        app, host=config.api.host, port=config.api.port, log_level=config.logging.level.lower()
    )


if __name__ == "__main__":
    main()
