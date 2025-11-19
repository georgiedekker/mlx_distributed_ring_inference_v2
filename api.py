#!/usr/bin/env python3
"""
API wrapper for the distributed MLX server.

Communicates via files with the distributed inference process.
Provides a REST API compatible with OpenAI's chat completion format.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from config import get_config


# Initialize configuration
config = get_config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format=config.logging.simple_format
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLX Distributed Inference API",
    description="REST API for distributed MLX inference",
    version="1.0.0"
)

# Add CORS middleware if enabled
if config.api.enable_cors:
    app.add_middleware(
        CORSMiddleware,
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

    @validator('role')
    def validate_role(cls, v):
        """Validate that role is one of the allowed values."""
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be one of: user, assistant, system')
        return v

    @validator('content')
    def validate_content(cls, v):
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v


class ChatRequest(BaseModel):
    """Chat completion request following OpenAI format."""
    model: str = Field(default="DeepSeek", description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    max_tokens: int = Field(
        default=100,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (not currently used)"
    )
    stream: bool = Field(default=False, description="Whether to stream responses (not currently supported)")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID for caching"
    )

    @validator('messages')
    def validate_messages(cls, v):
        """Validate that messages list is not empty."""
        if not v:
            raise ValueError('Messages list cannot be empty')
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
        "config": {
            "port": config.api.port,
            "cors_enabled": config.api.enable_cors
        }
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
            "chat": "/v1/chat/completions"
        }
    }


def check_server_running() -> bool:
    """Check if the distributed server is running.

    Returns:
        bool: True if server appears to be running
    """
    # Simple heuristic: if request file doesn't exist or is stale, server might not be running
    # This is not foolproof but provides basic feedback
    return True  # For now, always return True


async def wait_for_response_file(
    response_file: str,
    timeout: int,
    poll_interval: float = 0.1
) -> Optional[Dict]:
    """Wait for response file to be created and read it.

    Args:
        response_file: Path to the response file
        timeout: Maximum time to wait in seconds
        poll_interval: How often to check for file in seconds

    Returns:
        Response data dictionary, or None if timeout

    Raises:
        HTTPException: If file read fails or timeout occurs
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if os.path.exists(response_file):
            try:
                # Small delay to ensure file is fully written
                await asyncio.sleep(0.05)

                with open(response_file, 'r') as f:
                    response_data = json.load(f)

                # Clean up response file
                try:
                    os.remove(response_file)
                except (IOError, OSError) as e:
                    logger.warning(f"Failed to remove response file: {e}")

                return response_data

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response JSON: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Invalid response from server"
                )
            except (IOError, OSError) as e:
                logger.error(f"Failed to read response file: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to read response: {str(e)}"
                )

        await asyncio.sleep(poll_interval)

    # Timeout occurred
    logger.error(f"Timeout waiting for response (timeout={timeout}s)")
    raise HTTPException(
        status_code=504,
        detail=f"Request timeout after {timeout} seconds. Server may be overloaded or not running."
    )


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
        raise HTTPException(
            status_code=501,
            detail="Streaming is not currently supported"
        )

    request_file = config.paths.request_file
    response_file = config.paths.response_file

    # Remove old response file if exists
    if os.path.exists(response_file):
        try:
            os.remove(response_file)
        except (IOError, OSError) as e:
            logger.warning(f"Failed to remove old response file: {e}")

    # Write request file
    request_data = {
        "prompt": prompt,
        "max_tokens": req.max_tokens,
        "conversation_id": conversation_id
    }

    try:
        with open(request_file, 'w') as f:
            json.dump(request_data, f)
        logger.debug(f"Request written to {request_file}")
    except (IOError, OSError) as e:
        logger.error(f"Failed to write request file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write request: {str(e)}"
        )

    # Wait for response with timeout
    try:
        response_data = await wait_for_response_file(
            response_file,
            config.performance.request_timeout,
            config.performance.poll_interval
        )
    except HTTPException:
        # Clean up request file if it still exists
        if os.path.exists(request_file):
            try:
                os.remove(request_file)
            except (IOError, OSError):
                pass
        raise

    # Check for error in response
    if "error" in response_data:
        error_msg = response_data.get("error", "Unknown error")
        logger.error(f"Server returned error: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {error_msg}"
        )

    # Format response following OpenAI format
    response = ChatResponse(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model=req.model,
        conversation_id=conversation_id,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_data.get("response", "")
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": response_data.get("prompt_tokens", 0),
            "completion_tokens": response_data.get("generated_tokens", 0),
            "total_tokens": (
                response_data.get("prompt_tokens", 0) +
                response_data.get("generated_tokens", 0)
            )
        },
        performance={
            "prompt_eval_tokens_per_second": response_data.get("prompt_eval_tokens_per_second", 0),
            "eval_tokens_per_second": response_data.get("eval_tokens_per_second", 0),
            "cache_hit": response_data.get("cache_hit", False)
        }
    )

    logger.info(
        f"Request completed: {response.usage['completion_tokens']} tokens @ "
        f"{response.performance['eval_tokens_per_second']:.1f} tok/s"
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
    return JSONResponse(
        status_code=400,
        content={"error": "Validation error", "detail": str(exc)}
    )


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
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
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
    logger.info("This API communicates with the distributed MLX server via files:")
    logger.info(f"  - Request file: {config.paths.request_file}")
    logger.info(f"  - Response file: {config.paths.response_file}")
    logger.info("")
    logger.info("Make sure the distributed server is running with './launch.sh start'")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level=config.logging.level.lower()
    )


if __name__ == "__main__":
    main()
