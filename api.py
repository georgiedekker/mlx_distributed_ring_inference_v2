#!/usr/bin/env python3
"""
API wrapper for the distributed MLX server.
Communicates via files with the distributed inference process.
"""

import json
import os
import time
import uuid
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "DeepSeek"
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    conversation_id: str

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api": "ready"
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """
    Chat completions endpoint.
    Sends request to distributed server via file.
    """
    import asyncio
    
    # Generate conversation_id if not provided
    conversation_id = req.conversation_id or str(uuid.uuid4())
    
    # Extract prompt
    prompt = req.messages[-1].content if req.messages else ""
    
    # Write request file
    request_file = "/tmp/mlx_request.json"
    response_file = "/tmp/mlx_response.json"
    
    # Remove old response if exists
    if os.path.exists(response_file):
        os.remove(response_file)
    
    # Write request with conversation_id
    with open(request_file, 'w') as f:
        json.dump({
            "prompt": prompt,
            "max_tokens": req.max_tokens,
            "conversation_id": conversation_id
        }, f)
    
    print(f"Request sent: {prompt[:50]}...")
    
    # Wait for response (with timeout)
    timeout = 60  # seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if os.path.exists(response_file):
            # Read response
            with open(response_file, 'r') as f:
                response_data = json.load(f)
            
            # Clean up
            os.remove(response_file)
            
            # Return formatted response with metrics
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
                }]
            )
            
            # Add performance metrics as additional fields
            response_dict = response.model_dump()
            response_dict["usage"] = {
                "prompt_tokens": response_data.get("prompt_tokens", 0),
                "completion_tokens": response_data.get("generated_tokens", 0),
                "total_tokens": response_data.get("prompt_tokens", 0) + response_data.get("generated_tokens", 0)
            }
            response_dict["performance"] = {
                "prompt_eval_tokens_per_second": response_data.get("prompt_eval_tokens_per_second", 0),
                "eval_tokens_per_second": response_data.get("eval_tokens_per_second", 0)
            }
            
            return response_dict
        
        await asyncio.sleep(0.1)
    
    raise HTTPException(status_code=500, detail="Timeout waiting for response")

if __name__ == "__main__":
    print("Starting API server on port 8100...")
    print("This API communicates with the distributed MLX server via files.")
    print("Make sure the distributed server is running with './launch.sh start'")
    uvicorn.run(app, host="0.0.0.0", port=8100)