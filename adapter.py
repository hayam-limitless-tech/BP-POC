import os
import time
import uuid
from typing import List, Optional, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

LILI_API_BASE = os.getenv("LILI_API_BASE", "https://backend-lili-demo.limitless-tech.ai/api")
LILI_WORKFLOW_ID = os.getenv("LILI_WORKFLOW_ID", "213")
LILI_ENDPOINT = f"{LILI_API_BASE}/user-scope/website-chat/"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str = "lili-workflow-213"
    messages: List[ChatMessage] = Field(min_length=1)
    stream: bool = False
    user: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionsRequest):
    user_text = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
    if not user_text:
        raise HTTPException(status_code=400, detail="No user message found")

    sender_id = body.user or str(uuid.uuid4())

    payload = {
        "workflow_id": LILI_WORKFLOW_ID,
        "sender_id": sender_id,
        "user_message": user_text,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(LILI_ENDPOINT, json=payload)

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=r.text)

    data = r.json()
    assistant_text = data.get("message", "")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": "stop",
            }
        ],
    }
