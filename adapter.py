"""
adapter.py

OpenAI-compatible /v1/chat/completions adapter in front of Lili's
POST /api/user-scope/website-chat/ endpoint.

Supports:
- Non-streaming JSON response
- Streaming SSE response (adapter-level streaming by chunking final text)

Environment variables:
- LILI_API_BASE (default: https://backend-lili-demo.limitless-tech.ai/api)
- LILI_WORKFLOW_ID (default: 213)
- LILI_TIMEOUT_SECONDS (default: 60)
- STREAM_CHUNK_SIZE (default: 40)  # characters per SSE chunk
"""
from fastapi.responses import HTMLResponse
import json
import os
import time
import uuid
from typing import List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="BP PoC Adapter", version="1.0.0")

LILI_API_BASE = os.getenv("LILI_API_BASE", "https://backend-lili-demo.limitless-tech.ai/api").rstrip("/")
LILI_WORKFLOW_ID = os.getenv("LILI_WORKFLOW_ID", "213")
LILI_TIMEOUT_SECONDS = float(os.getenv("LILI_TIMEOUT_SECONDS", "60"))
STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "40"))

LILI_ENDPOINT = f"{LILI_API_BASE}/user-scope/website-chat/"


# -----------------------------
# Pydantic models (OpenAI-like)
# -----------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str = "lili-workflow"
    messages: List[ChatMessage] = Field(min_length=1)
    stream: bool = False
    user: Optional[str] = None  # optional stable session id


# -----------------------------
# Helpers
# -----------------------------
def last_user_message(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


def chunk_text(text: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "lili_endpoint": LILI_ENDPOINT,
        "workflow_id": LILI_WORKFLOW_ID,
    }


# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionsRequest):
    user_text = last_user_message(body.messages).strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="No user message found in messages[]")

    # If you want stable conversation memory in Lili across turns,
    # pass a consistent `user` from the caller; otherwise this will be random per request.
    sender_id = (body.user or str(uuid.uuid4())).strip()

    payload = {
        "workflow_id": str(LILI_WORKFLOW_ID),
        "sender_id": sender_id,
        "user_message": user_text,
    }

    # Call Lili (non-streaming upstream)
    try:
        async with httpx.AsyncClient(timeout=LILI_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                LILI_ENDPOINT,
                json=payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream Lili request failed: {str(e)}") from e

    if resp.status_code >= 400:
        # Return a 502 because the adapter is functioning but upstream failed
        raise HTTPException(status_code=502, detail=f"Lili error {resp.status_code}: {resp.text}")

    # Parse response
    try:
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Lili returned non-JSON: {resp.text}") from e

    assistant_text = (data.get("message") or data.get("error") or "").strip()

    created = int(time.time())
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Non-streaming response
    if not body.stream:
        return JSONResponse(
            {
                "id": stream_id,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": assistant_text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    # Streaming SSE response (adapter-level streaming)
    async def sse_gen():
        # Some clients like to see role once; harmless to include in first event.
        first = True
        for part in chunk_text(assistant_text, STREAM_CHUNK_SIZE):
            delta_obj = {"content": part}
            if first:
                delta_obj["role"] = "assistant"
                first = False

            event = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta_obj,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # Final stop chunk
        final_event = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_gen(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BP PoC Chat</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; max-width: 900px; margin: 24px auto; padding: 0 16px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; }
    #chat { height: 60vh; overflow: auto; border: 1px solid #eee; border-radius: 10px; padding: 12px; background: #fafafa; }
    .row { margin: 10px 0; }
    .user { text-align: right; }
    .msg { display: inline-block; padding: 10px 12px; border-radius: 14px; max-width: 80%; white-space: pre-wrap; }
    .user .msg { background: #dbeafe; }
    .assistant .msg { background: #e5e7eb; }
    textarea { width: 100%; height: 70px; padding: 10px; border-radius: 10px; border: 1px solid #ddd; }
    button { padding: 10px 14px; border: 0; border-radius: 10px; cursor: pointer; }
    .bar { display:flex; gap: 10px; align-items: center; margin-top: 10px; }
    .bar button { background:#111827; color:white; }
    .muted { color:#6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <h2>BP PoC Chat</h2>
  <div class="card">
    <div class="muted">This chat uses Lili via the adapter deployed on this Railway service.</div>
    <div id="chat"></div>

    <div class="bar">
      <textarea id="input" placeholder="Type your message..."></textarea>
    </div>
    <div class="bar">
      <button id="send">Send</button>
      <button id="clear" style="background:#6b7280;">Clear</button>
      <label class="muted"><input type="checkbox" id="stream" checked /> Stream</label>
    </div>
  </div>

<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const clearBtn = document.getElementById('clear');
const streamCb = document.getElementById('stream');

let history = []; // OpenAI-style messages

function addMessage(role, text) {
  const row = document.createElement('div');
  row.className = 'row ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'msg';
  bubble.textContent = text;
  row.appendChild(bubble);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  sendBtn.textContent = busy ? 'Sending...' : 'Send';
}

function parseSSEChunk(buffer) {
  // returns {events: [dataString...], restBuffer}
  const events = [];
  let idx;
  while ((idx = buffer.indexOf("\\n\\n")) !== -1) {
    const rawEvent = buffer.slice(0, idx);
    buffer = buffer.slice(idx + 2);
    const lines = rawEvent.split("\\n");
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        events.push(line.slice(6));
      }
    }
  }
  return { events, rest: buffer };
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = '';
  addMessage('user', text);
  history.push({ role: 'user', content: text });

  setBusy(true);

  const assistantBubble = addMessage('assistant', '');

  const body = {
    model: 'test',
    messages: history,
    stream: !!streamCb.checked
  };

  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      const t = await res.text();
      assistantBubble.textContent = 'Error: ' + t;
      setBusy(false);
      return;
    }

    if (!body.stream) {
      const data = await res.json();
      const content = data?.choices?.[0]?.message?.content ?? '';
      assistantBubble.textContent = content;
      history.push({ role: 'assistant', content });
      setBusy(false);
      return;
    }

    // Streaming SSE
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    let full = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });
      const parsed = parseSSEChunk(buf);
      buf = parsed.rest;

      for (const dataLine of parsed.events) {
        if (dataLine === '[DONE]') {
          history.push({ role: 'assistant', content: full });
          setBusy(false);
          return;
        }
        try {
          const obj = JSON.parse(dataLine);
          const delta = obj?.choices?.[0]?.delta;
          if (delta && typeof delta.content === 'string') {
            full += delta.content;
            assistantBubble.textContent = full;
            chatEl.scrollTop = chatEl.scrollHeight;
          }
        } catch (e) {
          // ignore non-JSON lines
        }
      }
    }

    // If stream ended without [DONE]
    history.push({ role: 'assistant', content: full });
    setBusy(false);

  } catch (err) {
    assistantBubble.textContent = 'Error: ' + (err?.message || String(err));
    setBusy(false);
  }
}

sendBtn.addEventListener('click', sendMessage);
clearBtn.addEventListener('click', () => {
  chatEl.innerHTML = '';
  history = [];
  inputEl.value = '';
});
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
</script>
</body>
</html>
"""

