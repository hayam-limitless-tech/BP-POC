import json
import sys
import requests

url = "https://bp-poc-production.up.railway.app/v1/chat/completions"
payload = {
    "model": "test",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": True,
}

with requests.post(url, json=payload, stream=True, timeout=120) as r:
    r.raise_for_status()
    for raw in r.iter_lines(decode_unicode=True):
        if not raw:
            continue
        print(raw)
        sys.stdout.flush()
