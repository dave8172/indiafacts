"""The Anthropic client, prices, and the spend ledger.

Use a key of its own (ANTHROPIC_API_KEY, or a file at IF_ANTHROPIC_KEY_FILE), ideally in its own Console
workspace with a spend limit, so a runaway here can't empty a budget other apps share.
"""
import json
import os
import time
from pathlib import Path

import anthropic

KEY_FILE = os.environ.get("IF_ANTHROPIC_KEY_FILE", "")
LEDGER = Path(__file__).resolve().parent.parent / "runs" / "spend.jsonl"

ANSWER_MODEL = "claude-opus-5-5"
WORKER_MODEL = "claude-haiku-5-5"  # planning and bulk topic tagging

# $ per million tokens: input, output, cache read, cache write (5 min)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
WEB_SEARCH = 10.00 / 1000  # per search


def client():
    key = os.environ.get("ANTHROPIC_API_KEY") or Path(KEY_FILE).read_text().strip()
    return anthropic.Anthropic(api_key=key, max_retries=3)


def cost(model, usage, batch=False):
    pin, pout, pread, pwrite = PRICES[model]
    u = usage if isinstance(usage, dict) else usage.model_dump()
    c = (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout
         + (u.get("cache_read_input_tokens") or 0) * pread + (u.get("cache_creation_input_tokens") or 0) * pwrite) / 1e6
    searches = ((u.get("server_tool_use") or {}).get("web_search_requests") or 0)
    return c * (0.5 if batch else 1) + searches * WEB_SEARCH


def record(kind, dollars, **extra):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"t": round(time.time()), "kind": kind, "usd": round(dollars, 5), **extra}) + "\n")


def spent(kind=None, since=0):
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text().splitlines():
        r = json.loads(line)
        if r["t"] >= since and (kind is None or r["kind"] == kind):
            total += r["usd"]
    return total
