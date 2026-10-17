"""indiafacts as a small HTTP service. Stdlib only.

Meant to sit behind a gateway (the live site reaches it through a Cloudflare Tunnel behind Access). Every request
must also carry `X-Engine-Key` (env IF_ENGINE_KEY).

  POST /indiafacts/ask {question} -> {ok, segments: [{text, cites, unsourced}], sources: [{n, title, author, year,
                                     locator, standing, quote, url}], topics, related, meta: {usd, seconds}}
                                   | {ok: false, reason: invalid | budget | busy | refused | error}
  GET  /indiafacts/health        -> {ok, sources, passages, topics, edges, today_usd, cap_usd}

Nothing about the asker is read, and the question is not stored: only each answer's cost goes in the ledger
(runs/spend.jsonl). Public spend stops at IF_DAILY_USD a day (UTC).
"""
import hmac
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import ask, llm, store

KEY = os.environ.get("IF_ENGINE_KEY", "")
DAILY_USD = float(os.environ.get("IF_DAILY_USD", "3"))
SLOTS = threading.BoundedSemaphore(int(os.environ.get("IF_PARALLEL", "2")))


def _today_start():
    t = time.gmtime()
    return int(time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, 0)) - time.timezone)


class Handler(BaseHTTPRequestHandler):
    server_version = "indiafacts"

    def log_message(self, fmt, *args):  # no request lines in the logs: they would carry nothing useful, and no IPs
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorised(self):
        return bool(KEY) and hmac.compare_digest(self.headers.get("X-Engine-Key", ""), KEY)

    def do_GET(self):
        if not self._authorised():
            return self._send(401, {"ok": False})
        if self.path == "/indiafacts/health":
            return self._send(200, {"ok": True, **store.stats(), "today_usd": round(llm.spent("answer", _today_start()), 3),
                                    "cap_usd": DAILY_USD})
        self._send(404, {"ok": False})

    def do_POST(self):
        if not self._authorised():
            return self._send(401, {"ok": False})
        if self.path != "/indiafacts/ask":
            return self._send(404, {"ok": False})
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 4096)) or b"{}")
        except json.JSONDecodeError:
            body = {}
        q = " ".join(str(body.get("question") or "").split())
        if not 8 <= len(q) <= 400:
            return self._send(200, {"ok": False, "reason": "invalid"})
        if llm.spent("answer", _today_start()) >= DAILY_USD:
            return self._send(200, {"ok": False, "reason": "budget"})
        if not SLOTS.acquire(timeout=20):
            return self._send(200, {"ok": False, "reason": "busy"})
        try:
            out = ask.answer(q)
            out["meta"] = {k: out["meta"][k] for k in ("usd", "seconds")}  # the plan stays internal
            self._send(200, out)
        except Exception:
            traceback.print_exc()
            self._send(200, {"ok": False, "reason": "error"})
        finally:
            SLOTS.release()


def main():
    port = int(os.environ.get("PORT", "8080"))
    print(f"indiafacts on :{port}, cap ${DAILY_USD}/day", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
