"""Widen web quotes: a web search citation quotes at most ~150 characters of its page (measured 2026-10-17), too
little to check a sentence on a court ruling against. So each cited page is fetched once more, through the API's
web fetch tool (court and Gazette servers time out for server addresses; the API's fetcher gets through), the
short quote is found in it, and the quote is replaced by ~900 characters around it. A page that can't be fetched,
or that doesn't contain the short quote, keeps the short quote: every quote shown is still text on the page.
A signed court PDF prints its e-signature stamp and page number across the text at every page break; they are
taken out first, so a widened quote doesn't carry "Signature Not Verified ... Page 70 of 82" mid-sentence.

Pages are fetched from here first (free, and whole: the API's fetcher cuts a long judgment off at its token cap).
"""
import base64
import io
import re
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from html.parser import HTMLParser
from urllib.parse import urlparse

from . import llm
from .corpus import UA

WIDTH = 900
DEADLINE = 10  # seconds for the whole step; a page not in by then keeps its short quote
BEFORE = 200  # how far back to look for the start of the quote's sentence
# The e-signature stamp of a signed court PDF (High Court: "Signing Date:", Supreme Court: "Reason:"), with the
# case number and page number printed after it, and a page number on a line of its own.
FURNITURE = re.compile(r"Signature Not Verified.{0,160}?(?:Reason:|Signing Date:\s*[\d.]+(?:\s+\d\d:\d\d:\d\d)?)"
                       r"(?:[^\n]{0,120}?Page \d+ of \d+)?|^[ \t]*Page \d+ of \d+[ \t]*$", re.S | re.M)
_SLOW = set()  # hosts that timed out from here (sci.gov.in, egazette): straight to the API's fetcher


def _norm(text):
    """Letters and digits only, lower case, with each kept character's position in `text`."""
    keep = [(c.lower(), i) for i, c in enumerate(text) if c.isalnum()]
    return "".join(c for c, _ in keep), [i for _, i in keep]


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "pre", "blockquote", "section"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        self.skip += tag in ("script", "style", "noscript")
        if tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        self.skip -= tag in ("script", "style", "noscript") and self.skip > 0

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def _pdf_text(data):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)


def _direct(url):
    host = urlparse(url).netloc
    if host in _SLOW:
        return None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=4) as r:
            data, kind = r.read(20_000_000), r.headers.get_content_type()
        if kind == "application/pdf" or data[:5] == b"%PDF-":
            return _pdf_text(data)
        if kind == "text/html":
            t = _Text()
            t.feed(data.decode(r.headers.get_content_charset() or "utf-8", "replace"))
            return "".join(t.out)
    except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
        if isinstance(e, (TimeoutError, socket.timeout)) or isinstance(getattr(e, "reason", None), (TimeoutError, socket.gaierror)):
            _SLOW.add(host)
    except Exception:
        pass
    return None


def _page_text(cl, url):
    """(text, cost); text is None when the page can't be fetched or read."""
    text = _direct(url)
    if text:
        return text, 0.0
    try:
        msg = cl.messages.create(
            model=llm.WORKER_MODEL, max_tokens=300, thinking={"type": "disabled"},
            tools=[{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1, "max_content_tokens": 60000}],
            messages=[{"role": "user", "content": f"Fetch this page with web_fetch, then reply only: done.\n{url}"}])
    except Exception:  # e.g. a 400 when the fetched "PDF" is an error page
        return None, 0.0
    cost = llm.cost(llm.WORKER_MODEL, msg.usage)
    for b in msg.content:
        c = b.model_dump().get("content") if b.type == "web_fetch_tool_result" else None
        if not c or c.get("type") != "web_fetch_result":
            continue
        src = c["content"]["source"]
        if src["type"] == "text":
            return src["data"], cost
        if src.get("media_type") == "application/pdf":
            try:
                return _pdf_text(base64.b64decode(src["data"])), cost
            except Exception:
                return None, cost
    return None, cost


def _around(text, quote):
    """~WIDTH characters of `text` starting at the sentence that holds `quote`, or None if it isn't there."""
    q, _ = _norm(re.sub(r"\.{3}$|…$", "", quote.strip()))
    if len(q) < 20:
        return None
    t, pos = _norm(text)
    at = t.find(q)
    if at < 0:
        at = t.find(q[:60])
        if at < 0:
            return None
    start = pos[at]
    dot = max(text.rfind(". ", max(0, start - BEFORE), start), text.rfind("\n\n", max(0, start - BEFORE), start))
    if dot >= 0:
        start = dot + 2
    end = pos[min(at + len(q), len(pos)) - 1] + 1
    stop = text.rfind(". ", end, start + WIDTH)
    end = stop + 1 if stop >= 0 else max(end, min(len(text), start + WIDTH))
    wide = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text[start:end])  # markdown links from the API's fetcher
    return re.sub(r"\s+", " ", wide.replace("**", "")).strip()


def widen(cl, sources):
    """Replace each web source's short quote with the text around it on its page. Returns the cost."""
    web = [s for s in sources if s.get("web")]
    if not web:
        return 0.0
    urls = list(dict.fromkeys(s["url"] for s in web))
    ex = ThreadPoolExecutor(min(4, len(urls)))
    futures = {ex.submit(_page_text, cl, u): u for u in urls}
    done, late = wait(futures, timeout=DEADLINE)
    ex.shutdown(wait=False)
    for f in late:  # a long PDF through the API's fetcher can take 30 s: answer without it, still count its cost
        f.add_done_callback(lambda f: llm.record("answer", f.result()[1], late="widen"))  # "answer": the daily cap counts it
    pages = {futures[f]: f.result() for f in done}
    for s in web:
        text = pages.get(s["url"], (None, 0))[0]
        wide = text and _around(FURNITURE.sub(" ", text), s["quote"])
        if wide and len(wide) > len(s["quote"]):
            s["quote"], s["widened"] = wide, True
    return sum(c for _, c in pages.values())
