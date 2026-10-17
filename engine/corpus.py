"""The corpus: fetch each source's full text, clean its OCR, cut it into passages.

archive.org's file servers refuse this box (TLS reset on *.archive.org data nodes, 2026-10-17), so the text
comes from the `/stream/` page, which the main host serves: the whole OCR text in one <pre>. It carries no page
breaks, so a passage is located by what the source itself numbers (article, section, paragraph, sitting date)
and by a search-inside link that opens the scan at the quote.

A passage is ~1,200 characters of whole paragraphs. Its text is also split into blocks of one or two sentences:
Claude cites whole blocks, so small blocks give tight quotes.
"""
import html
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SOURCES = ROOT / "sources.json"
UA = {"User-Agent": "indiafacts/0.1 (quirkyagents.com; hello@quirkyagents.com)"}

PASSAGE_CHARS = 1200
BLOCK_CHARS = 260


def sources():
    return json.loads(SOURCES.read_text())["sources"]


def standings():
    return json.loads(SOURCES.read_text())["standings"]


def _get(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def fetch(src, force=False):
    """The source's OCR text, cached in data/raw/<slug>.txt."""
    out = RAW / f"{src['slug']}.txt"
    if out.exists() and not force:
        return out.read_text()
    meta = json.loads(_get(f"https://archive.org/metadata/{src['ia']}"))
    if meta.get("metadata", {}).get("access-restricted-item"):
        raise RuntimeError(f"{src['ia']} is a restricted (lending) item")
    texts = [f for f in meta.get("files", []) if f["name"].endswith("_djvu.txt")]
    if not texts:
        raise RuntimeError(f"{src['ia']} has no OCR text")
    # Some items split a report into parts (PART1-A, PART1-B...; Vol 1, Vol 2): join them in name order.
    parts = []
    for f in sorted(texts, key=lambda f: f["name"]):
        page = _get(f"https://archive.org/stream/{src['ia']}/{urllib.parse.quote(f['name'])}", timeout=600).decode("utf-8", "replace")
        m = re.search(r"<pre>(.*?)</pre>", page, re.S)
        if not m:
            raise RuntimeError(f"{src['ia']}: no text in the stream page for {f['name']}")
        parts.append(html.unescape(m.group(1)))
    text = "\n\n".join(parts)
    RAW.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text


# --- cleaning -------------------------------------------------------------------------------------------

def _clean_paragraph(p):
    p = re.sub(r"(\w)[-­]\s*\n\s*(\w)", r"\1\2", p)  # word broken across lines
    p = re.sub(r"\s+", " ", p).strip()
    return p


def _looks_like_text(p):
    if len(p) < 40:
        return False
    letters = sum(c.isalpha() for c in p)
    words = p.split()
    return letters / len(p) > 0.6 and sum(len(w) > 1 for w in words) / len(words) > 0.6


def paragraphs(text):
    text = unicodedata.normalize("NFKC", text).replace("\x0c", "\n\n").replace("\r", "")
    for raw in re.split(r"\n\s*\n", text):
        p = _clean_paragraph(raw)
        if _looks_like_text(p):
            yield p


# --- locators: what the source itself numbers -----------------------------------------------------------

_DAYS = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
LOCATORS = {
    # "Monday, the 9th December, 1946"
    "sitting": (re.compile(_DAYS + r",?\s+the\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Z][a-z]+),?\s+(19[45]\d)"),
                lambda m: f"sitting of {m.group(1)} {m.group(2)} {m.group(3)}"),
    # "21. Protection of life and personal liberty.—No person..."
    "article": (re.compile(r"^(\d{1,3}[A-Z]{0,3})\.\s+[A-Z][^.]{3,140}\.\s*[—–-]"),
                lambda m: f"Article {m.group(1)}"),
    "section": (re.compile(r"^(\d{1,3}[A-Z]?)\.\s+(?:\(1\)\s+)?[A-Z][^.]{3,140}[.—–-]"),
                lambda m: f"section {m.group(1)}"),
    # "16.17 The story related by..."
    "para": (re.compile(r"^(\d{1,2}\.\d{1,3})\.?\s+[A-Z(]"), lambda m: f"para {m.group(1)}"),
}


def _sentences(text):
    marked = re.sub(r"([.?!;:][\"'\u201d\u2019)]?)\s+(?=[\"'\u201c\u2018(A-Z0-9])", "\\1\n", text)
    return [s for s in marked.split("\n") if s]


def blocks(text):
    """One or two sentences per block; a block is the smallest piece Claude can quote."""
    out, cur = [], ""
    for s in _sentences(text):
        if cur and len(cur) + len(s) > BLOCK_CHARS:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def passages(src, text):
    """Yield dicts {seq, locator, text, blocks} for one source."""
    kind = src.get("locator", "none")
    pattern, label = LOCATORS.get(kind, (None, None))
    locator, cur, cur_loc, seq = None, [], None, 0

    def emit():
        nonlocal cur, seq
        body = " ".join(cur)
        cur = []
        seq += 1
        return {"seq": seq, "locator": cur_loc, "text": body, "blocks": blocks(body)}

    for p in paragraphs(text):
        if pattern:
            m = pattern.search(p) if kind == "sitting" else pattern.match(p)
            if m:
                if cur and kind != "sitting":
                    yield emit()
                locator = label(m)
        if not cur:
            cur_loc = locator
        # long paragraphs are cut at sentence ends
        pieces = [p] if len(p) <= PASSAGE_CHARS * 1.6 else _split_long(p)
        for piece in pieces:
            if cur and sum(map(len, cur)) + len(piece) > PASSAGE_CHARS:
                yield emit()
                cur_loc = locator
            cur.append(piece)
    if cur:
        yield emit()


def _split_long(p):
    out, cur = [], ""
    for s in _sentences(p):
        if cur and len(cur) + len(s) > PASSAGE_CHARS:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def verify_url(src, quote):
    """A link that opens the scan with the quote searched, so anyone can check it."""
    # six consecutive clean words: search-inside matches a phrase, and OCR slips elsewhere don't break it
    tokens = [t.strip(",.;:()\"'“”‘’") for t in quote.split()]
    q = ""
    for i in range(max(1, len(tokens) - 5)):
        window = tokens[i:i + 6]
        if len(window) == 6 and all(re.fullmatch(r"[A-Za-z]{2,}", w) for w in window):
            q = " ".join(window)
            break
    return f"https://archive.org/details/{src['ia']}" + (f"?q={urllib.parse.quote(chr(34) + q + chr(34))}" if q else "")
