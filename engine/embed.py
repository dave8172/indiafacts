"""Meaning search: every passage as a vector from EmbeddingGemma (300M, multilingual), beside the word search.

Word search finds exact names and numbers ("Article 352", "Nizam"); vectors find the same idea in other words,
or in another language (a question in Hindi finds English passages). Answers use both.

The heavy part, embedding the whole corpus, runs once on any machine with a GPU; the server only embeds each
question (~60 ms on one CPU core). Both sides run the same ONNX export of the same pinned model, so their vectors
agree closely; `check` measures how closely.

  server:  python -m engine embed export            passages without a current vector -> data/passages.jsonl
  GPU box: python -m engine embed run data/passages.jsonl [--provider dml|cuda|cpu]   -> data/vectors-new.npz
  server:  python -m engine embed merge data/vectors-new.npz   -> data/vectors.npz (+ vectors.json manifest)
  server:  python -m engine embed check [--n 40]    re-embed a sample here, compare with the stored vectors
  server:  python -m engine embed search "<question>"

The GPU box needs only requirements-embed.txt (no database, no API key).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VECTORS = DATA / "vectors.npz"
MANIFEST = DATA / "vectors.json"

REPO = "onnx-community/embeddinggemma-300m-ONNX"
REVISION = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
DIM = 768
MAX_TOKENS = 512
QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: {title} | text: "
# The server embeds questions with the 4-bit export: 542 MB resident and ~60 ms a question on one core, where the
# int8 export expands its 262k-token embedding table at run time (1.4 GB, ~0.3-0.7 s). Measured 2026-10-17.
QUERY_VARIANT = os.environ.get("IF_EMBED_VARIANT", "model_q4")


def text_hash(title, text):
    return hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()[:16]


class Encoder:
    def __init__(self, variant=QUERY_VARIANT, provider="cpu", threads=None):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        d = snapshot_download(REPO, revision=REVISION, allow_patterns=[f"onnx/{variant}.onnx*", "tokenizer.json"],
                              local_dir=os.environ.get("IF_MODEL_DIR", str(Path.home() / ".cache" / "indiafacts" / "embeddinggemma")))
        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding()
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        providers = {"cpu": ["CPUExecutionProvider"], "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                     "dml": ["DmlExecutionProvider", "CPUExecutionProvider"]}[provider]
        self.session = ort.InferenceSession(os.path.join(d, "onnx", f"{variant}.onnx"), so, providers=providers)
        self.variant = variant

    def encode(self, texts):
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        v = self.session.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": mask})[0].astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def query(self, text):
        return self.encode([QUERY_PROMPT + text])[0]


def doc_text(title, text):
    return DOC_PROMPT.format(title=title) + text


# --- server side ----------------------------------------------------------------------------------------

def _load():
    if not VECTORS.exists():
        return None
    z = np.load(VECTORS)
    return {"ids": z["ids"], "hashes": z["hashes"], "vecs": z["vecs"]}


def export():
    from . import store
    have = _load()
    known = dict(zip(have["ids"].tolist(), have["hashes"].tolist())) if have else {}
    with store.connect() as c:
        rows = c.execute("""select p.id, p.text, s.title from indiafacts.passages p
                            join indiafacts.sources s on s.slug = p.source order by p.id""").fetchall()
    out, n = DATA / "passages.jsonl", 0
    with out.open("w") as f:
        for r in rows:
            h = text_hash(r["title"], r["text"])
            if known.get(r["id"]) != h:
                f.write(json.dumps({"id": r["id"], "h": h, "text": doc_text(r["title"], r["text"])}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} of {len(rows)} passages need vectors -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def merge(path):
    from . import store
    new = np.load(path)
    have = _load()
    with store.connect() as c:
        live = {r["id"] for r in c.execute("select id from indiafacts.passages").fetchall()}
    table = {}
    if have:
        for i, h, v in zip(have["ids"].tolist(), have["hashes"].tolist(), have["vecs"]):
            table[i] = (h, v)
    for i, h, v in zip(new["ids"].tolist(), new["hashes"].tolist(), new["vecs"]):
        table[i] = (h, v)
    ids = sorted(i for i in table if i in live)  # passages deleted since are dropped
    np.savez(VECTORS, ids=np.array(ids, dtype=np.int64), hashes=np.array([table[i][0] for i in ids]),
             vecs=np.stack([table[i][1] for i in ids]).astype(np.float16))
    MANIFEST.write_text(json.dumps({"repo": REPO, "revision": REVISION, "dim": DIM, "query_prompt": QUERY_PROMPT,
                                    "doc_prompt": DOC_PROMPT, "count": len(ids), "doc_variant": str(new["variant"]),
                                    "merged": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    print(f"{len(ids)} vectors in {VECTORS} ({len(new['ids'])} new or changed, {len(live) - len(ids)} passages still without one)")


_INDEX, _ENCODER = None, None


def search(question, k=40):
    """[(passage id, cosine)] best first; [] when no vectors are built yet."""
    global _INDEX, _ENCODER
    if _INDEX is None:
        z = _load()
        if z is None:
            return []
        _INDEX = (z["ids"], z["vecs"].astype(np.float32))  # float32: numpy's float16 matmul is ~20x slower
    if _ENCODER is None:
        _ENCODER = Encoder()
    ids, vecs = _INDEX
    scores = vecs @ _ENCODER.query(question)
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    top = top[np.argsort(-scores[top])]
    return [(int(ids[i]), float(scores[i])) for i in top]


def check(n=40):
    """Re-embed n stored passages here with the server's model; report how close they are to the GPU's vectors."""
    from . import store
    z = _load()
    pick = np.random.default_rng(0).choice(len(z["ids"]), size=min(n, len(z["ids"])), replace=False)
    want = [int(z["ids"][i]) for i in pick]
    with store.connect() as c:
        rows = {r["id"]: r for r in c.execute("""select p.id, p.text, s.title from indiafacts.passages p
                                                join indiafacts.sources s on s.slug = p.source
                                                where p.id = any(%s)""", (want,)).fetchall()}
    enc = Encoder()
    here = np.vstack([enc.encode([doc_text(rows[i]["title"], rows[i]["text"])]) for i in want])
    cos = np.sum(here * z["vecs"][pick].astype(np.float32), axis=1)
    print(f"{len(want)} passages: cosine server vs stored min {cos.min():.4f}, mean {cos.mean():.4f}")


# --- GPU side --------------------------------------------------------------------------------------------

def run(path, provider="dml", variant="model", batch=16, out=None):
    """Embed a passages.jsonl from `export`, longest-first batches (less padding). Progress is saved to
    <out>.part every ~2,000 passages; running the same command again resumes from it."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: (-len(r["text"]), r["id"]))
    out = Path(out or Path(path).with_name("vectors-new.npz"))
    part = out.with_suffix(".part.npz")
    vecs, done = [], 0
    if part.exists():
        z = np.load(part)
        if str(z["variant"]) == variant and z["ids"].tolist() == [r["id"] for r in rows[:len(z["ids"])]]:
            vecs, done = [z["vecs"]], len(z["ids"])
            print(f"resuming after {done} passages", flush=True)
    enc = Encoder(variant=variant, provider=provider)
    print(f"{len(rows)} passages on {enc.session.get_providers()[0]}, variant {variant}, batch {batch}", flush=True)
    t0, start = time.time(), done

    def save(target, n):
        np.savez(target, ids=np.array([r["id"] for r in rows[:n]], dtype=np.int64),
                 hashes=np.array([r["h"] for r in rows[:n]]), vecs=np.vstack(vecs), variant=np.array(variant))

    while done < len(rows):
        chunk = rows[done:done + batch]
        vecs.append(enc.encode([r["text"] for r in chunk]).astype(np.float16))
        done += len(chunk)
        if (done // batch) % 125 == 0 or done == len(rows):
            rate = (done - start) / (time.time() - t0)
            print(f"  {done}/{len(rows)}  {rate:.1f}/s  ~{(len(rows) - done) / rate / 60:.0f} min left", flush=True)
            save(part, done)
            vecs = [np.vstack(vecs)]
    save(out, done)
    part.unlink(missing_ok=True)
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB) in {(time.time() - t0) / 60:.1f} min")


def main(args):
    cmd = args[0] if args else ""
    opt = lambda name, default: args[args.index(name) + 1] if name in args else default
    if cmd == "export":
        export()
    elif cmd == "run":
        run(args[1], provider=opt("--provider", "dml"), variant=opt("--variant", "model"), batch=int(opt("--batch", 16)),
            out=opt("--out", None))
    elif cmd == "merge":
        merge(args[1])
    elif cmd == "check":
        check(int(opt("--n", 40)))
    elif cmd == "search":
        from . import store
        hits = search(" ".join(args[1:]), k=8)
        with store.connect() as c:
            text = {r["id"]: r for r in c.execute("select id, source, text from indiafacts.passages where id = any(%s)",
                                                  ([i for i, _ in hits],)).fetchall()}
        for i, s in hits:
            print(f"[{s:.3f}] {text[i]['source']}: {text[i]['text'][:160]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
