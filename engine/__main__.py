"""indiafacts CLI.

  python -m engine fetch [slug...]     download OCR text into data/raw/ (free)
  python -m engine build [slug...]     clean, cut into passages, load into Postgres (free)
  python -m engine search "<query>"    full-text search, to look at what retrieval returns (free)
  python -m engine graph [--pilot N]   tag passages with topics (paid: Haiku 5.5, batch) and build the graph
  python -m engine ask "<question>"    one answer, printed with its sources and cost (paid)
  python -m engine embed ...           meaning search: export | merge | check | search (see engine/embed.py)
  python -m engine stats
"""
import sys

from . import corpus, store


def _pick(args):
    want = set(args)
    return [s for s in corpus.sources() if not want or s["slug"] in want]


def main(argv):
    cmd, args = (argv[0], argv[1:]) if argv else ("help", [])
    if cmd == "fetch":
        for s in _pick(args):
            try:
                t = corpus.fetch(s)
                print(f"{s['slug']:24} {len(t):>10,} chars")
            except Exception as e:  # one bad source must not stop the rest
                print(f"{s['slug']:24} FAILED {e}")
    elif cmd == "build":
        store.init()
        for s in _pick(args):
            if not (corpus.RAW / f"{s.get('file', s['slug'])}.txt").exists():
                print(f"{s['slug']:24} not fetched")
                continue
            text = corpus.text(s)
            rows = list(corpus.passages(s, text))
            store.load_source(s, rows, len(text))
            located = sum(1 for r in rows if r["locator"])
            print(f"{s['slug']:24} {len(rows):>6} passages, {located:>6} located")
    elif cmd == "search":
        for r in store.search(" ".join(args), limit=10):
            print(f"[{r['rank']:.3f}] {r['source']} {r['locator'] or ''}\n    {r['text'][:220]}\n")
    elif cmd == "graph":
        from . import graph
        graph.main(args)
    elif cmd == "embed":
        from . import embed
        embed.main(args)
    elif cmd == "ask":
        from . import ask
        ask.cli(" ".join(args))
    elif cmd == "stats":
        print(store.stats())
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
