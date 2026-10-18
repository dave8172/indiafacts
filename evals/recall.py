"""indiafacts search test: do the passages each question needs reach the writer?

  .venv/bin/python -m evals.recall [--only id,id] [--replan]

For each question in evals/recall.json: the planner (Haiku, about $0.002 a question) writes the searches, then
the same retrieval the engine uses runs, for free. A target is
- given: among the passages handed to the writer (after the cap per source),
- found: among every candidate the three searches return, but cut by the caps,
- missed: in no candidate.
Each given target also says which searches found it (words, meaning, graph). Plans are kept in
evals/runs/recall-plans.json, keyed by the question and the planner prompt, so a change to retrieval alone
reruns for $0; --replan writes new ones. Writes evals/runs/recall-<stamp>.json and prints the summary.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from engine import ask, llm, store

HERE = Path(__file__).parent
PLANS = HERE / "runs" / "recall-plans.json"


def _norm(t):
    return re.sub(r"\s+", " ", t).lower()


def matches(r, t):
    if r["source"] != t["source"]:
        return False
    loc = t.get("locator")
    if loc and not (r["locator"] == loc or (r["locator"] or "").startswith(loc + " ")):
        return False
    has = t.get("has")
    if has:
        text = _norm(r["text"])
        return any(_norm(h) in text for h in ([has] if isinstance(has, str) else has))
    return True


def exists(t):
    """The target is in the corpus at all: a test can't be failed by a passage that isn't there."""
    with store.connect() as c:
        rows = c.execute("select source, locator, text from indiafacts.passages where source = %s", (t["source"],)).fetchall()
    return any(matches(r, t) for r in rows)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--replan", action="store_true")
    a = ap.parse_args(argv)
    qs = json.loads((HERE / "recall.json").read_text())["questions"]
    if a.only:
        want = set(a.only.split(","))
        qs = [q for q in qs if q["id"] in want]
    key = hashlib.sha256(ask.PLAN_SYSTEM.encode()).hexdigest()[:12]
    plans = json.loads(PLANS.read_text()) if PLANS.exists() else {}
    cl, cost, rows = llm.client(), 0.0, []
    for q in qs:
        k = f"{key}:{q['q']}"
        if a.replan or k not in plans:
            p, c = ask.plan(cl, q["q"])
            cost += c
            plans[k] = p
        p = dict(plans[k], question=q["q"])
        full = p.get("depth") == "full"
        ranked, _ = ask.candidates(p)
        given = ask.diversify(ranked, ask.FULL_PASSAGES, ask.FULL_PER_SOURCE) if full else ask.diversify(ranked)
        res = []
        for t in q["targets"]:
            if not exists(t):
                res.append({"target": t, "status": "absent"})
                continue
            g = next((r for r in given if matches(r, t)), None)
            f = next((i for i, r in enumerate(ranked) if matches(r, t)), None)
            res.append({"target": t, "status": "given" if g else "found" if f is not None else "missed",
                        "via": g["via"] if g else ranked[f]["via"] if f is not None else [],
                        "rank": f})
        rows.append({"id": q["id"], "q": q["q"], "kind": p["kind"], "depth": p.get("depth"),
                     "searches": p["searches"], "candidates": len(ranked), "given": len(given), "targets": res})
    PLANS.parent.mkdir(exist_ok=True)
    PLANS.write_text(json.dumps(plans, indent=1, ensure_ascii=False))
    if cost:
        llm.record("eval", cost)
    out = HERE / "runs" / f"recall-{time.strftime('%Y%m%d-%H%M')}.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    ts = [t for r in rows for t in r["targets"]]
    n = {s: sum(t["status"] == s for t in ts) for s in ("given", "found", "missed", "absent")}
    scored = len(ts) - n["absent"]
    print(f"{len(rows)} questions, {len(ts)} targets: given {n['given']}/{scored} ({n['given'] / max(scored, 1):.0%}), "
          f"found but cut {n['found']}, missed {n['missed']}, not in the corpus {n['absent']}  (planner ${cost:.3f})")
    paths = {}
    for t in ts:
        if t["status"] == "given":
            for v in t["via"]:
                paths[v] = paths.get(v, 0) + 1
            if len(t["via"]) == 1:
                paths["only " + t["via"][0]] = paths.get("only " + t["via"][0], 0) + 1
    print("given targets found by:", ", ".join(f"{k} {v}" for k, v in sorted(paths.items())))
    for r in rows:
        for t in r["targets"]:
            if t["status"] != "given":
                tt = t["target"]
                where = f"  (candidate {t['rank'] + 1} of {r['candidates']})" if t["status"] == "found" else ""
                print(f"  {t['status']:7} {r['id']:20} {tt['source']} {tt.get('locator') or ''} {tt.get('has') or ''}{where}")
                if t["status"] == "missed":
                    print(f"          searches: {r['searches']}")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
