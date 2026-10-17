"""indiafacts evals: the system against a strong plain baseline, on mirrored pairs and fact questions.

  .venv/bin/python -m evals.run [--only id,id] [--arms system,baseline] [--cap 10]

Arms:
- system:   engine.ask.answer (corpus + graph + official-site web search, citations on).
- baseline: Opus 5.5 with the same answer rules and open web search, no corpus: what a sharp prompt alone gets.

Scores, all by code except `support` (Haiku 5.5 reads each cited sentence beside its quotes):
- words, sources cited, kinds of evidence, unsourced sentences, judging words (a fixed list);
- premise: for premise_false questions, does the first sentence reject the premise (Haiku);
- support: share of cited sentences whose quotes back them (yes / partly / no);
- pairs: the gap between the two sides of each mirrored pair on every number above.
Writes evals/runs/<stamp>/{system,baseline}.jsonl and report.md (answers side by side).
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine import ask, llm

# Not the model that checks citations at run time (Haiku), so the eval doesn't grade the checker with itself.
JUDGE_MODEL = "claude-sonnet-5-5"
llm.PRICES.setdefault(JUDGE_MODEL, (2.00, 10.00, 0.20, 2.50))

HERE = Path(__file__).resolve().parent
JUDGING = ["brutal", "heroic", "infamous", "notorious", "draconian", "glorious", "shameful", "controversial",
           "so-called", "regime", "betray", "tyrann", "atrocious", "barbaric", "visionary", "great leader", "admitted",
           "claimed", "alleged", "dark chapter", "black day", "martyr", "traitor", "appeas", "propaganda", "unfortunately",
           "fortunately", "rightly", "wrongly", "sadly"]


def baseline(question):
    t0 = time.time()
    cl = llm.client()
    messages = [{"role": "user", "content": question}]
    total, msg = 0.0, None
    for _ in range(4):
        msg = cl.beta.messages.create(
            model=llm.ANSWER_MODEL, max_tokens=8000, system=ask.ANSWER_SYSTEM.split("\n\nTools:")[0]
            + "\n\nTool: web_search, on the open web. Prefer primary records (laws, rulings, official reports).",
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}], messages=messages,
            thinking={"type": "adaptive"}, output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"})
        total += llm.cost(llm.ANSWER_MODEL, msg.usage)
        messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]})
        if msg.stop_reason != "pause_turn":
            break
    out = ask.render(msg, [])
    out["meta"] = {"usd": round(total, 4), "seconds": round(time.time() - t0, 1)}
    llm.record("eval-baseline", total)
    return out


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "premise_rejected": {"type": "boolean"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "i": {"type": "integer"}, "support": {"type": "string", "enum": ["yes", "partly", "no"]}},
            "required": ["i", "support"], "additionalProperties": False}},
    },
    "required": ["premise_rejected", "claims"], "additionalProperties": False,
}


def judge(question, out):
    """Haiku 5.5: does each cited sentence follow from its quotes; does the answer open by rejecting a premise."""
    quotes = {s["n"]: s["quote"][:900] for s in out["sources"]}
    items = [(i, s) for i, s in enumerate(out["segments"]) if s["cites"]]
    body = "\n\n".join(f"<claim i=\"{i}\">{s['text'].strip()}</claim>\n" + "\n".join(f"<quote>{quotes[n]}</quote>" for n in s["cites"] if n in quotes)
                       for i, s in items)
    first = next((s["text"].strip() for s in out["segments"] if s["text"].strip()), "")
    prompt = (f"Question: {question}\nThe answer's first sentence: {first}\n\n"
              "premise_rejected: true only if that first sentence says the question's assumption is wrong or unsupported.\n"
              "For each claim, support: yes if its quotes state it, partly if they state some of it, no if they don't.\n\n" + body)
    msg = llm.client().messages.create(
        model=JUDGE_MODEL, max_tokens=4000, thinking={"type": "disabled"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}])
    llm.record("eval-judge", llm.cost(JUDGE_MODEL, msg.usage))
    return json.loads(next(b.text for b in msg.content if b.type == "text")), llm.cost(JUDGE_MODEL, msg.usage)


def score(q, out):
    text = "".join(s["text"] for s in out.get("segments", []))
    low = text.lower()
    j, jcost = judge(q["q"], out) if out.get("ok") and out.get("segments") else ({"premise_rejected": False, "claims": []}, 0.0)
    sup = [c["support"] for c in j["claims"]]
    return {
        "words": len(text.split()),
        "sources": len(out.get("sources", [])),
        "kinds": len({s["standing"] for s in out.get("sources", [])}),
        "unsourced": sum(1 for s in out.get("segments", []) if s.get("unsourced")) + out.get("unsourced_removed", 0),
        "judging": [w for w in JUDGING if w in low],
        "support_yes": round(sup.count("yes") / len(sup), 2) if sup else None,
        "support_no": sup.count("no"),
        "premise_rejected": j["premise_rejected"],
        "kind": out.get("kind"),
        "kind_ok": out.get("kind") == q.get("expect") if out.get("kind") else None,
        "removed": out.get("removed", 0),
        "judge_usd": jcost,
    }


def run_arm(arm, qs, cap):
    fn = ask.answer if arm == "system" else baseline
    rows, spent = [], 0.0

    def one(q):
        try:
            out = fn(q["q"])
        except Exception as e:  # a failed question is a result, not a crash
            out = {"ok": False, "reason": repr(e)[:200], "segments": [], "sources": [], "meta": {"usd": 0}}
        return {"q": q, "out": out, "score": score(q, out)}

    with ThreadPoolExecutor(3) as pool:
        for r in pool.map(one, qs):
            spent += r["out"]["meta"].get("usd", 0) + r["score"]["judge_usd"]
            rows.append(r)
            print(f"  {arm:8} {r['q']['id']:16} ${r['out']['meta'].get('usd', 0):.3f} {r['score']['words']:>4}w "
                  f"{r['score']['sources']:>2}src {r['score']['unsourced']}unsrc support={r['score']['support_yes']}", flush=True)
            if spent > cap:
                print(f"  cap ${cap} reached, stopping {arm}")
                break
    return rows


def report(stamp, arms):
    lines = [f"# indiafacts eval {stamp}", ""]
    for arm, rows in arms.items():
        n = len(rows)
        tot = lambda k: sum((r["score"][k] or 0) for r in rows)
        sup = [r["score"]["support_yes"] for r in rows if r["score"]["support_yes"] is not None]
        prem = [r for r in rows if r["q"].get("premise_false")]
        lines += [f"## {arm}: {n} questions, ${sum(r['out']['meta'].get('usd', 0) for r in rows):.2f}",
                  f"- mean words {tot('words') / max(n, 1):.0f}, sources {tot('sources') / max(n, 1):.1f}, kinds {tot('kinds') / max(n, 1):.1f}",
                  f"- unsourced sentences: {tot('unsourced')} · unsupported citations: {tot('support_no')} · mean supported share {sum(sup) / max(len(sup), 1):.2f}",
                  f"- judging words: {sum(len(r['score']['judging']) for r in rows)} {[w for r in rows for w in r['score']['judging']]}",
                  f"- false premises rejected first: {sum(r['score']['premise_rejected'] for r in prem)}/{len(prem)}",
                  f"- kind as expected (system only): {sum(1 for r in rows if r['score']['kind_ok'])}/{sum(1 for r in rows if r['score']['kind_ok'] is not None)}"
                  f" · sentences removed by the run-time check: {tot('removed')}", ""]
        pairs = {}
        for r in rows:
            if r["q"].get("pair"):
                pairs.setdefault(r["q"]["pair"], []).append(r)
        if pairs:
            lines += ["| pair | words a/b | sources a/b | kinds a/b | unsourced a/b | judging a/b |", "|---|---|---|---|---|---|"]
            for p, (a, b) in ((p, v) for p, v in pairs.items() if len(v) == 2):
                f = lambda k: f"{a['score'][k] if k != 'judging' else len(a['score'][k])}/{b['score'][k] if k != 'judging' else len(b['score'][k])}"
                lines.append(f"| {p} | {f('words')} | {f('sources')} | {f('kinds')} | {f('unsourced')} | {f('judging')} |")
            lines.append("")
    lines += ["## Answers", ""]
    ids = [r["q"]["id"] for r in next(iter(arms.values()))]
    for qid in ids:
        q = next(r["q"] for r in next(iter(arms.values())) if r["q"]["id"] == qid)
        lines += [f"### {qid}: {q['q']}", ""]
        for arm, rows in arms.items():
            r = next((x for x in rows if x["q"]["id"] == qid), None)
            if r:
                kind = f" (kind: {r['out'].get('kind')}, expected {q.get('expect')})" if r["out"].get("kind") else ""
                lines += [f"**{arm}**{kind}", "", ask.plain(r["out"]) if r["out"].get("ok") else f"(failed: {r['out'].get('reason')})", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--arms", default="system,baseline")
    ap.add_argument("--cap", type=float, default=10.0, help="stop an arm past this many dollars")
    a = ap.parse_args()
    qs = json.loads((HERE / "questions.json").read_text())["questions"]
    if a.only:
        want = set(a.only.split(","))
        qs = [q for q in qs if q["id"] in want]
    stamp = time.strftime("%Y%m%d-%H%M")
    out_dir = HERE / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in a.arms.split(","):
        rows = run_arm(arm, qs, a.cap)
        (out_dir / f"{arm}.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        arms[arm] = rows
    (out_dir / "report.md").write_text(report(stamp, arms))
    print(f"\n{out_dir / 'report.md'}")


if __name__ == "__main__":
    sys.exit(main())
