"""The topic graph: every passage tagged with the topics it is about; topics linked where they meet.

Tagging is Haiku 5.5 over the Batch API, six passages per request (the instructions are most of each request,
so grouping cuts the bill). Topic names are English Wikipedia article titles where one exists: that one rule
makes the same event tagged from thirty books land on one node without a merge step.

A node knows which sources cover it, so an answer can draw on several kinds of evidence for the same topic
(a commission report, a first-hand account, the law) instead of the five best-ranked passages of one book.
An edge counts the passages, and the distinct sources, in which two topics appear together. Nothing is learned
from anyone's questions: the graph comes from the corpus alone.

  python -m engine graph --pilot 30     synchronous, 30 passages: check quality and cost first
  python -m engine graph --submit       one batch per 10,000 untagged passages (ids kept in runs/batches.json)
  python -m engine graph --collect      fetch finished batches, store tags
  python -m engine graph --link         rebuild counts and edges from the stored tags (free)
  python -m engine graph --prune        remove stored tags naming a year after their source (free)
"""
import json
import re
import time
from pathlib import Path

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from . import corpus, llm, store

GROUP = 6
BATCH_MAX = 10_000 // GROUP  # requests per batch
BATCHES = Path(__file__).resolve().parent.parent / "runs" / "batches.json"
KINDS = ["event", "person", "organisation", "place", "law", "institution", "document", "concept"]

SYSTEM = """You tag passages from historical, legal and official Indian sources with the topics they are about.

For each passage, give 0 to 4 topics it is substantially about: events, people, organisations, places, laws or
constitutional provisions, institutions, named documents, or specific concepts (e.g. "Doctrine of lapse").

Naming rules:
- Use the exact title of the English Wikipedia article for the topic when one exists, e.g. "The Emergency (India)",
  "Jallianwala Bagh massacre", "Article 370 of the Constitution of India", "Instrument of Accession",
  "Vallabhbhai Patel", "Indian Rebellion of 1857", "Bharatiya Nyaya Sanhita", "Simla Agreement".
- Otherwise a short, specific, neutral name in the same style. Never a value judgment or a stance.
- People by their full common name, no titles ("Jawaharlal Nehru", not "Pandit Nehru" or "Panditji").
- An event must be the one the passage itself describes. Each passage carries its source's year: never tag an
  event that happened after it (a famine in a book of 1902 is not the Bengal famine of 1943). When you can't
  tell which event it is, leave it out.
- Skip topics merely mentioned in passing, and generic ones ("India", "history", "politics", "government").
- A passage that is an index, table of contents, list of names, garbled scan or blank page gets no topics."""

SCHEMA = {
    "type": "object",
    "properties": {"passages": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "topics": {"type": "array", "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "kind": {"type": "string", "enum": KINDS}},
                "required": ["name", "kind"], "additionalProperties": False}},
        },
        "required": ["n", "topics"], "additionalProperties": False}}},
    "required": ["passages"], "additionalProperties": False,
}


def _params(group):
    body = "\n\n".join(f"<passage n=\"{i + 1}\" source=\"{p['title']} ({p['year']})\">\n{p['text']}\n</passage>"
                       for i, p in enumerate(group))
    return {
        "model": llm.WORKER_MODEL, "max_tokens": 1500, "system": SYSTEM,
        "thinking": {"type": "disabled"}, "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": f"Tag each passage.\n\n{body}"}],
    }


def _untagged(limit=None):
    with store.connect() as c:
        return c.execute("""select p.id, p.text, s.slug, s.title, s.year from indiafacts.passages p
                            join indiafacts.sources s on s.slug = p.source
                            where not exists (select 1 from indiafacts.passage_topics t where t.passage = p.id)
                              and p.id not in (select unnest(%s::int[]))
                            order by """ + (f"random() limit {int(limit)}" if limit else "p.source, p.seq"),
                         (_pending_ids(),)).fetchall()


def _pending_ids():
    """Passages already sent in a batch not yet collected, or tagged with no topics (kept in batches.json)."""
    state = _state()
    return [i for b in state["batches"] if not b.get("collected") for g in b["groups"] for i in g] + state.get("empty", [])


def _state():
    return json.loads(BATCHES.read_text()) if BATCHES.exists() else {"batches": [], "empty": []}


def _save(state):
    BATCHES.parent.mkdir(parents=True, exist_ok=True)
    BATCHES.write_text(json.dumps(state))


def _norm(name):
    name = re.sub(r"\s+", " ", name.replace("–", "-").replace("—", "-")).strip().strip(".")
    return name[:1].upper() + name[1:] if name else name


# The last year each source covers: a topic named with a later year ("2002 Gujarat riots" on a 1936 memoir) is
# kept only if the passage itself has that year (an amended Act citing a later Act). The prompt asks the same;
# this catches the misses.
LAST_YEAR = {x["slug"]: x.get("to", x["year"]) for x in corpus.sources()}


def _too_late(name, p):
    last = LAST_YEAR.get(p["slug"])
    return bool(last) and any(int(y) > last + 1 and y not in p["text"] for y in re.findall(r"1[6-9]\d\d|20\d\d", name))


def _store_tags(group, parsed):
    """Write tags for one group; returns the passage ids that got no topics."""
    by_n = {p["n"]: p["topics"] for p in parsed.get("passages", [])}
    empty = []
    with store.connect() as c, c.transaction():
        for i, p in enumerate(group):
            topics = by_n.get(i + 1, [])
            if not topics:
                empty.append(p["id"])
            for t in topics[:4]:
                name = _norm(t["name"])
                if not name or len(name) > 120 or _too_late(name, p):
                    continue
                tid = c.execute("""insert into indiafacts.topics (name, kind) values (%s, %s)
                                   on conflict (name) do update set name = excluded.name returning id""",
                                (name, t["kind"])).fetchone()["id"]
                c.execute("insert into indiafacts.passage_topics values (%s, %s) on conflict do nothing", (p["id"], tid))
    return empty


def pilot(n):
    cl = llm.client()
    rows = _untagged(limit=n)
    total = 0.0
    for k in range(0, len(rows), GROUP):
        group = rows[k:k + GROUP]
        msg = cl.messages.create(**_params(group))
        total += llm.cost(llm.WORKER_MODEL, msg.usage)
        parsed = json.loads(next(b.text for b in msg.content if b.type == "text"))
        empty = _store_tags(group, parsed)
        state = _state()
        state["empty"] = state.get("empty", []) + empty
        _save(state)
        for i, p in enumerate(group):
            tags = next((x["topics"] for x in parsed["passages"] if x["n"] == i + 1), [])
            print(f"- {p['title'][:40]}: {p['text'][:90]!r}\n    -> {[t['name'] for t in tags]}")
    llm.record("graph-pilot", total, passages=len(rows))
    print(f"\n{len(rows)} passages, ${total:.4f} (${total / max(1, len(rows)) * 1000:.3f} per 1,000 at sync prices; batch is half)")


def submit():
    cl = llm.client()
    rows = _untagged()
    groups = [rows[k:k + GROUP] for k in range(0, len(rows), GROUP)]
    state = _state()
    for b in range(0, len(groups), BATCH_MAX):
        chunk = groups[b:b + BATCH_MAX]
        batch = cl.messages.batches.create(requests=[
            Request(custom_id=f"g{b + j}", params=MessageCreateParamsNonStreaming(**_params(g)))
            for j, g in enumerate(chunk)])
        state["batches"].append({"id": batch.id, "first": b, "groups": [[p["id"] for p in g] for g in chunk],
                                 "submitted": round(time.time())})
        _save(state)
        print(f"batch {batch.id}: {len(chunk)} requests, {sum(map(len, chunk))} passages")


def collect():
    cl = llm.client()
    state = _state()
    with store.connect() as c:
        meta = {r["id"]: r for r in c.execute("""select p.id, p.text, s.slug, s.title, s.year from indiafacts.passages p
                                                 join indiafacts.sources s on s.slug = p.source""").fetchall()}
    for b in state["batches"]:
        if b.get("collected"):
            continue
        info = cl.messages.batches.retrieve(b["id"])
        if info.processing_status != "ended":
            print(f"batch {b['id']}: {info.processing_status} {info.request_counts}")
            continue
        total, failed = 0.0, 0
        for res in cl.messages.batches.results(b["id"]):
            j = int(res.custom_id[1:]) - b["first"]
            group = [meta[i] for i in b["groups"][j] if i in meta]
            if res.result.type != "succeeded":
                failed += 1
                continue
            msg = res.result.message
            total += llm.cost(llm.WORKER_MODEL, msg.usage, batch=True)
            try:
                parsed = json.loads(next(x.text for x in msg.content if x.type == "text"))
            except (StopIteration, json.JSONDecodeError):
                failed += 1
                continue
            state["empty"] = state.get("empty", []) + _store_tags(group, parsed)
        b["collected"] = True
        b["usd"] = round(total, 4)
        b["failed"] = failed
        _save(state)
        llm.record("graph-batch", total, batch=b["id"])
        print(f"batch {b['id']}: collected, ${total:.4f}, {failed} failed requests (their passages stay untagged)")


def link():
    """Counts per topic and co-occurrence edges, from the stored tags."""
    with store.connect() as c, c.transaction():
        c.execute("update indiafacts.topics set passages = 0, sources = 0")  # topics left with no passages
        c.execute("""update indiafacts.topics t set passages = x.p, sources = x.s
                     from (select pt.topic, count(*) p, count(distinct p.source) s
                           from indiafacts.passage_topics pt join indiafacts.passages p on p.id = pt.passage
                           group by pt.topic) x where x.topic = t.id""")
        c.execute("truncate indiafacts.topic_edges")
        c.execute("""insert into indiafacts.topic_edges (a, b, passages, sources)
                     select x.topic, y.topic, count(*), count(distinct p.source)
                     from indiafacts.passage_topics x
                     join indiafacts.passage_topics y on y.passage = x.passage and y.topic > x.topic
                     join indiafacts.passages p on p.id = x.passage
                     join indiafacts.topics tx on tx.id = x.topic and tx.passages >= 2
                     join indiafacts.topics ty on ty.id = y.topic and ty.passages >= 2
                     group by x.topic, y.topic""")
    print(store.stats())


def prune():
    """Remove stored tags that fail _too_late (tags from before the rule, or after a source's `to` changes)."""
    with store.connect() as c, c.transaction():
        rows = c.execute("""select pt.passage, pt.topic, t.name, p.text, s.slug from indiafacts.passage_topics pt
                            join indiafacts.topics t on t.id = pt.topic join indiafacts.passages p on p.id = pt.passage
                            join indiafacts.sources s on s.slug = p.source
                            where t.name ~ '(1[6-9][0-9][0-9]|20[0-9][0-9])'""").fetchall()
        bad = [(r["passage"], r["topic"]) for r in rows if _too_late(r["name"], r)]
        for pair in bad:
            c.execute("delete from indiafacts.passage_topics where passage = %s and topic = %s", pair)
    print(f"removed {len(bad)} tags naming a year after their source; run --link next")


def main(args):
    if args[:1] == ["--pilot"]:
        pilot(int(args[1]) if len(args) > 1 else 30)
    elif args[:1] == ["--submit"]:
        submit()
    elif args[:1] == ["--collect"]:
        collect()
    elif args[:1] == ["--link"]:
        link()
    elif args[:1] == ["--prune"]:
        prune()
    else:
        print(__doc__)
