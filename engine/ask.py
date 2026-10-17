"""One question in, one sourced answer out: brief when one fact answers it, fuller and organised by source when the
question needs it. Stateless: nothing about the asker is read or kept.

1. Plan (Haiku 5.5): restate the question neutrally, list the premises it assumes, write search queries
   (both sides' vocabulary), name the topics, say whether it needs the web (court rulings, recent events), and
   how deep the answer should go (brief or full).
2. Retrieve (free): full-text search for each query, plus the passages tagged with the planned topics in the
   graph; then spread the picks across sources and kinds of evidence, so one book can't fill the answer.
3. Answer (Sonnet 5.5): the passages go in as search results with citations on, so every quote the answer
   shows is text that really is in the source (the API guarantees it). The writer may search the corpus again, or
   search the web on official and court sites only. A sentence with no citation is marked as unsourced.
4. Check (Haiku 5.5): each web quote is widened from its page (`pages.py`), then every sentence is read beside
   its quotes; one they don't back is removed.
5. Number by source (`by_source`): one number per source, its cited passages listed under it.
"""
import json
import re
import sys
import time
from urllib.parse import urlparse

from . import corpus, llm, pages, store

MAX_PASSAGES = 16
PER_SOURCE = 3
# A full answer (what a report found, how an event came about) needs more of each main source: 24 passages, 6 a source
FULL_PASSAGES = 24
FULL_PER_SOURCE = 6
MIN_SCORE = 0.3  # a hit must rank at least this share of its query's best
MEANING_FLOOR = 0.85  # a meaning hit must reach this share of the best one's cosine
MAX_TURNS = 4
WEB_USES = 3

STANDINGS = corpus.standings()
SOURCES = {s["slug"]: s for s in corpus.sources()}

# Web search stays on the sites that publish records (sites.json, "web": true): courts, statutes, Parliament,
# ministries, official data, and faithful copies of them. Never the media (grade D).
SITES = json.loads((corpus.ROOT / "sites.json").read_text())["sites"]
WEB_DOMAINS = [s["domain"] for s in SITES if s.get("web") and s["grade"] in ("A", "B", "C")]


WEB_LABEL = {"A": "Original record", "B": "Copy of an official record", "C": "Research", "D": "News report"}


def web_standing(url):
    """The cited page's grade in sites.json, said in words, with who published it."""
    host = urlparse(url).netloc.lower().removeprefix("www.")
    site = max((s for s in SITES if host == s["domain"] or host.endswith("." + s["domain"])),
               key=lambda s: len(s["domain"]), default=None)
    if not site:
        return "Web page, not graded"
    return f"{WEB_LABEL[site['grade']]}: {site['publisher']}"


# --- 1. plan ---------------------------------------------------------------------------------------------

# The corpus as the planner sees it, from sources.json, so adding a source needs no prompt edit.
CORPUS = "\n".join(f"- {s['title']}, {s['author']} [{STANDINGS[s['standing']]}]" for s in corpus.sources())

PLAN_SYSTEM = f"""You prepare a question for indiafacts, a fact-only research tool on Indian history, law, politics and
geopolitics. indiafacts reports what sources record. It never judges people, parties, governments or events.

First, sort the question:
- fact: asks what happened, when, who did or said what, what a text or law provides, what a source records,
  what the numbers are, or what the law says about an act (e.g. "can someone be jailed for X").
- loaded: has a factual core but asks for, or assumes, a judgment: responsibility, blame, credit, betrayal,
  excesses, mishandling, "drain", "provoked", or a contested label ("was 1857 a war of independence or a
  mutiny"). Example: "How was the Congress responsible for Partition?" -> core: "How did Partition come about,
  and what did the Congress decide and do in those events?"
- opinion: asks only for a judgment, with no factual core to report: good or bad, best or worst, right or
  wrong, should or shouldn't, deserved, admirable, "who was the greatest".

Then fill in:
- kind: fact, loaded or opinion.
- factual: the question as a plain question of fact. For fact, the question itself with any loaded words
  removed. For loaded, the factual core: ask what happened, keep the people and organisations the question
  names, and drop every word of blame, credit, responsibility or label. For opinion, empty.
- suggest: for loaded or opinion, three short questions of fact a person could ask instead, on the same
  subject, framed the same way whichever side the question seems to favour. Else empty.
- premises: dates, events or facts that the question's own words take for granted (not judgments, and
  nothing the question doesn't say). These will be checked.
- act: if it asks whether a named person can be sued, charged or held liable, the act alone, without the
  name ("publishing a false statement that harms someone's reputation"); else empty.
- searches: 5 to 8 keyword queries of 2 to 4 words each for Postgres full-text search, where every word must
  appear in a passage (no quotes or operators). Cover each part of the factual question, other names and
  spellings, the terms the period's documents use, and the vocabulary each side of a dispute would use.
  Empty for opinion.
- topics: up to 5 topics as English Wikipedia article titles.
- web: true if it needs court rulings, Constitution amendments after 2021, laws after 2023, events the corpus
  can't cover, or current figures.
- depth: brief if one short fact answers it: what a provision says, a date, a penalty, whether the law covers
  an act. full if a good answer brings together several findings or accounts: what a commission, report,
  committee, debate or book recorded, found or recommended; how an event came about; what each side said or
  did; a broad "what happened" question. When unsure, full.

The corpus:
{{corpus}}""".replace("{corpus}", CORPUS)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["fact", "loaded", "opinion"]},
        "factual": {"type": "string"},
        "suggest": {"type": "array", "items": {"type": "string"}},
        "premises": {"type": "array", "items": {"type": "string"}},
        "act": {"type": "string"},
        "searches": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "web": {"type": "boolean"},
        "depth": {"type": "string", "enum": ["brief", "full"]},
    },
    "required": ["kind", "factual", "suggest", "premises", "act", "searches", "topics", "web", "depth"],
    "additionalProperties": False,
}

OPINION = ("India Facts answers questions of fact, not opinion. It reports what the sources record and never "
           "judges whether a person, party, government or event was good or bad, right or wrong. Ask about "
           "what happened, what was said or decided, or what a law provides.")
LOADED = "India Facts answers questions of fact, so it answered this instead:"


def plan(cl, question):
    msg = cl.messages.create(
        model=llm.WORKER_MODEL, max_tokens=1500, system=PLAN_SYSTEM, thinking={"type": "disabled"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
        messages=[{"role": "user", "content": question}])
    return json.loads(next(b.text for b in msg.content if b.type == "text")), llm.cost(llm.WORKER_MODEL, msg.usage)


# --- 2. retrieve -----------------------------------------------------------------------------------------

def topic_passages(names, per_topic=8):
    """Passages tagged with the nearest graph topics, at most two per source per topic."""
    if not names:
        return [], []
    with store.connect() as c:
        found = []
        for n in names:
            row = c.execute("""select id, name, passages, sources from indiafacts.topics
                               where lower(name) = lower(%s) or tsv @@ plainto_tsquery('simple', %s)
                               order by (lower(name) = lower(%s)) desc, passages desc limit 1""", (n, n, n)).fetchone()
            if row and row not in found:
                found.append(row)
        rows = []
        for t in found:
            rows += c.execute("""select * from (
                                   select p.id, p.source, p.seq, p.locator, p.text, p.blocks, s.title, s.author, s.year,
                                          s.standing, s.ia, 0.05 as rank,
                                          row_number() over (partition by p.source order by p.seq) as k
                                   from indiafacts.passage_topics pt join indiafacts.passages p on p.id = pt.passage
                                   join indiafacts.sources s on s.slug = p.source where pt.topic = %s) x
                                 where k <= 2 limit %s""", (t["id"], per_topic)).fetchall()
        return rows, found


def related(topic_ids, limit=6):
    if not topic_ids:
        return []
    with store.connect() as c:
        return [r["name"] for r in c.execute(
            """select t.name, sum(e.sources) w from indiafacts.topic_edges e
               join indiafacts.topics t on t.id = case when e.a = any(%s) then e.b else e.a end
               where (e.a = any(%s) or e.b = any(%s)) and not t.id = any(%s)
               group by t.name order by w desc limit %s""", (topic_ids, topic_ids, topic_ids, topic_ids, limit)).fetchall()]


def meaning_passages(question, k=12):
    """Passages close in meaning to the question (engine/embed.py), scored against the best of them (0..1)."""
    from . import embed
    if not question:
        return []
    hits = embed.search(question, k=k)
    if not hits:
        return []
    best = hits[0][1]
    keep = {i: s / best for i, s in hits if s >= MEANING_FLOOR * best}
    with store.connect() as c:
        rows = c.execute("""select p.id, p.source, p.seq, p.locator, p.text, p.blocks, s.title, s.author, s.year,
                                   s.standing, s.ia, 0.0 as rank
                            from indiafacts.passages p join indiafacts.sources s on s.slug = p.source
                            where p.id = any(%s)""", (list(keep),)).fetchall()
    for r in rows:
        r["score"] = keep[r["id"]]
    return rows


def retrieve(p, full=False):
    """Each query's hits scored against that query's best (0..1); weak hits dropped; then spread across sources."""
    hits = {}
    for q in p["searches"]:
        rows = store.search(q, limit=12)  # every word must match; an any-word search here pulled in noise
        top = max((r["rank"] for r in rows), default=0) or 1
        for r in rows:
            r["score"] = r["rank"] / top
            if r["score"] >= MIN_SCORE and (r["id"] not in hits or r["score"] > hits[r["id"]]["score"]):
                hits[r["id"]] = r
    for r in meaning_passages(p.get("factual") or p.get("question", "")):
        if r["id"] not in hits or r["score"] > hits[r["id"]]["score"]:
            hits[r["id"]] = r
    trows, topics = topic_passages(p["topics"])
    for r in trows:  # passages the graph ties to the topic: a second way in, from every source that covers it
        r["score"] = 0.5
        hits.setdefault(r["id"], r)
    ranked = sorted(hits.values(), key=lambda r: -r["score"])
    return (diversify(ranked, FULL_PASSAGES, FULL_PER_SOURCE) if full else diversify(ranked)), topics


def diversify(rows, n=MAX_PASSAGES, per_source=PER_SOURCE):
    """Best first, at most `per_source` passages from any one source, so no single book fills the answer."""
    picked, used = [], {}
    for r in rows:
        if used.get(r["source"], 0) < per_source:
            used[r["source"]] = used.get(r["source"], 0) + 1
            picked.append(r)
            if len(picked) == n:
                break
    return picked


def _title(r):
    loc = f", {r['locator']}" if r.get("locator") else ""
    return f"{r['title']}{loc}. {r['author']}. [{STANDINGS.get(r['standing'], r['standing'])}]"


def as_search_result(r):
    return {"type": "search_result", "source": f"indiafacts:{r['id']}", "title": _title(r),
            "content": [{"type": "text", "text": b} for b in r["blocks"]], "citations": {"enabled": True}}


# --- 3. answer -------------------------------------------------------------------------------------------

ANSWER_SYSTEM = """You are indiafacts: you answer questions about Indian history, law, politics and geopolitics with
facts from sources, like a careful historian or court reporter. You hold no views and give none.

Rules:
- Every sentence carries a citation, the first one too: open with a cited fact, not a summary of what follows.
  Never fill a gap from memory. Only two sentences may go uncited: a first sentence saying the question's
  premise isn't supported, and one sentence at the end saying what the sources (and your searches) don't
  establish. Never write "I" or "the sources I have". Never comment on the
  passages themselves ("these sources don't show", "this passage doesn't give the year"): search for a missing
  fact such as a date, or leave it out.
- Attribute by kind of evidence. Law and rulings: state what the text says ("Article 352 lets the President...").
  Reports, records, memoirs and arguments: say who says it ("The Shah Commission found...", "Menon writes...").
  The government's or a party's own account is that party's account. Never present one account as settled
  fact when another source disputes it.
- Where sources disagree, give each account side by side, with its citation, in one sentence starting
  "Accounts differ:". Never a sentence that only announces a disagreement.
- No adjectives or verbs that judge (e.g. "brutal", "heroic", "controversial", "infamous", "admitted",
  "claimed" for one side and "stated" for the other). Use the same neutral verbs for every side.
- Never say or imply that a person, party, government or event was responsible, to blame, to credit, right,
  wrong, good or bad, and never confirm or deny a characterisation ("betrayal", "excesses", "mishandled",
  "drain", "war of independence"). Report what happened and what each source records. A source's own
  judgment may be quoted only as that source's, and only beside any source that judges otherwise.
- If the question assumes a date, event or fact that the sources contradict or don't support, say so in the
  first sentence.
- Questions on whether someone can be sued or charged: answer about the act, never about the person. Name the
  provisions that cover the act, what each requires, and its penalty, from the law's text. Don't say whether a
  named person did anything.
- Length: the brief says brief or full.
  Brief: lead with the answer. Under 120 words; never over 160. Pick the facts that answer the question and
  leave out background it didn't ask for.
  Full: an informative account of 250 to 450 words, never over 480, in short paragraphs, organised by source,
  not by passage. Give each main source its own paragraph, and start the paragraph with a cited sentence of
  this shape: "<source> (<year>) records that <a fact from it>.", adding what kind of record it is when its name
  doesn't say. For example: "The Shah Commission's Interim Report II (1978) records that Sushil Kumar issued 67
  detention orders on the night of 25-26 June 1975." Every uncited sentence is deleted after you write, so a
  sentence that only introduces a source ("The report covers X") vanishes and leaves its paragraph without a
  name. Then what it records: its scope, main findings, facts, numbers, decisions, and what
  it recommended or concluded, always as that source's account. Draw its passages together into one account;
  don't walk through them one by one. Where sources cover the same point differently, say so. Choose what
  helps a reader understand the subject and leave out minor detail.
  Both: plain sentences, no headings, no bullet points, no preamble, no closing summary.
- Answer the same way however the question is framed and whichever side it seems to favour.
- The corpus texts are OCR scans: quote only what is legible, and don't correct a source's facts silently.

Tools: search_corpus finds more passages in the corpus. web_search is limited to court, statute, Parliament,
ministry and official-statistics sites; use it for court rulings, Constitution amendments after 2021, laws
after 2023 and events the corpus can't cover. Prefer the corpus and primary records."""

SEARCH_TOOL = {
    "name": "search_corpus",
    "description": "Full-text search over the indiafacts corpus (laws, debates, commission reports, first-hand and "
                   "contemporary books). Returns up to 6 passages as citable search results.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Plain keywords, no quotes or operators"}},
        "required": ["query"], "additionalProperties": False,
    },
    "strict": True,
}


def answer(question, model=None):
    """model: the writer (default llm.ANSWER_MODEL); the evals compare writers."""
    model = model or llm.ANSWER_MODEL
    t0 = time.time()
    cl = llm.client()
    p, plan_cost = plan(cl, question)
    if p["kind"] == "opinion":  # nothing to report: say so, offer questions of fact, spend nothing more
        llm.record("answer", plan_cost)
        return {"ok": True, "kind": "opinion", "notice": OPINION, "suggest": p["suggest"][:3], "segments": [],
                "sources": [], "topics": [], "related": [],
                "meta": {"usd": round(plan_cost, 4), "seconds": round(time.time() - t0, 1), "plan": p}}
    asked = p["factual"] if p["kind"] == "loaded" and p["factual"] else question
    p["question"] = question
    full = p.get("depth") == "full"
    picked, topics = retrieve(p, full)
    results = list(picked)  # search_result_index -> passage, in the order they appear in the request

    brief = [f"Question: {asked}"]
    if p["premises"]:
        brief.append("Premises it assumes (check, don't assume): " + "; ".join(p["premises"]))
    if p["act"]:
        brief.append(f"Treat it as a question about this act, not the person: {p['act']}")
    if p["web"]:
        brief.append("This likely needs court rulings or recent official records: use web_search.")
    brief.append(f"Length: {'full' if full else 'brief'}.")
    content = [as_search_result(r) for r in picked] + [{"type": "text", "text": "\n".join(brief)}]
    messages = [{"role": "user", "content": content}]
    # web_search_20250305, not _20260209: the newer one filters results through code and returns no citations
    # (measured 2026-10-17), and every sentence here must carry a checkable quote.
    tools = [SEARCH_TOOL, {"type": "web_search_20250305", "name": "web_search", "max_uses": WEB_USES,
                           "allowed_domains": WEB_DOMAINS}]
    total = plan_cost
    final = None

    def turn():
        nonlocal total
        msg = cl.beta.messages.create(
            model=model, max_tokens=8000, system=ANSWER_SYSTEM, tools=tools, messages=messages,
            thinking={"type": "adaptive"}, output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default", "cache_control": {"type": "ephemeral"}})
        total += llm.cost(msg.model if msg.model in llm.PRICES else model, msg.usage)
        messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]})
        return msg

    for _ in range(MAX_TURNS):
        msg = final = turn()
        if msg.stop_reason == "pause_turn":
            continue
        if msg.stop_reason != "tool_use":
            break
        outs = []
        for b in msg.content:
            if b.type == "tool_use" and b.name == "search_corpus":
                rows = diversify(store.search(b.input["query"], limit=12) or store.search_any(b.input["query"].split()), n=6)
                results += rows
                outs.append({"type": "tool_result", "tool_use_id": b.id,
                             "content": [as_search_result(r) for r in rows] or "No passages found."})
        messages.append({"role": "user", "content": outs})
    if final is not None and final.stop_reason == "end_turn" and _has_markup(final):
        # Seen in 3 of 36 eval answers: the model writes a made-up "cite" tool call as text. Ask once more.
        messages.append({"role": "user", "content": "Your answer contains tool-call markup written as text. Write the "
                         "answer again in plain sentences. Citations attach by themselves when you state what the "
                         "search results say; there is no cite tool."})
        final = turn()
    out = render(final, results)
    if out["ok"]:
        total += pages.widen(cl, out["sources"])
        total += verify(cl, out)
        by_source(out)
    out["kind"] = p["kind"]
    if p["kind"] == "loaded":
        out["notice"], out["answered"], out["suggest"] = LOADED, asked, p["suggest"][:3]
    out["topics"] = [t["name"] for t in topics]
    out["related"] = related([t["id"] for t in topics])
    out["meta"] = {"usd": round(total, 4), "seconds": round(time.time() - t0, 1), "passages": len(results),
                   "plan": p, "model": final.model if final else None, "stop": final.stop_reason if final else None}
    llm.record("answer", total)
    return out


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "object", "properties": {
        "i": {"type": "integer"}, "support": {"type": "string", "enum": ["yes", "partly", "no", "gap"]}},
        "required": ["i", "support"], "additionalProperties": False}}},
    "required": ["claims"], "additionalProperties": False,
}


# a gap sentence says something is missing; one with none of these words isn't one
_ABSENCE = re.compile(r"\b(?:not|no|none|nothing|neither|nor|without|unclear|silent|unknown)\b|n't", re.I)


def verify(cl, out):
    """Second check, one whole sentence at a time. Cited: do its quotes state it? (The API already guarantees each
    quote is in the source; this checks the sentence doesn't go beyond its quotes.) Uncited: does it only say what
    the sources don't establish? Unbacked sentences and uncited statements of fact are removed and counted;
    "partly" ones stay, labelled. Returns the cost."""
    quotes = {s["n"]: s["quote"][:900] for s in out["sources"]}
    sentences = {}
    for s in out["segments"]:
        sentences.setdefault(s["sent"], []).append(s)
    claims = []
    for k, parts in sentences.items():
        text = "".join(p["text"] for p in parts).strip()
        if len(re.sub(r"\W", "", text)) > 25:
            claims.append((k, text, list(dict.fromkeys(n for p in parts for n in p["cites"] if n in quotes))))
    out["removed"] = out["unsourced_removed"] = 0
    if not claims:
        return 0.0
    body = "\n\n".join(f"<claim i=\"{k}\">{text}</claim>\n"
                        + ("\n".join(f"<quote>{quotes[n]}</quote>" for n in cites) or "<no-quotes/>")
                        for k, text, cites in claims)
    msg = cl.messages.create(
        model=llm.WORKER_MODEL, max_tokens=2000, thinking={"type": "disabled"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": VERIFY_SCHEMA}},
        system="You check citations. For each claim, judge only from its quotes: yes if the quotes state it, partly "
               "if they state some of it, no if they don't state it. A claim that a source says something is "
               "backed if the quote shows that source saying it. OCR slips in quotes don't count against a claim. "
               "A claim with no quotes: gap if it only says that something was not found or is not established by "
               "the sources; no for anything else, including a sentence saying what a source records or covers.",
        messages=[{"role": "user", "content": body}])
    verdict = {c["i"]: c["support"] for c in json.loads(next(b.text for b in msg.content if b.type == "text"))["claims"]}
    kept, gap = [], False  # gap: a removed sentence opened a paragraph, so the next kept one does
    for k, parts in sentences.items():
        v, cited = verdict.get(k), any(p["cites"] for p in parts)
        if not cited and v == "gap" and not _ABSENCE.search("".join(p["text"] for p in parts)):
            v = "no"  # in a long answer the checker called intro sentences ("X records evidence on Y") gaps
        drop = "removed" if cited and v == "no" else "unsourced_removed" if not cited and v not in (None, "gap") else None
        if drop:
            out[drop] += 1
            gap = gap or parts[0]["text"].lstrip(" ").startswith("\n")
            continue
        if cited and v == "partly":
            next(p for p in reversed(parts) if p["cites"])["partly"] = True
        if gap and kept:
            parts[0]["text"] = "\n\n" + parts[0]["text"].lstrip()
        gap = False
        kept += parts
    out["segments"] = kept
    used = {n for s in kept for n in s["cites"]}
    out["sources"] = [s for s in out["sources"] if s["n"] in used]
    return llm.cost(llm.WORKER_MODEL, msg.usage)


def by_source(out):
    """One citation number per source, not per passage, so an answer drawing on five passages of one report shows
    one number. Each source lists its cited passages (locator, quote, link that opens the scan at the quote) and a
    link to the source itself; each segment keeps `at` = [[source n, passage index]], the exact passages it cites,
    for the evals' judge and the page. Sources are numbered in the order the answer first cites them."""
    groups, num, at = [], {}, {}
    for s in out["sources"]:
        key = s.get("slug") or s["url"]
        if key not in num:
            num[key] = len(groups) + 1
            groups.append({"n": num[key], "title": s["title"], "author": s["author"], "year": s["year"],
                           "standing": s["standing"], "url": s["url"].split("?")[0], "web": s.get("web", False),
                           "passages": []})
        g = groups[num[key] - 1]
        same = next((i for i, x in enumerate(g["passages"]) if x["quote"] == s["quote"]), None)
        if same is None:
            same = len(g["passages"])
            g["passages"].append({"locator": s["locator"], "quote": s["quote"], "url": s["url"]})
        at[s["n"]] = [g["n"], same]
    for seg in out["segments"]:
        seg["at"] = list(dict.fromkeys(tuple(at[n]) for n in seg["cites"] if n in at))
        seg["at"] = [list(a) for a in seg["at"]]
        seg["cites"] = list(dict.fromkeys(a[0] for a in seg["at"]))
    out["sources"] = groups
    return out


_MARKUP = re.compile(r"</?(?:invoke|parameter|function_calls|antml:\w+)\b")


def _has_markup(msg):
    return any(b.type == "text" and _MARKUP.search(b.text) for b in msg.content)


def render(msg, results):
    """Answer segments with citation numbers, and the numbered source list."""
    if msg is None or msg.stop_reason == "refusal":
        return {"ok": False, "reason": "refused", "segments": [], "sources": []}
    if _has_markup(msg):
        return {"ok": False, "reason": "malformed", "segments": [], "sources": []}
    segments, sources, index = [], [], {}
    for b in msg.content:
        if b.type != "text":
            continue
        nums = []
        for c in (b.citations or []):
            if c.type == "search_result_location":
                r = results[c.search_result_index] if c.search_result_index < len(results) else None
                if r is None:
                    continue
                src = SOURCES.get(r["source"], {"ia": r["ia"]})
                key = (r["id"], c.start_block_index)
                entry = {"slug": r["source"], "title": r["title"], "author": r["author"], "year": r["year"], "locator": r["locator"],
                         "standing": STANDINGS.get(r["standing"], r["standing"]), "quote": c.cited_text.strip(),
                         "url": corpus.verify_url(src, c.cited_text)}
            elif c.type == "web_search_result_location":
                key = (c.url, (c.cited_text or "")[:60])
                entry = {"title": c.title, "author": urlparse(c.url).netloc, "year": None, "locator": None,
                         "standing": web_standing(c.url), "quote": (c.cited_text or "").strip(), "url": c.url,
                         "web": True}
            else:
                continue
            if key not in index:
                index[key] = len(sources) + 1
                sources.append({"n": index[key], **entry})
            if index[key] not in nums:
                nums.append(index[key])
        text = b.text
        if segments and not nums and not text.strip():
            continue
        if segments and segments[-1]["text"][-1:] in ".?!" and text[:1].isalnum():
            text = " " + text  # two cited sentences in a row come back unspaced
        segments.append({"text": text, "cites": nums})
    return {"ok": True, "segments": _mark_unsourced(segments), "sources": sources}


# a sentence ends at . ? ! before a space, but not after an initial ("V. P. Menon", "P. C. Chatterjee")
_SENTENCE_END = re.compile(r"(?<=[.?!])(?<!\b[A-Z]\.)(?=\s)|(?<=[.?!][\"'\u201d\u2019)])(?=\s)")


def _mark_unsourced(segments):
    """Flag sentences with no citation in any part. Citations split a sentence into an uncited lead-in ("Menon
    writes that") and the cited quote, so uncited text is cut at sentence ends and each piece joins its sentence."""
    pieces = []
    for s in segments:
        parts = [s["text"]] if s["cites"] else [t for t in _SENTENCE_END.split(s["text"]) if t]
        pieces += [{**s, "text": t} for t in parts]
    out, group = [], []

    def close():
        text = "".join(x["text"] for x in group)
        bare = not any(x["cites"] for x in group) and len(re.sub(r"\W", "", text)) > 25 and not text.strip().endswith(":")
        n = out[-1]["sent"] + 1 if out else 0
        out.extend({**x, "unsourced": bare and not x["cites"], "sent": n} for x in group)
        group.clear()

    for p in pieces:
        if group and p["text"].lstrip(" ").startswith("\n"):  # a new paragraph starts a new sentence
            close()
        group.append(p)
        if re.search(r"(?<!\b[A-Z])[.?!][\"'\u201d\u2019)]?\s*$", p["text"]) or p["text"].rstrip(" ").endswith("\n"):
            close()
    if group:
        close()
    return out


def plain(out):
    """The answer as text with [n] markers, then the sources (CLI and evals)."""
    if out.get("kind") == "opinion":
        return f"{out['notice']}\nTry: " + " | ".join(out["suggest"])
    text = "".join(s["text"] + "".join(f"[{n}]" for n in s["cites"]) + (" [NO SOURCE]" if s["unsourced"] else "")
                   + (" [PARTLY BACKED]" if s.get("partly") else "") for s in out["segments"])
    lines = ([f"{out['notice']} {out['answered']}", ""] if out.get("kind") == "loaded" else []) + [text.strip(), ""]
    if out.get("removed"):
        lines.append(f"({out['removed']} sentence(s) removed: their quotes did not back them)")
    if out.get("unsourced_removed"):
        lines.append(f"({out['unsourced_removed']} sentence(s) removed: no source)")
    for s in out["sources"]:
        lines.append(f"[{s['n']}] {s['title']}, {s['author']} ({s['standing']}) {s['url']}")
        for x in s["passages"]:
            lines.append(f"     {x['locator'] or ''}: \"{x['quote'][:240]}\"\n     {x['url']}")
    return "\n".join(lines)


def cli(question):
    out = answer(question)
    print(plain(out))
    m = out["meta"]
    print(f"\ntopics: {out['topics']}\nrelated: {out['related']}\n${m['usd']} · {m['seconds']}s · {m['passages']} passages · {m['model']} · {m['stop']}")
    print("plan:", json.dumps(m["plan"], indent=1))


if __name__ == "__main__":
    cli(" ".join(sys.argv[1:]))
