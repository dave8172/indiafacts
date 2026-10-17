# indiafacts: sourced, opinion-free answers on Indian history, law, politics and geopolitics

Live at `quirkyagents.com/wizard/projects/indiafacts/` (the page, widget and Pages Function live in the site's own repo). One question in; a short
answer out where every sentence carries a citation to a passage, with the kind of evidence it is. Stateless:
nothing about the asker is read or stored, so it can't drift towards anyone's views.

## What it promises (the line we hold; README says it to the public)

- **Questions of fact only.** The planner sorts each question: fact, loaded, opinion.
  Opinion -> no answer; the notice and three questions of fact instead. Loaded ("How was the Congress
  responsible for Partition?") -> its factual core is answered ("how did Partition come about, and what did
  the Congress decide and do"), never whether the judgment holds; the page shows the question answered.
- **Every citation is checked twice:** the API guarantees each quote is in the passage; then Haiku reads each
  sentence beside its quotes (`verify` in `engine/ask.py`): unbacked sentences are removed and counted,
  partly backed ones labelled.
- **Facts, attributed; never verdicts.** Not "true/false": each claim says who records it (the law, a court,
  a commission, the government's own account, a party, a memoir). Where sources disagree, both accounts.
- **Media is never a source of truth.** Not in the corpus; web search is limited to court, statute,
  Parliament, ministry and official-data sites (`WEB_DOMAINS` in `engine/ask.py`).
- **Legal questions are about the act, not the person**; the page says it isn't legal advice.
- **Rights:** only texts public in India: author dead 60+ years (s.22), government works 60+ years old, or
  Acts/judgments/Gazette/commission reports laid before a legislature (s.52(1)(q)). In-copyright books stay
  out even when archive.org has a copy (e.g. M. O. Mathai, *Reminiscences of the Nehru Age*: author d. 1981,
  protected to the end of 2041). V. D. Savarkar (d. 1966) becomes eligible on 1 January 2027.

## Layout

| | |
|---|---|
| `sites.json` | websites for India facts, graded by what they are the original of (A issuer, B faithful copy, C research, D media: never a source); `web: true` ones are the answer's web-search domains; where to refresh what |
| `sources.json` | every source: archive.org id, standing (kind of evidence), rights basis, locator kind. The page's corpus table is built from it |
| `engine/corpus.py` | fetch OCR text (archive.org `/stream/`; the data nodes reset TLS from this box), clean, cut into ~1,200-char passages of 1-2-sentence blocks, locators (article, section, para, sitting date) |
| `engine/store.py` | Postgres schema `indiafacts` in the site DB: sources, passages (FTS), topics, passage_topics, topic_edges |
| `engine/graph.py` | topic tagging (Haiku 5.5, Batch API, 6 passages per request, Wikipedia titles as names) and co-occurrence edges |
| `engine/ask.py` | plan (Haiku 5.5) → retrieve (FTS + meaning + graph; 16 max, 3 per source) → answer (Sonnet 5.5 by default, `llm.ANSWER_MODEL`; search results with citations, `search_corpus` tool, official-site web search) → render → widen web quotes (`pages.py`) → per-sentence check (Haiku) |
| `engine/pages.py` | web search quotes only ~150 chars: refetch the page (direct, else the API's web_fetch), keep ~900 chars around the quote; 10 s deadline, late cost still logged as `answer` |
| `engine/server.py` | HTTP: `POST /indiafacts/ask`, `GET /indiafacts/health`; `X-Engine-Key`; daily dollar cap `IF_DAILY_USD` |
| `evals/` | `questions.json` (mirrored pairs, fact, loaded and opinion questions, false premises, web-only; `expect` = the kind), `run.py` (system vs plain Opus+web baseline; code scores + a Sonnet 5.5 support judge, not the run-time checker) |
| `deploy/` | Dockerfile, `up.sh` (container `indiafacts-engine` on network `das`, 127.0.0.1:8091) |
| `engine/embed.py` | meaning search: EmbeddingGemma 300M ONNX (pinned revision); corpus embedded on a GPU machine (`python -m engine.embed run`, needs only `requirements-embed.txt`), merged and searched on the server |
| the site | in the site's repo: page `wizard/projects/indiafacts.md` (`widget-indiafacts`, reads this repo's `sources.json` at build), widget JS/CSS, Pages Function `functions/wizard/api/indiafacts.js`, share card `#indiafacts` |

## Run

```
.venv/bin/python -m engine fetch|build|search "<q>"|stats      # free
.venv/bin/python -m engine graph --pilot 30 | --submit | --collect | --link
.venv/bin/python -m engine ask "<question>"
.venv/bin/python -m evals.run [--only id,id] [--arms system,baseline] [--cap 10]
deploy/up.sh
```

Settings come from the environment or a gitignored `.env` (IF_DB_*, IF_ANTHROPIC_KEY_FILE, IF_SECRETS for
`deploy/up.sh`). Use an Anthropic key of its own in a capped Console workspace. Spend ledger: `runs/spend.jsonl`.

## Known limits

- No page numbers: archive.org's page index lives on the data nodes this box can't reach. Citations carry
  the source's own numbering where it has one, and a search-inside link that opens the scan at the quote.
- OCR slips show in quotes; the model is told not to correct sources silently.
- Corpus ends where its books end: later events and court rulings come only from the restricted web search.
- **Most .gov.in sites time out from data-centre servers, even in India** (checked 2026-10-17: sci.gov.in
  judgment PDFs, egazette, upload.indiacode, legislative.gov.in, sansad.in, ncrb). The API's web_fetch reaches some. New documents come from their archive.org
  mirrors (DLI, Public Resource, `rsdebate.nic.in.*`, `eparlib.nic.in.*` items) or a connection in India.
  Answer-time web search runs on Anthropic's side and is unaffected.
- **Passage ids change when a source is rebuilt**: re-run `embed export` (it sends only new or changed text).
- **Rights check before adding Parliament debates after 1966**: not over 60 years, and not plainly covered
  by s.52(1)(q). Commission reports laid before Parliament are.
- Bonds and Kesavananda answers stay weakly supported (0.2-0.4): their citations point to sci.gov.in judis PDFs no
  fetcher reaches. Fix later: add landmark judgments to the corpus (s.52(1)(q)), downloaded from the laptop.

