# indiafacts: sourced, opinion-free answers on Indian history, law, politics and geopolitics

**What:** one question in; an answer out where every sentence cites its source (one number per source, its quoted passages listed under it), with the kind of evidence it is. Brief for a narrow question; for a broad one a fuller 250-480 word account, a paragraph per main source, opening with a cited sentence naming it. Stateless: nothing about the asker is read or stored, so it can't drift towards anyone's views.

**State:** live at `quirkyagents.com/wizard/projects/indiafacts/`. The engine runs here as container `indiafacts-engine`; the page, widget and Pages Function live in the site's own repo.

## Run

```
.venv/bin/python -m engine fetch|build|search "<q>"|stats      # free
.venv/bin/python -m engine graph --pilot 30 | --submit | --collect | --link
.venv/bin/python -m engine ask "<question>"
.venv/bin/python -m evals.run [--only id,id] [--arms system,baseline] [--cap 10]
.venv/bin/python -m evals.recall [--only id,id] [--replan]      # do the needed passages reach the writer (~$0.10 replanned, $0 on saved plans)
deploy/up.sh
```

**`evals.run` spends from the live engine's key:** the full default (37 questions x 2 arms) costs ~$14, ~45 min at `--workers 1`. It once hit the Console workspace's monthly limit and took the public site down: check `runs/spend.jsonl` for the month first.

Settings come from the environment or a gitignored `.env` (`IF_DB_*`, `IF_ANTHROPIC_KEY_FILE`, `IF_SECRETS` for `deploy/up.sh`; `IF_ANTHROPIC_KEY` names the key file, default `$IF_SECRETS/indiafacts-anthropic.key`). Use a key in a capped Console workspace. On quirkyagents.com it runs on the site's one Anthropic key. Spend ledger: `runs/spend.jsonl`.

## Rules: what it promises (README says it to the public)

- **Questions of fact only.** The planner sorts each question: fact, loaded, opinion. Opinion → no answer; the notice and three questions of fact instead. Loaded → its factual core is answered, never whether the judgment holds; the page shows the question answered.
- **Every citation is checked twice:** the API guarantees each quote is in the passage; then Haiku reads each sentence beside its quotes (`verify` in `engine/ask.py`). Unbacked sentences are removed and counted, partly backed ones labelled.
- **Facts, attributed; never verdicts.** Not "true/false": each claim says who records it (the law, a court, a commission, the government's own account, a party, a memoir). Where sources disagree, both accounts.
- **Media is never a source of truth.** Not in the corpus; web search is limited to court, statute, Parliament, ministry and official-data sites (`WEB_DOMAINS` in `engine/ask.py`).
- **Legal questions are about the act, not the person;** the page says it isn't legal advice.
- **Rights:** only texts public in India: author dead 60+ years (s.22), government works 60+ years old, or Acts/judgments/Gazette/commission reports laid before a legislature (s.52(1)(q)). In-copyright books stay out even when archive.org has a copy (e.g. M. O. Mathai, *Reminiscences of the Nehru Age*: protected to the end of 2041). V. D. Savarkar (d. 1966) becomes eligible on 1 January 2027.
- **Rights check before adding Parliament debates after 1966:** not over 60 years, and not plainly covered by s.52(1)(q). Commission reports laid before Parliament are.

## Layout

| | |
|---|---|
| `sites.json` | websites graded by what they are the original of (A issuer, B faithful copy, C research, D media: never a source); `web: true` ones are the answer's web-search domains |
| `sources.json` | every source: archive.org id, standing, rights basis, locator kind. The page's corpus table is built from it |
| `engine/corpus.py` | fetch OCR text (archive.org `/stream/`), clean, cut into ~1,200-char passages, locators (article, section, para, sitting date; an Act's sections = the longest rising run of section numbers, so footnote numbers aren't sections) |
| `engine/store.py` | Postgres schema `indiafacts`: sources, passages (FTS), topics, passage_topics, topic_edges, vocab (a search word also matches words spelt almost like it, for old and OCR spellings) |
| `engine/graph.py` | topic tagging (Haiku 5.5, Batch API, Wikipedia titles as names) and co-occurrence edges |
| `engine/ask.py` | plan (Haiku; an opinion stops here, `suggest()` offers three questions of fact) → retrieve (FTS + meaning + graph; 16 max, 3 per source; depth full: 24, 6) → answer (Sonnet 5.5 by default, `llm.ANSWER_MODEL`; `search_corpus` tool, official-site web search) → render → widen web quotes → per-sentence check (an uncited sentence survives only as a gap statement, `_ABSENCE`) → `by_source` |
| `engine/pages.py` | web search quotes only ~150 chars: refetch the page, keep ~900 chars around the quote; 10 s deadline, late cost still logged |
| `engine/embed.py` | meaning search: EmbeddingGemma 300M ONNX (pinned revision); corpus embedded on a GPU machine (`python -m engine.embed run`, needs only `requirements-embed.txt`), merged and searched on the server |
| `engine/server.py` | `POST /indiafacts/ask`, `GET /indiafacts/health`; `X-Engine-Key`; daily dollar cap `IF_DAILY_USD`, counting only answers it served (ledger `by: public`; evals and the CLI are `local`) |
| `evals/` | `questions.json` (mirrored pairs, fact, loaded and opinion questions, false premises, web-only), `run.py` (system vs plain Opus+web baseline; code scores + a Sonnet 5.5 support judge, not the run-time checker), `recall.py` |
| `deploy/` | Dockerfile, `up.sh` (container on network `engines`, 127.0.0.1:8091; the tunnel is started from the site repo's `scripts/tunnel-up.sh`) |
| the site | in its own repo: page `wizard/projects/indiafacts.md` (question box + corpus table read from this repo's `sources.json` at build), widget JS/CSS, Pages Function `functions/wizard/api/indiafacts.js`, share card `#indiafacts` |

## Gotchas and known limits

- **Passage ids change when a source is rebuilt:** unchanged text keeps its topics (`load_source`) and its vector (`embed export` moves it to the new id); re-run `embed export` for the rest, and `graph --submit` to tag changed text.
- **Most .gov.in sites time out from data-centre servers, even in India** (sci.gov.in judgment PDFs, egazette, upload.indiacode, legislative.gov.in, sansad.in, ncrb). The API's web_fetch reaches some. New documents come from archive.org mirrors (DLI, Public Resource, `rsdebate.nic.in.*`, `eparlib.nic.in.*`) or a connection in India. Answer-time web search runs on Anthropic's side and is unaffected.
- **archive.org's data nodes reset TLS from this box,** so there are no page numbers: citations carry the source's own numbering where it has one, plus a search-inside link that opens the scan at the quote.
- OCR slips show in quotes; the model is told not to correct sources silently.
- The corpus ends where its books end: later events and rulings come only from the restricted web search.
- Bonds and Kesavananda answers stay weakly supported (0.2-0.4): their citations point to sci.gov.in judis PDFs no fetcher reaches. Fix later: add landmark judgments to the corpus (s.52(1)(q)), downloaded from a laptop.

## Pointers

`README.md` (public description, sources, method) · `CONTRIBUTING.md`.
