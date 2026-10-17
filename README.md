# indiafacts

Ask a question about Indian history, law, politics or geopolitics. Get an answer in which every sentence
links to its source, with the exact passages quoted and the kind of evidence each source is: short for a narrow
question, a fuller account (a paragraph per main source) for a broad one. No opinions.

Runs at [quirkyagents.com/wizard/projects/indiafacts](https://quirkyagents.com/wizard/projects/indiafacts/) once launched.

## What it promises

- **Questions of fact only.** Each question is sorted first:
  - **fact** ("How did Hyderabad become part of India?"): answered.
  - **loaded**, a question of fact wrapped in a judgment ("How was party X responsible for Partition?"): the
    factual core is answered (how Partition came about, and what that party decided and did), and the answer
    shows the question it actually answered. It never says whether the judgment holds.
  - **opinion** ("Was this leader good?"): not answered. It says it gives facts, not opinions, and offers
    questions of fact on the same subject.
- **Every sentence is cited, and every citation is checked twice.** Passages go to the model as search
  results with citations on, so every quote shown is text that really is in the source (the API enforces
  this, not the model). Web search quotes only ~150 characters of a page, so each cited page is fetched again
  and its quote widened to the text around it; a page that can't be fetched keeps the short quote. Then a second model reads each sentence beside its quotes. A sentence its quotes don't
  back is removed and counted, and one they back only in part is labelled.
- **It says what kind of evidence each source is:** the law itself, a court record, a commission report,
  an official record, a first-hand account, writing from the time that argues a view, history written
  later, a party document. A government report is the government's account. A memoir is its writer's.
- **No media and no Wikipedia as sources.** News is a report of claims; Wikipedia can be edited by anyone and
  summarises other sources. Web search at answer time is limited to sites that issue records (courts,
  Parliament, the Gazette, ministries, official statistics) and faithful copies of them: see `sites.json`.
- **No memory of you.** Each question is answered on its own; nothing about the asker is read or stored.

## Only what it may legally use

The corpus holds only texts that are free to use in India:
- books whose authors died more than 60 years ago (Copyright Act, 1957, s.22);
- government publications more than 60 years old (s.28);
- Acts, the Gazette, judgments, and reports of government-appointed commissions laid before a legislature,
  which anyone may reproduce (s.52(1)(q)).

Many books still in copyright have copies on archive.org and other sites, uploaded by people with no right to
share them. indiafacts does not use them. That is not a judgment on whether they are true: using them would
break copyright law. Every source and its rights basis is in `sources.json`.

## How it works

| Step | Where | What |
|---|---|---|
| Corpus | `engine/corpus.py` | OCR text of each source from archive.org, cleaned, cut into ~1,200-character passages made of 1-2-sentence blocks; each passage keeps the source's own numbering (Article, section, paragraph, sitting date) |
| Store | `engine/store.py` | Postgres: sources, passages with full-text search, topics, topic links |
| Graph | `engine/graph.py` | each passage tagged with the topics it is about (English Wikipedia article titles, used only as names); topics linked where they appear together, so one event can be checked across a report, a debate and a memoir |
| Meaning search | `engine/embed.py` | every passage as a vector from EmbeddingGemma (300M, multilingual, ONNX); a question in Hindi finds English passages. The corpus is embedded once on any GPU machine; the server only embeds questions |
| Answer | `engine/ask.py` | sort the question, search by words and by meaning, spread the picks across sources, answer with citations, check every cited sentence |
| Serve | `engine/server.py` | small HTTP service with a daily spending cap |
| Evals | `evals/` | mirrored pairs (the same question framed from each side), opinion and loaded questions, false premises; the system against a plain model-plus-web-search baseline |

## Run it

Needs Python 3.11+, Postgres, and an Anthropic API key.

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cat > .env <<EOF
IF_DB_USER=indiafacts
IF_DB_NAME=indiafacts
IF_DB_PASSWORD_FILE=/path/to/db.password
IF_ANTHROPIC_KEY_FILE=/path/to/anthropic.key
EOF
.venv/bin/python -m engine fetch          # download the sources' text
.venv/bin/python -m engine build          # passages into Postgres
.venv/bin/python -m engine graph --submit # topic tags (Batch API), then --collect and --link
.venv/bin/python -m engine embed export   # passages that need vectors -> data/passages.jsonl
.venv/bin/python -m engine ask "How did Hyderabad become part of India?"
```

**Embedding on a GPU machine** (no database or key needed there; copy `data/passages.jsonl` over and the
result back):

```
pip install -r requirements-embed.txt
python -m engine.embed run data/passages.jsonl --provider dml     # Windows (DirectML); cuda or cpu elsewhere
# back on the server:
python -m engine embed merge data/vectors-new.npz && python -m engine embed check
```

## Contributing

See `CONTRIBUTING.md`. The two rules that matter most: a new source needs a rights basis, and nothing may
make answers take a side.

## Licence

Code: MIT. The texts belong to the public domain or are official texts as described above; this repository
holds no corpus text.
