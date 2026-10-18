"""Postgres: everything lives in schema `indiafacts` (database, role and password file from IF_DB_* in the environment).

Sources, passages (full-text indexed), and the topic graph. Nobody's questions are stored: an answer is built
fresh from the question alone, and only its cost is kept (runs/spend.jsonl). Eval runs keep their traces in
evals/runs/, on disk.
"""
import difflib
import os
import re

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

HOST = os.environ.get("IF_DB_HOST", "127.0.0.1")
ROLE = os.environ.get("IF_DB_USER", "indiafacts")
NAME = os.environ.get("IF_DB_NAME", ROLE)
PASSWORD_FILE = os.environ.get("IF_DB_PASSWORD_FILE", "")

SCHEMA = """
create schema if not exists indiafacts;
create table if not exists indiafacts.sources (
  slug text primary key, ia text not null, title text not null, author text, year int,
  standing text not null, rights text, note text, chars int, passages int
);
create table if not exists indiafacts.passages (
  id serial primary key,
  source text not null references indiafacts.sources(slug) on delete cascade,
  seq int not null, locator text, text text not null, blocks jsonb not null,
  tsv tsvector generated always as (to_tsvector('english', text)) stored,
  unique (source, seq)
);
create index if not exists passages_tsv on indiafacts.passages using gin (tsv);
create table if not exists indiafacts.topics (
  id serial primary key, name text not null unique, kind text, passages int default 0, sources int default 0,
  tsv tsvector generated always as (to_tsvector('simple', name)) stored
);
create index if not exists topics_tsv on indiafacts.topics using gin (tsv);
create table if not exists indiafacts.passage_topics (
  passage int not null references indiafacts.passages(id) on delete cascade,
  topic int not null references indiafacts.topics(id) on delete cascade,
  primary key (passage, topic)
);
create index if not exists passage_topics_topic on indiafacts.passage_topics (topic);
create table if not exists indiafacts.topic_edges (
  a int not null references indiafacts.topics(id) on delete cascade,
  b int not null references indiafacts.topics(id) on delete cascade,
  passages int not null, sources int not null, primary key (a, b)
);
create table if not exists indiafacts.vocab (word text primary key, n int not null);
"""


def connect():
    return psycopg.connect(host=HOST, dbname=NAME, user=ROLE, password=open(PASSWORD_FILE).read().strip() if PASSWORD_FILE else None,
                           connect_timeout=5, autocommit=True, row_factory=dict_row)


def init():
    with connect() as c:
        c.execute(SCHEMA)


def load_source(src, rows, chars):
    """Replace one source's passages. A passage whose text is unchanged keeps its topics; the rest are tagged again
    (engine graph), and every new passage id needs a vector, which `embed export` reuses for unchanged text."""
    with connect() as c, c.transaction():
        kept = {}
        for r in c.execute("""select p.text, array_agg(pt.topic) topics from indiafacts.passages p
                              join indiafacts.passage_topics pt on pt.passage = p.id where p.source = %s
                              group by p.id""", (src["slug"],)).fetchall():
            kept[r["text"]] = r["topics"]
        c.execute("""insert into indiafacts.sources (slug, ia, title, author, year, standing, rights, note, chars, passages)
                     values (%(slug)s, %(ia)s, %(title)s, %(author)s, %(year)s, %(standing)s, %(rights)s, %(note)s, %(chars)s, %(n)s)
                     on conflict (slug) do update set ia = excluded.ia, title = excluded.title, author = excluded.author,
                       year = excluded.year, standing = excluded.standing, rights = excluded.rights, note = excluded.note,
                       chars = excluded.chars, passages = excluded.passages""",
                  {**{k: src.get(k) for k in ("slug", "ia", "title", "author", "year", "standing", "rights", "note")},
                   "chars": chars, "n": len(rows)})
        c.execute("delete from indiafacts.passages where source = %s", (src["slug"],))
        with c.cursor().copy("copy indiafacts.passages (source, seq, locator, text, blocks) from stdin") as cp:
            for r in rows:
                cp.write_row((src["slug"], r["seq"], r["locator"], r["text"], Jsonb(r["blocks"])))
        for r in c.execute("select id, text from indiafacts.passages where source = %s", (src["slug"],)).fetchall():
            for t in kept.get(r["text"], []):
                c.execute("insert into indiafacts.passage_topics values (%s, %s) on conflict do nothing", (r["id"], t))


# Old and OCR spellings of one word: "Jallianwala" is "Jalleanwala" in the Congress report and "Jalianwala" in
# places in the Hunter report. A long search word also matches the corpus words spelt almost like it.
VARIANT_MIN, VARIANT_CUTOFF = 7, 0.88
PER_CENT = re.compile(r"(\d)\s*(?:per\s?cent|percent)\b", re.I)


def refresh_vocab():
    """The corpus's own long words (stemmed, in 2+ passages), for spelling variants. Run after a build."""
    with connect() as c, c.transaction():
        c.execute("truncate indiafacts.vocab")
        c.execute(f"""insert into indiafacts.vocab select word, nentry from ts_stat('select tsv from indiafacts.passages')
                      where length(word) >= {VARIANT_MIN} and nentry >= 2 and word ~ '^[a-z]+$'""")


def with_variants(c, query):
    """The full-text query for `query`, each long word OR'd with the corpus words spelt almost like it."""
    query = PER_CENT.sub(r"\1", query)  # "27 per cent" is "27%" in some reports: both index as 27
    tsq = c.execute("select websearch_to_tsquery('english', %s)::text t", (query,)).fetchone()["t"]
    words = {w for w in re.findall(r"'([a-z]+)'", tsq) if len(w) >= VARIANT_MIN}
    if not words:
        return tsq
    vocab = [r["word"] for r in c.execute("select word from indiafacts.vocab where left(word, 2) = any(%s)",
                                          ([w[:2] for w in words],)).fetchall()]
    for w in words:
        alt = [v for v in difflib.get_close_matches(w, [v for v in vocab if v[:2] == w[:2]], n=6, cutoff=VARIANT_CUTOFF) if v != w]
        if alt:
            tsq = tsq.replace(f"'{w}'", "(" + " | ".join(f"'{x}'" for x in [w] + alt) + ")")
    return tsq


def search(query, limit=24, standing=None, source=None):
    """Passages ranked by Postgres full-text rank (cover density, length-normalised)."""
    sql = """select p.id, p.source, p.seq, p.locator, p.text, p.blocks, s.title, s.author, s.year, s.standing, s.ia,
                    ts_rank_cd(p.tsv, q, 32) as rank
             from indiafacts.passages p join indiafacts.sources s on s.slug = p.source,
                  cast(%(q)s as tsquery) q
             where p.tsv @@ q {extra}
             order by rank desc limit %(limit)s"""
    extra, args = "", {"limit": limit}
    if standing:
        extra += " and s.standing = %(standing)s"
        args["standing"] = standing
    if source:
        extra += " and s.slug = %(source)s"
        args["source"] = source
    with connect() as c:
        args["q"] = with_variants(c, query)
        return c.execute(sql.format(extra=extra), args).fetchall()


def search_any(words, limit=24):
    """OR-search when the AND search finds too little: any of the words, best rank first."""
    terms = [t for t in (re.sub(r"[^\w]", "", w) for w in words) if len(t) > 2]
    if not terms:
        return []
    with connect() as c:
        return c.execute("""select p.id, p.source, p.seq, p.locator, p.text, p.blocks, s.title, s.author, s.year,
                                   s.standing, s.ia, ts_rank_cd(p.tsv, q, 32) as rank
                            from indiafacts.passages p join indiafacts.sources s on s.slug = p.source,
                                 to_tsquery('english', %s) q
                            where p.tsv @@ q order by rank desc limit %s""",
                         (" | ".join(terms), limit)).fetchall()


def stats():
    with connect() as c:
        return c.execute("""select (select count(*) from indiafacts.sources) as sources,
                                   (select count(*) from indiafacts.passages) as passages,
                                   (select count(*) from indiafacts.topics) as topics,
                                   (select count(*) from indiafacts.topic_edges) as edges""").fetchone()
