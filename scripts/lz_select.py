"""LZ-Select: a 3-stage reusable pipeline for agentic-trajectory corpora.

Composes three stages over the doc-per-episode serialization defined in the
REGISTRY (score_agentic_datasets.py):

  1. STRIP  — remove cross-episode boilerplate *within* each episode before
     selection. Two configurable rules (both on by default):
       (a) drop duplicate system turn(s): the leading turn block(s) that are
           exact/near-duplicates of the modal leading block across a sample;
       (b) collapse boilerplate lines: drop lines that appear verbatim in
           >THRESH (default 0.5) fraction of sampled episodes.
  2. SELECT — per-episode template_gain (sibling-dictionary compression) and
     loop_gain (conditional compression of 2nd half | 1st half), reusing the
     episode_filter_pilot.py logic; drop the worst quartile of *each* score.
  3. GATE   — score the resulting corpus with lz_oracle and report (alpha,
     H_inf).

Episodes are rendered as turns ("[role]\\ntext") joined by blank lines and
lines joined by "\\n" — so a "turn" is a "\\n\\n"-separated block and a "line"
is a "\\n"-separated line, matching the serializers exactly.

Usage:
    python scripts/lz_select.py --slug kwai-klear-mini-swe-66k -n 300 \\
        --stages raw,strip,select,strip+select \\
        --out data/lz_select_pilot.csv
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter

import zstandard as zstd

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
import score_agentic_datasets as sad

DICT_SIZE = 110 * 1024
LEVEL = 19


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_episodes(slug, n):
    """Load first-N serialized episodes (strings) for a REGISTRY slug."""
    entry = next(e for e in sad.REGISTRY if e[4] == slug)
    path, cfg, splits, ser = entry[:4]
    group_key = entry[5] if len(entry) > 5 else None
    drop_cols = entry[6] if len(entry) > 6 else None
    docs = []
    for doc, _ in sad.iter_docs(path, cfg, splits, ser, group_key=group_key,
                                drop_cols=drop_cols):
        if doc:
            docs.append(doc)
        if len(docs) >= n:
            break
    return docs


# ---------------------------------------------------------------------------
# STRIP stage
# ---------------------------------------------------------------------------
def _norm(s):
    return " ".join(s.split())


def strip_episodes(docs, drop_system=True, drop_boilerplate_lines=True,
                   line_thresh=0.5, sys_thresh=0.5):
    """Remove cross-episode boilerplate within each episode.

    (a) drop_system: if a leading turn block (\\n\\n-separated) is the modal
        leading block across >= sys_thresh of episodes, drop that leading block
        from every episode that starts with it (matched on whitespace-normalized
        text, so near-duplicates with reformatting collapse together). Repeats
        for the next leading block too (some corpora stack system + a constant
        instruction turn).
    (b) drop_boilerplate_lines: drop any line that appears verbatim
        (whitespace-normalized) in > line_thresh of episodes.
    """
    n = len(docs)
    turns = [d.split("\n\n") for d in docs]

    if drop_system:
        # Peel leading blocks while the modal leading block is shared by a
        # >= sys_thresh majority (handles system turn, then a constant
        # instruction turn, etc.).
        peeled = 0
        while True:
            leads = [(_norm(t[0]) if t else None) for t in turns]
            counts = Counter(l for l in leads if l is not None)
            if not counts:
                break
            modal, freq = counts.most_common(1)[0]
            if freq < sys_thresh * n or not modal:
                break
            for t, lead in zip(turns, leads):
                if lead == modal and t:
                    t.pop(0)
            peeled += 1
            if peeled >= 4:  # safety: never strip an unbounded prefix
                break

    docs = ["\n\n".join(t) for t in turns]

    if drop_boilerplate_lines:
        # Count, per episode, the set of normalized lines, then drop globally
        # frequent ones. Set-per-episode so a line repeated inside one episode
        # (a loop) does not by itself count as cross-episode boilerplate.
        line_doc_freq = Counter()
        for d in docs:
            for ln in set(_norm(x) for x in d.split("\n")):
                if ln:
                    line_doc_freq[ln] += 1
        cut = line_thresh * n
        boiler = {ln for ln, c in line_doc_freq.items() if c > cut}
        if boiler:
            new = []
            for d in docs:
                kept = [x for x in d.split("\n") if _norm(x) not in boiler]
                new.append("\n".join(kept))
            docs = new

    return docs


# ---------------------------------------------------------------------------
# SELECT stage  (reuses episode_filter_pilot.py logic)
# ---------------------------------------------------------------------------
def _csize(cctx, data):
    return len(cctx.compress(data))


def episode_scores(docs):
    """Return (template_gain, loop_gain) per episode (episode_filter_pilot.py)."""
    enc = [d.encode("utf-8", errors="replace") for d in docs]
    plain = zstd.ZstdCompressor(level=LEVEL)
    tg, lg = [], []
    n = len(enc)
    for i, e in enumerate(enc):
        n_sib = min(40, n - 1)
        if n_sib >= 5:
            sibs = [enc[j] for j in random.Random(i).sample(
                [j for j in range(n) if j != i], n_sib)]
            try:
                d = zstd.train_dictionary(DICT_SIZE, sibs)
                cdict = zstd.ZstdCompressor(level=LEVEL, dict_data=d)
                template_gain = 1 - _csize(cdict, e) / max(_csize(plain, e), 1)
            except Exception:
                template_gain = 0.0
        else:
            template_gain = 0.0
        h = len(e) // 2
        e1, e2 = e[:h], e[h:]
        cond = _csize(plain, e1 + e2) - _csize(plain, e1)
        loop_gain = 1 - cond / max(_csize(plain, e2), 1)
        tg.append(template_gain)
        lg.append(loop_gain)
    return tg, lg


def select_episodes(docs, return_idx=False):
    """Drop worst quartile of template_gain and of loop_gain (high = boilerplate
    / self-looping). Kept = below both 75th-percentile cuts."""
    tg, lg = episode_scores(docs)
    tg_cut = sorted(tg)[int(0.75 * len(tg))]
    lg_cut = sorted(lg)[int(0.75 * len(lg))]
    kept_idx = [i for i in range(len(docs))
                if tg[i] <= tg_cut and lg[i] <= lg_cut]
    kept = [docs[i] for i in kept_idx]
    if return_idx:
        return kept, kept_idx, (tg_cut, lg_cut)
    return kept


# ---------------------------------------------------------------------------
# GATE stage
# ---------------------------------------------------------------------------
def gate(docs):
    return lz_oracle.score(docs)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
def run_stage(stage, docs, strip_kw):
    """Apply a named stage transform and return (corpus_docs, n_episodes)."""
    if stage == "raw":
        out = list(docs)
    elif stage == "strip":
        out = strip_episodes(docs, **strip_kw)
    elif stage == "select":
        out = select_episodes(docs)
    elif stage == "strip+select":
        out = select_episodes(strip_episodes(docs, **strip_kw))
    else:
        raise ValueError(f"unknown stage {stage}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="kwai-klear-mini-swe-66k")
    ap.add_argument("-n", "--n-episodes", type=int, default=300)
    ap.add_argument("--stages", default="raw,strip,select,strip+select")
    ap.add_argument("--out", default="data/lz_select_pilot.csv")
    # strip config
    ap.add_argument("--no-drop-system", action="store_true")
    ap.add_argument("--no-drop-lines", action="store_true")
    ap.add_argument("--line-thresh", type=float, default=0.5)
    args = ap.parse_args()

    strip_kw = dict(
        drop_system=not args.no_drop_system,
        drop_boilerplate_lines=not args.no_drop_lines,
        line_thresh=args.line_thresh,
    )

    print(f"loading {args.n_episodes} episodes for {args.slug} ...", flush=True)
    docs = load_episodes(args.slug, args.n_episodes)
    print(f"loaded {len(docs)} episodes "
          f"({sum(len(d) for d in docs)/1e6:.2f} MB)", flush=True)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    fields = ["slug", "stage", "n_episodes_in", "n_episodes_corpus",
              "alpha", "h_inf", "bpc_128", "bpc_2048", "bpc_32768",
              "n_bytes", "mean_doc_bytes"]
    rows = []
    for stage in stages:
        corpus = run_stage(stage, docs, strip_kw)
        res = gate(corpus)
        row = dict(
            slug=args.slug, stage=stage, n_episodes_in=len(docs),
            n_episodes_corpus=len(corpus),
            alpha=round(res["alpha"], 4) if res["alpha"] == res["alpha"] else "nan",
            h_inf=round(res["h_inf"], 4),
            bpc_128=round(res["bpc_128"], 4),
            bpc_2048=round(res["bpc_2048"], 4),
            bpc_32768=round(res["bpc_32768"], 4),
            n_bytes=res["n_bytes"],
            mean_doc_bytes=round(sum(len(d) for d in corpus) / max(len(corpus), 1)),
        )
        rows.append(row)
        a = row["alpha"]
        print(f"[{stage:>12}] n={row['n_episodes_corpus']:>4} "
              f"alpha={a} H_inf={row['h_inf']} bytes={row['n_bytes']}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
