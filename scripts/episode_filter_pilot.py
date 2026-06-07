"""Episode-level data filter pilot (LZ-Select Tier B).

Two per-episode scores, each operationalizing a registry finding:
  - template_gain (finding 8/17): 1 - C_dict(e)/C(e), where the zstd
    dictionary is trained on *other* episodes of the same dataset. High =
    episode is mostly cross-episode boilerplate.
  - loop_gain (finding 12): 1 - C(e2|e1)/C(e2), second half conditioned on
    first half via concatenation. High = self-repeating episode (retry loop).

Validation: filter the bottom tail of either score on a borderline corpus
(Kwai-Klear 66k, registry H_inf=0.26) and compare the kept corpus's oracle
(alpha, H_inf) against a same-size random subset.

Outputs data/episode_filter_pilot.csv (per-episode scores) and prints the
kept-vs-random oracle comparison.
"""

import csv
import os
import random
import sys

import zstandard as zstd

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
import score_agentic_datasets as sad

SLUG = "kwai-klear-mini-swe-66k"
N_EPISODES = 300
DICT_SIZE = 110 * 1024
LEVEL = 19


def csize(cctx, data):
    return len(cctx.compress(data))


def main():
    entry = next(e for e in sad.REGISTRY if e[4] == SLUG)
    path, cfg, splits, ser = entry[:4]
    docs = []
    for doc, _ in sad.iter_docs(path, cfg, splits, ser):
        docs.append(doc.encode("utf-8", errors="replace"))
        if len(docs) >= N_EPISODES:
            break
    print(f"sampled {len(docs)} episodes", flush=True)

    plain = zstd.ZstdCompressor(level=LEVEL)
    rows = []
    for i, e in enumerate(docs):
        # dictionary from 40 sibling episodes (excluding self)
        sibs = [docs[j] for j in random.Random(i).sample(
            [j for j in range(len(docs)) if j != i], 40)]
        try:
            d = zstd.train_dictionary(DICT_SIZE, sibs)
            cdict = zstd.ZstdCompressor(level=LEVEL, dict_data=d)
            template_gain = 1 - csize(cdict, e) / csize(plain, e)
        except Exception:
            template_gain = 0.0
        h = len(e) // 2
        e1, e2 = e[:h], e[h:]
        cond = csize(plain, e1 + e2) - csize(plain, e1)
        loop_gain = 1 - cond / max(csize(plain, e2), 1)
        rows.append(dict(idx=i, n_bytes=len(e),
                         template_gain=round(template_gain, 4),
                         loop_gain=round(loop_gain, 4)))
        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}", flush=True)

    with open("data/episode_filter_pilot.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "n_bytes", "template_gain",
                                          "loop_gain"])
        w.writeheader()
        w.writerows(rows)

    # filter: drop worst quartile on each score
    tg = sorted(r["template_gain"] for r in rows)
    lg = sorted(r["loop_gain"] for r in rows)
    tg_cut = tg[int(0.75 * len(tg))]
    lg_cut = lg[int(0.75 * len(lg))]
    kept_idx = [r["idx"] for r in rows
                if r["template_gain"] <= tg_cut and r["loop_gain"] <= lg_cut]
    print(f"kept {len(kept_idx)}/{len(rows)} "
          f"(template_gain<= {tg_cut:.3f}, loop_gain<= {lg_cut:.3f})", flush=True)

    def oracle(idxs, label):
        sel = [docs[i].decode("utf-8", errors="replace") for i in idxs]
        res = lz_oracle.score(sel)
        print(f"{label}: n={len(idxs)} alpha={res['alpha']:.3f} "
              f"H_inf={res['h_inf']:.3f} bytes={res['n_bytes']}", flush=True)
        return res

    oracle(kept_idx, "FILTERED ")
    rnd = random.Random(0).sample(range(len(docs)), len(kept_idx))
    oracle(rnd, "RANDOM   ")
    worst = [r["idx"] for r in rows
             if r["template_gain"] > tg_cut or r["loop_gain"] > lg_cut]
    oracle(worst[:len(kept_idx)] if len(worst) >= 20 else worst, "REJECTED ")


if __name__ == "__main__":
    main()
