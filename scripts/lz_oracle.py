"""LZ data oracle for (alpha, H_inf) scoring — agentic-data port.

Replicates the protocol described in data_format.md (formal-math survey):
  - BPC(n) is measured by compressing the corpus in independent n-byte chunks
    (the compressor's usable context is capped at the chunk length).
  - 3-point analytical estimation at n1=128, n2=2048, n3=32768 (geometric, r=16)
    under the scaling model  BPC(n) = H_inf + c * n^(-alpha):
        alpha = log_r( (B1-B2) / (B2-B3) )
        H_inf = B3 - (B2-B3) / (r^alpha - 1)
  - Corpus = up to ~1500 documents or 8 MB, docs joined with "\n\n".

alpha: scaling exponent of BPC vs context length (higher = more long-range
       structure/templating exploitable by an LZ compressor).
H_inf: extrapolated incompressible entropy in bits-per-character.
"""

import math

import zstandard as zstd

N_POINTS = (128, 2048, 32768)
R = 16  # geometric ratio between the three context sizes
MAX_DOCS = 1500
MAX_BYTES = 8 * 1024 * 1024
ZSTD_LEVEL = 19

_params = zstd.ZstdCompressionParameters.from_level(
    ZSTD_LEVEL, write_checksum=False, write_content_size=False, write_dict_id=False
)
_cctx = zstd.ZstdCompressor(compression_params=_params)
_EMPTY_OVERHEAD = len(_cctx.compress(b""))


def _bpc_at(corpus: bytes, n: int) -> float:
    """Bits-per-character when compression context is capped at n bytes."""
    total_bits = 0
    total_bytes = 0
    for i in range(0, len(corpus) - n + 1, n):  # drop ragged tail chunk
        chunk = corpus[i : i + n]
        comp = len(_cctx.compress(chunk)) - _EMPTY_OVERHEAD
        total_bits += 8 * max(comp, 1)
        total_bytes += len(chunk)
    if total_bytes == 0:  # corpus shorter than n: single padded measurement
        comp = len(_cctx.compress(corpus)) - _EMPTY_OVERHEAD
        return 8 * max(comp, 1) / max(len(corpus), 1)
    return total_bits / total_bytes


def build_corpus(docs) -> bytes:
    """Join up to MAX_DOCS docs (cap MAX_BYTES) with blank-line separators."""
    parts, size = [], 0
    for d in docs:
        if not d:
            continue
        b = d.encode("utf-8", errors="replace")
        parts.append(b)
        size += len(b) + 2
        if len(parts) >= MAX_DOCS or size >= MAX_BYTES:
            break
    return b"\n\n".join(parts)[:MAX_BYTES]


def score(docs):
    """Return dict with alpha, H_inf, the three BPC points, and corpus stats."""
    corpus = build_corpus(docs)
    n_docs = corpus.count(b"\n\n") + 1 if corpus else 0
    b1, b2, b3 = (_bpc_at(corpus, n) for n in N_POINTS)
    d12, d23 = b1 - b2, b2 - b3
    if d12 <= 0 or d23 <= 0:
        alpha, h_inf = float("nan"), b3  # degenerate / non-monotone corpus
    else:
        alpha = math.log(d12 / d23) / math.log(R)
        # NO CLAMP (fixed 2026-06-07): report the raw extrapolated floor.
        # A negative value is an honest signal that the BPC curve has not
        # flattened within the 32 KB window, so the n->inf floor is
        # unresolved by 3-point extrapolation — do NOT hide it behind max(.,0).
        # For a robust, directly-measured content metric use bpc_32768.
        h_inf = b3 - d23 / (R**alpha - 1)
    return {
        "alpha": alpha,
        "h_inf": h_inf,            # raw, unclamped (may be negative = unresolved)
        "bpc_32k": b3,             # directly-measured content metric (no fit)
        "bpc_128": b1,
        "bpc_2048": b2,
        "bpc_32768": b3,
        "n_docs": n_docs,
        "n_bytes": len(corpus),
    }


# --- v2: multi-point floor fit (added 2026-06-07) -------------------------
# The 3-point analytic score() caps context at 32768 B. For highly compressible
# agentic data the BPC curve is still falling steeply there, so the power-law
# floor extrapolates negative and gets clamped to 0 — collapsing a real spread
# of floors (genuinely ~0 vs ~0.4+) into a single artifactual zero. score_v2
# measures more (and larger) context points and fits H_inf by least squares,
# resolving the floor where score() clamped. See reports/hinf_clamp_fix.md.
N_POINTS_V2 = (128, 512, 2048, 8192, 32768, 131072, 524288)
MIN_CHUNKS = 8  # require >= this many chunks at a context size to trust its BPC


def _fit_floor(ns, bpcs):
    """Fit BPC(n) = H_inf + c * n^(-alpha) by scanning H_inf, linear in log-log.
    Returns (h_inf, alpha, r2). H_inf is NOT clamped — negative means the curve
    has not flattened within the measured window (floor unresolved/<=0)."""
    import numpy as np
    ns = np.asarray(ns, float)
    b = np.asarray(bpcs, float)
    x = np.log(ns)
    lo, hi = -1.0, float(b.min()) - 1e-3
    best = (-1e9, 0.0, 0.0)  # (r2, h_inf, alpha)
    for hf in np.linspace(lo, hi, 600):
        y = np.log(b - hf)
        A = np.vstack([x, np.ones_like(x)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ sol
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss if ss > 0 else 0.0
        if r2 > best[0]:
            best = (r2, hf, -sol[0])
    r2, hf, alpha = best
    return hf, alpha, r2


def score_v2(docs):
    """Robust floor measurement: 7 context points to 524 KB + least-squares fit.
    Returns h_inf_resolved (unclamped real floor; <=0 => unresolved/fully
    compressible), h_inf (clamped >=0 for ranking), alpha, fit_r2, the BPC curve,
    and corpus stats."""
    corpus = build_corpus(docs)
    n_docs = corpus.count(b"\n\n") + 1 if corpus else 0
    ns, bpcs = [], []
    for n in N_POINTS_V2:
        if len(corpus) // n < MIN_CHUNKS:
            break
        ns.append(n)
        bpcs.append(_bpc_at(corpus, n))
    if len(ns) >= 4 and all(bpcs[i] > bpcs[i + 1] for i in range(len(bpcs) - 1)):
        h_res, alpha, r2 = _fit_floor(ns, bpcs)
    else:  # too few points or non-monotone — fall back to 3-point analytic
        s = score(docs)
        h_res, alpha, r2 = s["h_inf"], s["alpha"], float("nan")
    return {
        "h_inf_resolved": h_res,
        "h_inf": h_res,            # raw, unclamped (negative = unresolved floor)
        "alpha": alpha,
        "fit_r2": r2,
        "bpc_curve": dict(zip(ns, bpcs)),
        "n_points": len(ns),
        "n_docs": n_docs,
        "n_bytes": len(corpus),
    }
