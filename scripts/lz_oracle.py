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
        h_inf = max(b3 - d23 / (R**alpha - 1), 0.0)
    return {
        "alpha": alpha,
        "h_inf": h_inf,
        "bpc_128": b1,
        "bpc_2048": b2,
        "bpc_32768": b3,
        "n_docs": n_docs,
        "n_bytes": len(corpus),
    }
