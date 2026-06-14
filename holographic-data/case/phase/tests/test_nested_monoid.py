import numpy as np
import torch
from data_nested_monoid import NestedMonoidGenerator

VOCAB = 31 + 512 + 3 + 6  # p + n_names + specials + filler = 552


def test_shape_and_vocab():
    g = NestedMonoidGenerator(p=31, n_names=512, n_filler=6)
    assert g.vocab_size == VOCAB
    x, y, m = g.generate_batch(batch_size=4, seq_len=128, beta=1.0, gamma=0.2)
    assert x.shape == (4, 128) and y.shape == (4, 128) and m.shape == (4, 128)
    assert x.max().item() < g.vocab_size and x.min().item() >= 0


def test_determinism():
    g = NestedMonoidGenerator(p=31, n_names=512, n_filler=6)
    np.random.seed(123)
    x1, _, _ = g.generate_batch(2, 96, beta=0.5, gamma=0.3)
    np.random.seed(123)
    x2, _, _ = g.generate_batch(2, 96, beta=0.5, gamma=0.3)
    assert torch.equal(x1, x2)


def test_answer_is_correct_affine_fold():
    g = NestedMonoidGenerator(p=31, n_names=512, n_filler=6)
    np.random.seed(7)
    x, y, m, meta = g.generate_batch(8, 320, beta=0.8, gamma=0.1, return_meta=True)
    for bi, md in enumerate(meta):
        v = md["x0"]
        for payload in md["applied"]:          # apply ops in recall order
            if md["op_kind"] == "perm":
                v = int(g.perms[payload, v])
            else:
                a, b = payload
                v = (a * v + b) % g.p
        assert v == md["answer"]
        # ANS token is the last masked position and equals the answer value token
        ans_pos = int(np.where(m[bi].numpy() > 0)[0][-1])
        assert int(x[bi, ans_pos].item()) == md["answer"]
        assert int(y[bi, ans_pos].item()) == md["answer"]


def test_mask_marks_only_value_positions():
    g = NestedMonoidGenerator(p=31, n_names=512, n_filler=6)
    np.random.seed(1)
    x, y, m, meta = g.generate_batch(4, 320, beta=0.8, gamma=0.1, return_meta=True)
    masked_tokens = x[m > 0]
    assert int(masked_tokens.max().item()) < g.p   # only value tokens are masked
    for bi, md in enumerate(meta):
        assert int(m[bi].sum().item()) == md["n_uses"] + 1  # results + answer


def _filler_fraction(g, x):
    return float((x >= g.filler_tokens.min()).float().mean().item())


def test_gamma_controls_filler_fraction():
    g = NestedMonoidGenerator(p=31, n_names=512, n_filler=6)
    np.random.seed(0)
    x_lo, _, _ = g.generate_batch(16, 512, beta=1.0, gamma=0.1)
    x_hi, _, _ = g.generate_batch(16, 512, beta=1.0, gamma=0.6)
    assert _filler_fraction(g, x_lo) < _filler_fraction(g, x_hi)
    # gamma is a PER-STEP filler prob (KV convention); each non-filler step emits
    # a ~3-4 token block, so per-token filler ~= gamma/(gamma+(1-gamma)*3.5) < gamma.
    assert _filler_fraction(g, x_hi) > 0.25   # high gamma => substantial filler


def test_beta_controls_recall_distance():
    # Smaller beta => recalls reach farther back (larger token lag) on average.
    g = NestedMonoidGenerator(p=31, n_names=512, n_filler=6)

    def mean_use_lag(beta):
        np.random.seed(5)
        _, _, _, meta = g.generate_batch(16, 512, beta=beta, gamma=0.0,
                                         return_meta=True)
        lags = [l for md in meta for l in md["use_lags"]]
        return float(np.mean(lags))

    far = mean_use_lag(0.05)   # holographic: long-range recalls
    near = mean_use_lag(3.0)   # local: recent recalls
    assert far > near


def test_depth_scales_with_seq_len():
    g = NestedMonoidGenerator(p=31, n_names=512, n_filler=6)
    np.random.seed(2)
    _, _, _, m_short = g.generate_batch(8, 128, beta=1.0, gamma=0.1, return_meta=True)
    _, _, _, m_long = g.generate_batch(8, 1024, beta=1.0, gamma=0.1, return_meta=True)
    assert np.mean([d["n_uses"] for d in m_long]) > 4 * np.mean([d["n_uses"] for d in m_short])


def test_truncated_mode_returns_window():
    g = NestedMonoidGenerator(p=31, n_names=512, n_filler=6,
                              mode="truncated", source_len=1024)
    x, y, m = g.generate_batch(4, 128, beta=1.0, gamma=0.1)
    assert x.shape == (4, 128)
    # The window is a slice of a length-1024 program, so it contains far fewer
    # DEFs than the full program (dangling recalls).
    n_defs_in_window = int((x == g.DEF).sum(dim=1).float().mean().item())
    assert n_defs_in_window < 200
