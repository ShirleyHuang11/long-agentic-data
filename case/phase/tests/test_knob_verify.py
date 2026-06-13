import numpy as np
from knob_verify import estimate_beta, estimate_gamma


def test_iid_has_fast_decay_low_correlation():
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 8, size=40000)
    beta = estimate_beta(toks, vocab_size=8, lags=[1, 2, 4, 8, 16, 32])
    # iid => correlation amplitude tiny at all lags => steep (large) beta or ~0 corr
    assert beta["corr_at_lag1"] < 0.05


def test_long_range_block_repeat_has_slower_decay_than_iid():
    rng = np.random.default_rng(1)
    iid = rng.integers(0, 8, size=40000)
    block = rng.integers(0, 8, size=200)
    rep = np.tile(block, 200)                 # strong long-range periodic structure
    b_iid = estimate_beta(iid, vocab_size=8, lags=[1, 2, 4, 8, 16, 32, 64])
    b_rep = estimate_beta(rep, vocab_size=8, lags=[1, 2, 4, 8, 16, 32, 64])
    assert b_rep["corr_at_lag32"] > b_iid["corr_at_lag32"]


def test_gamma_estimator_monotone_entropy():
    rng = np.random.default_rng(2)
    toks = rng.integers(0, 8, size=40000)
    g = estimate_gamma(toks, vocab_size=8, max_order=4)
    # conditional entropies are non-increasing in context order (within noise)
    hs = g["cond_entropy_by_order"]
    assert hs[0] >= hs[-1] - 0.05
