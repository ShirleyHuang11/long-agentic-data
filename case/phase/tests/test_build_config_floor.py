import pytest
from omegaconf import OmegaConf
from phase_sweep import build_config


def _model_cfg(**over):
    base = dict(architecture="transformer_rope", vocab_size=40, d_model=128,
                nhead=4, ff_mult=4, num_layers=4, dropout=0.1, train_len=128,
                eval_lengths=[128, 512], train_steps=10, train_batch_size=8,
                eval_batch_size=4, eval_batches=2, lr=3e-4, grad_clip=1.0,
                renyi_seq_len=2000, renyi_qs=[1.0], renyi_orders=[1, 2])
    base.update(over)
    return OmegaConf.create(base)


def test_small_model_rejected_by_default_floor():
    with pytest.raises(ValueError):
        build_config(_model_cfg(), long_len=512, task="nested_monoid")


def test_small_model_passes_with_lowered_floor():
    cfg = build_config(_model_cfg(min_non_embed_params=100_000),
                       long_len=512, task="nested_monoid")
    assert cfg["task"] == "nested_monoid"
    assert cfg["architecture"] == "transformer_rope"
    assert cfg["_non_embed_params"] > 100_000
