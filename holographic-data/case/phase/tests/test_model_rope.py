import torch
from model_rope import RoPECausalTransformer
from model import count_non_embedding_params


def _small():
    return RoPECausalTransformer(vocab_size=40, d_model=32, nhead=4,
                                 ff_mult=4, num_layers=2, dropout=0.0)


def test_forward_shape():
    m = _small().eval()
    x = torch.randint(0, 40, (3, 17))
    y = m(x)
    assert y.shape == (3, 17, 40)


def test_causal_future_does_not_leak():
    # Changing the LAST token must not change outputs at earlier positions.
    torch.manual_seed(0)
    m = _small().eval()
    x = torch.randint(0, 40, (2, 12))
    with torch.no_grad():
        a = m(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 40
        b = m(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_length_extrapolation_runs():
    # RoPE has no learned position table → forward at a length never "configured".
    m = _small().eval()
    for T in (8, 64, 256):
        y = m(torch.randint(0, 40, (1, T)))
        assert y.shape == (1, T, 40)


def test_param_count_matches_helper():
    d, V = 32, 40
    m = _small()
    non_embed = sum(p.numel() for n, p in m.named_parameters()
                    if not n.startswith("embedding."))
    expected = count_non_embedding_params(d_model=d, nhead=4, ff_mult=4,
                                          num_layers=2, vocab_size=V) + 2 * d
    assert non_embed == expected  # +2d for the final LayerNorm
