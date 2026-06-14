import torch
from phase_core import masked_next_token_acc, make_generator


def test_masked_accuracy_counts_only_masked_positions():
    # vocab=3, T=4. logits argmax = [2,2,2,2] for both rows.
    logits = torch.zeros(1, 4, 3)
    logits[..., 2] = 1.0
    targets = torch.tensor([[2, 0, 2, 0]])    # correct at positions 0 and 2
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    acc = masked_next_token_acc(logits, targets, mask)
    assert abs(acc - 1.0) < 1e-6              # both masked positions correct
    mask2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    acc2 = masked_next_token_acc(logits, targets, mask2)
    assert abs(acc2 - 0.5) < 1e-6             # 2 of 4 correct


def test_make_generator_nested_monoid():
    g = make_generator({"task": "nested_monoid", "monoid_p": 31,
                        "monoid_names": 512, "monoid_filler": 6})
    x, y, m = g.generate_batch(2, 64, beta=1.0, gamma=0.2)
    assert x.shape == (2, 64)
    assert g.vocab_size == 31 + 512 + 3 + 6  # 552
