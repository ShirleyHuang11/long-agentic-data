from data_generator import _FACTORIES
from omegaconf import OmegaConf


def test_holo_anchors_pairs():
    pl = _FACTORIES["holo_anchors"](OmegaConf.create({}))
    pairs = {(round(b, 3), round(g, 3)) for b, g in pl.pairs}
    assert pairs == {(2.0, 0.8), (0.5, 0.4), (0.05, 0.05), (0.0, 0.0)}


def test_holo_grid_shape():
    pl = _FACTORIES["holo_grid"](OmegaConf.create({}))
    betas = sorted({round(b, 3) for b, _ in pl.pairs})
    gammas = sorted({round(g, 3) for _, g in pl.pairs})
    assert betas == [0.0, 0.05, 0.2, 0.5, 1.0, 2.0]
    assert gammas == [0.0, 0.05, 0.2, 0.4, 0.6, 0.8]
    assert len(pl.pairs) == 36
