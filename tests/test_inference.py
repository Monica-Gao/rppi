import csv
import json

import numpy as np
import pytest
import torch

from rppi.model import build_model
from rppi.inference import load_model, symmetric_logits, predict_pairs


@pytest.fixture
def bundle(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = dict(input_dim=8, hidden_dim=12, num_heads=3, num_layers=1, cross_layers=1, ff_dim=24, dropout=0.15, chunk_size=4, sequence_length=16, projection_dim=8)
    model = build_model(cfg).eval()
    torch.save({"model": model.state_dict(), "config": {"model": cfg}, "epoch": 1}, tmp_path / "fixture.pt")
    (tmp_path / "config.json").write_text(json.dumps({"model": cfg, "sequence_length": 16, "original_checkpoint": "fixture.pt"}))
    return tmp_path, model


def test_weight_roundtrip_and_order_invariance(bundle):
    root, original = bundle
    model, cfg = load_model(root)
    for k, v in original.state_dict().items():
        assert torch.equal(v, model.state_dict()[k])
    a, b = torch.randn(3, 16, 8), torch.randn(3, 16, 8)
    mask = torch.ones(3, 16, dtype=torch.bool)
    score, ab, ba = symmetric_logits(model, a, b, mask, mask)
    reverse, _, _ = symmetric_logits(model, b, a, mask, mask)
    torch.testing.assert_close(score, reverse, rtol=0, atol=0)
    torch.testing.assert_close(score.sigmoid(), ((ab + ba) / 2).sigmoid())


def test_unlabeled_pairs_and_wrong_dimension(bundle):
    root, _ = bundle
    model, cfg = load_model(root)
    (root / "pairs.tsv").write_text("proteinA\tproteinB\nA\tB\nB\tA\n")
    np.save(root / "A.npy", np.ones((7, 8), dtype=np.float32))
    np.save(root / "B.npy", np.ones((9, 8), dtype=np.float32))
    predict_pairs(model, cfg, root / "pairs.tsv", root, root / "pred.tsv")
    rows = list(csv.DictReader((root / "pred.tsv").open(), delimiter="\t"))
    assert rows[0]["prediction"] == rows[1]["prediction"] and "label" not in rows[0]
    np.save(root / "B.npy", np.ones((9, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="expected finite"):
        predict_pairs(model, cfg, root / "pairs.tsv", root, root / "bad.tsv")
    assert not (root / "bad.tsv").exists()


def test_training_mode_rejected(bundle):
    _, model = bundle
    model.train()
    x = torch.ones(1, 16, 8)
    m = torch.ones(1, 16, dtype=torch.bool)
    with pytest.raises(ValueError, match="model.eval"):
        symmetric_logits(model, x, x, m, m)
