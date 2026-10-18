import csv
import json
from pathlib import Path

import torch

from rppi.model import build_model
from rppi.data import load_protein_embedding, pad_embedding


def _load_checkpoint(root, config):
    name = config.get("original_checkpoint")
    candidates = [root / name] if name else []
    candidates += sorted(root.glob("*.pt"))
    candidates += sorted(root.glob("*.snapshot"))
    checkpoint_path = next((p for p in candidates if p.is_file()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No PyTorch checkpoint found in {root}; expected {name or '*.pt'}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.name}")
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a model state_dict: {checkpoint_path.name}")
    return checkpoint, state


def load_model(model_dir, device="cpu"):
    root = Path(model_dir)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    checkpoint, state = _load_checkpoint(root, config)
    checkpoint_config = checkpoint.get("config", {})
    model_config = config.get("model") or checkpoint_config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Model configuration is missing from config.json and checkpoint")
    model = build_model(model_config)
    model.load_state_dict(state, strict=True)
    if not config:
        config = dict(checkpoint_config)
    return model.to(device).eval(), config


@torch.inference_mode()
def symmetric_logits(model, x1, x2, mask1, mask2):
    if model.training:
        raise ValueError("Inference requires model.eval(); dropout must be disabled")
    ab = model(x1, x2, mask1, mask2)["logits"]
    ba = model(x2, x1, mask2, mask1)["logits"]
    return (ab + ba) * 0.5, ab, ba


def read_pair_rows(path):
    with Path(path).open() as f:
        for i, row in enumerate(csv.reader(f, delimiter="\t")):
            if not row:
                continue
            if i == 0 and row[0].lower() in {"proteina", "protein1"}:
                continue
            if len(row) not in (2, 3):
                raise ValueError(f"Line {i + 1}: expected two IDs and optional binary label")
            for pid in row[:2]:
                if not pid or Path(pid).name != pid or pid in {".", ".."} or chr(92) in pid:
                    raise ValueError(f"Invalid identifier on line {i + 1}")
            label = int(row[2]) if len(row) == 3 else None
            if label not in (None, 0, 1):
                raise ValueError("Labels must be 0 or 1")
            yield row[0], row[1], label


@torch.inference_mode()
def predict_pairs(model, config, pairs, embeddings, output, batch_size=16, threshold=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    rows = list(read_pair_rows(pairs))
    if not rows:
        raise ValueError("Pair file is empty")
    labeled = {r[2] is not None for r in rows}
    if len(labeled) != 1:
        raise ValueError("Do not mix labeled and unlabeled rows")
    n = config.get("sequence_length", 795)
    dim = config["model"].get("input_dim", 1024)
    device = next(model.parameters()).device

    def embedding(pid):
        x = load_protein_embedding(str(embeddings), pid)
        if x.ndim != 2 or x.shape[1] != dim or x.shape[0] == 0 or not torch.isfinite(x).all():
            raise ValueError(f"{pid}: expected finite nonempty embeddings [L,{dim}], got {tuple(x.shape)}")
        return pad_embedding(x, n, dim)

    for pid in sorted({p for row in rows for p in row[:2]}):
        embedding(pid)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["proteinA", "proteinB", "logit_ab", "logit_ba", "prediction"]
    if True in labeled:
        fields.append("label")
    if threshold is not None:
        fields.append("predicted_label")
    with target.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            a = [embedding(r[0]) for r in batch]
            b = [embedding(r[1]) for r in batch]
            x1 = torch.stack([x for x, m in a]).to(device)
            x2 = torch.stack([x for x, m in b]).to(device)
            m1 = torch.stack([m for x, m in a]).to(device)
            m2 = torch.stack([m for x, m in b]).to(device)
            logits, ab, ba = symmetric_logits(model, x1, x2, m1, m2)
            for row, l1, l2, p in zip(batch, ab.cpu().tolist(), ba.cpu().tolist(), logits.sigmoid().cpu().tolist()):
                item = dict(proteinA=row[0], proteinB=row[1], logit_ab=l1, logit_ba=l2, prediction=p)
                if row[2] is not None:
                    item["label"] = row[2]
                if threshold is not None:
                    item["predicted_label"] = int(p >= threshold)
                w.writerow(item)
    return {"pairs": len(rows), "output": str(target), "aggregation": "sigmoid(mean(logits_ab,logits_ba))"}
