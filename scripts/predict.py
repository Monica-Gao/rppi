#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rppi.config import load_config
from rppi.data import PPIPairDataset, ppi_collate
from rppi.metrics import binary_metrics
from rppi.model import build_model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    split_cfg = data_cfg[args.split]
    dataset = PPIPairDataset(
        tsv_file=split_cfg["tsv"],
        embedding_dir=split_cfg.get("embedding_dir", data_cfg["embedding_dir"]),
        packed_embedding_dir=split_cfg.get("packed_embedding_dir", data_cfg.get("packed_embedding_dir")),
        sequence_length=int(data_cfg.get("sequence_length", 795)),
        embed_dim=int(data_cfg.get("embed_dim", 1024)),
        preload=bool(split_cfg.get("preload", data_cfg.get("preload", False))),
        max_samples=split_cfg.get("max_samples"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("eval_batch_size", cfg["training"].get("batch_size", 16))),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 2)),
        pin_memory=True,
        collate_fn=ppi_collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    labels, probs, rows = [], [], []
    for batch in loader:
        x1 = batch["x1"].to(device)
        x2 = batch["x2"].to(device)
        mask1 = batch["mask1"].to(device)
        mask2 = batch["mask2"].to(device)
        logits = model(x1, x2, mask1, mask2)["logits"]
        if bool(cfg["training"].get("symmetric_eval", False)):
            swap_logits = model(x2, x1, mask2, mask1)["logits"]
            logits = 0.5 * (logits + swap_logits)
        prob = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        lab = batch["label"].numpy().tolist()
        labels.extend(lab)
        probs.extend(prob)
        for p1, p2, y, p in zip(batch["p1"], batch["p2"], lab, prob):
            rows.append({"proteinA": p1, "proteinB": p2, "label": int(y), "prediction": p})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["proteinA", "proteinB", "label", "prediction"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(binary_metrics(labels, probs))


if __name__ == "__main__":
    main()
