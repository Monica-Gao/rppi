#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rppi.config import load_config, save_config
from rppi.data import PPIPairDataset, ppi_collate
from rppi.losses import compute_losses
from rppi.metrics import binary_metrics
from rppi.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def forward_batch(model, batch: Dict[str, object]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    out = model(batch["x1"], batch["x2"], batch["mask1"], batch["mask2"])
    return out["logits"], out


def effective_loss_weights(cfg, epoch: int) -> Dict[str, float]:
    weights = dict(cfg["training"].get("loss_weights", {}))
    ramp_epochs = int(cfg["training"].get("symmetry_rampup_epochs", 0))
    if ramp_epochs <= 0 or float(weights.get("symmetry", 0.0)) <= 0:
        return weights

    ramp_start = int(cfg["training"].get("symmetry_rampup_start_epoch", 1))
    if epoch < ramp_start:
        factor = 0.0
    else:
        factor = min(1.0, float(epoch - ramp_start + 1) / float(ramp_epochs))
    weights["symmetry"] = float(weights["symmetry"]) * factor
    return weights


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch: int) -> Dict[str, float]:
    model.train()
    loss_weights = effective_loss_weights(cfg, epoch)
    print(
        f"epoch={epoch} effective_symmetry_weight={float(loss_weights.get('symmetry', 0.0)):.6f}",
        flush=True,
    )
    temperature = float(cfg["training"].get("contrastive_temperature", 0.1))
    rank_margin = float(cfg["training"].get("rank_margin", 0.5))
    symmetric_train = bool(cfg["training"].get("symmetric_train", False))
    shortcut_lambda = float(cfg["training"].get("shortcut_lambda", 1.0))
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    accum = int(cfg["training"].get("grad_accum_steps", 1))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    totals: Dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for step, raw_batch in enumerate(loader, start=1):
        batch = to_device(raw_batch, device)
        labels = batch["label"]
        with torch.cuda.amp.autocast(enabled=use_amp):
            main = model(
                batch["x1"],
                batch["x2"],
                batch["mask1"],
                batch["mask2"],
                shortcut_lambda=shortcut_lambda,
            )
            if symmetric_train or float(loss_weights.get("symmetry", 0.0)) > 0:
                swap = model(
                    batch["x2"],
                    batch["x1"],
                    batch["mask2"],
                    batch["mask1"],
                    shortcut_lambda=shortcut_lambda,
                )
                swap_logits = swap["logits"]
            else:
                swap_logits = main["logits"].detach()
            train_logits = 0.5 * (main["logits"] + swap_logits) if symmetric_train else main["logits"]

            if float(loss_weights.get("contrastive", 0.0)) > 0:
                view = model(batch["x1"], batch["x2"], batch["mask1"], batch["mask2"])
                view_embedding = view["embedding"]
            else:
                view_embedding = main["embedding"].detach()

            if float(loss_weights.get("ranking", 0.0)) > 0:
                cf = model(batch["cf_x1"], batch["cf_x2"], batch["cf_mask1"], batch["cf_mask2"])
                cf_logits = cf["logits"]
            else:
                cf_logits = main["logits"].detach()
            losses = compute_losses(
                logits=train_logits,
                labels=labels,
                swap_logits=swap_logits,
                view_a=main["embedding"],
                view_b=view_embedding,
                cf_logits=cf_logits,
                weights=loss_weights,
                temperature=temperature,
                rank_margin=rank_margin,
                teacher_probs=batch.get("teacher_prob"),
                label_smoothing=label_smoothing,
                same_species_logits=main.get("same_species_logits"),
                same_species_labels=batch.get("same_species"),
                degree_logits=main.get("degree_logits"),
                degree_labels=batch.get("degree_bin"),
            )
            loss = losses["total"] / accum

        scaler.scale(loss).backward()
        if step % accum == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

        log_every = int(cfg["training"].get("log_every", 100))
        if step % log_every == 0:
            avg = totals["total"] / step
            print(f"epoch={epoch} step={step}/{len(loader)} loss={avg:.4f} elapsed={(time.time()-start)/60:.1f}m", flush=True)

    if len(loader) % accum != 0:
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return {key: value / max(1, len(loader)) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, symmetric_eval: bool = False) -> Dict[str, float]:
    model.eval()
    labels, probs = [], []
    bce_total = 0.0
    n_examples = 0
    use_amp = device.type == "cuda"
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        batch_labels = batch["label"]
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(batch["x1"], batch["x2"], batch["mask1"], batch["mask2"])
            if symmetric_eval:
                swap = model(batch["x2"], batch["x1"], batch["mask2"], batch["mask1"])
                logits = 0.5 * (out["logits"] + swap["logits"])
            else:
                logits = out["logits"]
            bce = F.binary_cross_entropy_with_logits(logits, batch_labels, reduction="sum")
        bce_total += float(bce.detach().cpu())
        n_examples += int(batch_labels.numel())
        prob = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        probs.extend(prob)
        labels.extend(batch_labels.detach().cpu().numpy().tolist())
    metrics = binary_metrics(labels, probs)
    metrics["bce"] = bce_total / max(1, n_examples)
    return metrics


def make_loader(cfg, split: str, shuffle: bool) -> DataLoader:
    data_cfg = cfg["data"]
    split_cfg = data_cfg[split]
    ranking_weight = float(cfg["training"].get("loss_weights", {}).get("ranking", 0.0))
    dataset = PPIPairDataset(
        tsv_file=split_cfg["tsv"],
        embedding_dir=split_cfg.get("embedding_dir", data_cfg["embedding_dir"]),
        packed_embedding_dir=split_cfg.get("packed_embedding_dir", data_cfg.get("packed_embedding_dir")),
        sequence_length=int(data_cfg.get("sequence_length", 795)),
        embed_dim=int(data_cfg.get("embed_dim", 1024)),
        preload=bool(split_cfg.get("preload", data_cfg.get("preload", False))),
        max_samples=split_cfg.get("max_samples"),
        teacher_prob_paths=split_cfg.get("teacher_prob_paths"),
        cache_embeddings=split_cfg.get("cache_embeddings", data_cfg.get("cache_embeddings")),
        include_counterfactual=ranking_weight > 0,
    )
    batch_size = int(cfg["training"].get("batch_size", 16))
    if split != "train":
        batch_size = int(cfg["training"].get("eval_batch_size", batch_size))
    num_workers = int(cfg["training"].get("num_workers", 2))
    if split != "train":
        num_workers = int(cfg["training"].get("eval_num_workers", num_workers))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=ppi_collate,
        drop_last=False,
    )


def write_metrics(path: str, rows) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def infer_monitor_mode(monitor: str, configured: str | None) -> str:
    mode = (configured or "auto").lower()
    if mode == "auto":
        lowered = monitor.lower()
        mode = "min" if lowered in {"loss", "bce", "valid_loss", "valid_bce"} or lowered.endswith("_loss") else "max"
    if mode not in {"min", "max"}:
        raise ValueError(f"training.monitor_mode must be 'min', 'max', or 'auto', got {configured!r}")
    return mode


def monitor_improved(metric: float, best_metric: float | None, mode: str) -> bool:
    if math.isnan(metric):
        return False
    if best_metric is None or math.isnan(best_metric):
        return True
    return metric < best_metric if mode == "min" else metric > best_metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    out_dir = Path(cfg["output"]["dir"])
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(out_dir / "config_used.yaml"))

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    print(f"device={device}")
    train_loader = make_loader(cfg, "train", shuffle=True)
    valid_loader = make_loader(cfg, "valid", shuffle=False)
    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model_params={n_params}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 2e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.05)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)) and device.type == "cuda")

    best_metric = None
    best_path = ckpt_dir / f"{cfg['output']['name']}.best.pt"
    last_path = ckpt_dir / f"{cfg['output']['name']}.last.pt"
    history_path = out_dir / "training_history.tsv"
    history = []
    patience = int(cfg["training"].get("patience", 5))
    bad_epochs = 0
    monitor = cfg["training"].get("monitor", "aupr")
    monitor_mode = infer_monitor_mode(monitor, cfg["training"].get("monitor_mode"))
    symmetric_eval = bool(cfg["training"].get("symmetric_eval", cfg["training"].get("symmetric_train", False)))
    start_epoch = int(cfg["training"].get("start_epoch", 1))

    resume_checkpoint = cfg["training"].get("resume_checkpoint")
    auto_resume = bool(cfg["training"].get("auto_resume", False))
    if auto_resume and not resume_checkpoint and last_path.exists():
        resume_checkpoint = str(last_path)

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state)
        if isinstance(checkpoint, dict):
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scaler" in checkpoint and checkpoint["scaler"] is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(cfg["training"].get("start_epoch", int(checkpoint.get("epoch", 0)) + 1))
            checkpoint_monitor = checkpoint.get("monitor")
            checkpoint_monitor_mode = checkpoint.get("monitor_mode")
            same_monitor = (
                checkpoint_monitor == monitor
                and checkpoint_monitor_mode == monitor_mode
                and "best_metric" in checkpoint
                and checkpoint["best_metric"] is not None
            )
            if same_monitor:
                best_metric = float(checkpoint["best_metric"])
                bad_epochs = int(checkpoint.get("bad_epochs", bad_epochs))
            else:
                valid = checkpoint.get("valid", {})
                if monitor in valid:
                    best_metric = float(valid[monitor])
                bad_epochs = 0
            history = checkpoint.get("history", read_metrics(history_path))
        print(
            f"resumed_checkpoint={resume_checkpoint} start_epoch={start_epoch} "
            f"best_{monitor}={best_metric} monitor_mode={monitor_mode} bad_epochs={bad_epochs}",
            flush=True,
        )

    init_checkpoint = cfg["training"].get("init_checkpoint")
    if init_checkpoint and not resume_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state)
        if isinstance(checkpoint, dict):
            if "epoch" in checkpoint:
                start_epoch = int(cfg["training"].get("start_epoch", int(checkpoint["epoch"]) + 1))
            valid = checkpoint.get("valid", {})
            if monitor in valid:
                best_metric = float(valid[monitor])
                bad_epochs = 0
        print(
            f"loaded_init_checkpoint={init_checkpoint} start_epoch={start_epoch} "
            f"best_{monitor}={best_metric} monitor_mode={monitor_mode}",
            flush=True,
        )

    for epoch in range(start_epoch, int(cfg["training"].get("epochs", 20)) + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        valid_metrics = evaluate(model, valid_loader, device, symmetric_eval=symmetric_eval)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_losses.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(row)
        print(row, flush=True)
        metric = float(valid_metrics[monitor])
        if monitor_improved(metric, best_metric, monitor_mode):
            best_metric = metric
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "valid": valid_metrics}, best_path)
            print(f"saved_best={best_path} {monitor}={metric:.5f} monitor_mode={monitor_mode}", flush=True)
        else:
            bad_epochs += 1

        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "config": cfg,
            "epoch": epoch,
            "valid": valid_metrics,
            "best_metric": best_metric,
            "monitor": monitor,
            "monitor_mode": monitor_mode,
            "bad_epochs": bad_epochs,
            "history": history,
        }
        epoch_path = ckpt_dir / f"{cfg['output']['name']}.epoch{epoch}.pt"
        torch.save(checkpoint_payload, epoch_path)
        tmp_last = last_path.with_suffix(".tmp")
        torch.save(checkpoint_payload, tmp_last)
        os.replace(tmp_last, last_path)
        write_metrics(str(history_path), history)

        if bad_epochs >= patience:
            print(f"early_stop epoch={epoch} best_{monitor}={best_metric:.5f}", flush=True)
            break

    write_metrics(str(history_path), history)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    rows = []
    eval_loaders = [("valid", valid_loader)]
    if bool(cfg["training"].get("evaluate_test_at_end", True)):
        eval_loaders.append(("test", make_loader(cfg, "test", shuffle=False)))
    for split, loader in eval_loaders:
        metrics = evaluate(model, loader, device, symmetric_eval=symmetric_eval)
        rows.append({"model": cfg["output"]["name"], "split": split, **metrics})
    write_metrics(str(out_dir / "evaluation_results.tsv"), rows)
    print(f"wrote {out_dir / 'evaluation_results.tsv'}")


if __name__ == "__main__":
    main()
