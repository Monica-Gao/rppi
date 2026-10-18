#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rppi.config import load_config
from rppi.metrics import binary_metrics
from rppi.model import build_model
from scripts.train import make_loader, to_device


def parse_thresholds(spec: str) -> List[float]:
    if ":" in spec:
        start_s, stop_s, step_s = spec.split(":")
        start = float(start_s)
        stop = float(stop_s)
        step = float(step_s)
        if step <= 0:
            raise ValueError("--thresholds step must be positive")
        values = []
        current = start
        while current <= stop + (step / 2):
            values.append(round(current, 10))
            current += step
        return values
    return [float(item) for item in spec.split(",") if item.strip()]


@torch.no_grad()
def collect_predictions(model, loader, device, symmetric_eval: bool, log_every: int) -> Tuple[List[float], List[float], float]:
    model.eval()
    labels, probs = [], []
    bce_total = 0.0
    n_examples = 0
    use_amp = device.type == "cuda"
    start = time.time()
    for step, raw_batch in enumerate(loader, start=1):
        batch = to_device(raw_batch, device)
        batch_labels = batch["label"]
        with torch.cuda.amp.autocast(enabled=use_amp):
            if symmetric_eval:
                x1 = torch.cat([batch["x1"], batch["x2"]], dim=0)
                x2 = torch.cat([batch["x2"], batch["x1"]], dim=0)
                mask1 = torch.cat([batch["mask1"], batch["mask2"]], dim=0)
                mask2 = torch.cat([batch["mask2"], batch["mask1"]], dim=0)
                out = model(x1, x2, mask1, mask2)
                logits_ab, logits_ba = out["logits"].chunk(2, dim=0)
                logits = 0.5 * (logits_ab + logits_ba)
            else:
                out = model(batch["x1"], batch["x2"], batch["mask1"], batch["mask2"])
                logits = out["logits"]
            bce = F.binary_cross_entropy_with_logits(logits, batch_labels, reduction="sum")
        bce_total += float(bce.detach().cpu())
        n_examples += int(batch_labels.numel())
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels.extend(batch_labels.detach().cpu().numpy().tolist())
        if log_every > 0 and (step % log_every == 0 or step == len(loader)):
            elapsed_min = (time.time() - start) / 60
            print(
                f"eval step={step}/{len(loader)} examples={len(labels)} "
                f"elapsed={elapsed_min:.1f} min",
                flush=True,
            )
    return labels, probs, bce_total / max(1, n_examples)


def threshold_rows(name: str, split: str, labels: Iterable[float], probs: Iterable[float], thresholds: Iterable[float], bce: float) -> List[Dict[str, float]]:
    rows = []
    for threshold in thresholds:
        metrics = binary_metrics(labels, probs, threshold=threshold)
        rows.append({"model": name, "split": split, "threshold": threshold, **metrics, "bce": bce})
    return rows


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def best_threshold(rows: List[Dict[str, float]], metric: str) -> float:
    if not rows:
        raise ValueError("No threshold rows available")
    if metric not in rows[0]:
        raise ValueError(f"Metric {metric!r} not found in threshold rows")
    best = max(rows, key=lambda row: float(row[metric]))
    return float(best["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="checkpoint_eval")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--thresholds", default="0.40:0.55:0.01")
    parser.add_argument(
        "--select-threshold-on-valid",
        action="store_true",
        help="Evaluate valid and test, select the threshold on valid, then apply it to test.",
    )
    parser.add_argument("--selection-metric", default="f1", choices=("f1", "accuracy", "precision", "recall"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    if args.max_samples > 0:
        cfg["data"][args.split]["max_samples"] = args.max_samples
        if args.select_threshold_on_valid:
            cfg["data"]["valid"]["max_samples"] = args.max_samples
    cfg["training"]["eval_num_workers"] = int(cfg["training"].get("eval_num_workers", 0))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    print(f"device={device}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"split={args.split}", flush=True)
    print(f"max_samples={args.max_samples if args.max_samples > 0 else 'all'}", flush=True)

    if args.select_threshold_on_valid:
        split_order = ["valid", "test"]
    elif args.split != "test":
        split_order = [args.split]
    else:
        split_order = ["test"]

    model = build_model(cfg["model"]).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    symmetric_eval = bool(cfg["training"].get("symmetric_eval", cfg["training"].get("symmetric_train", False)))
    out_dir = Path(args.output_dir)

    predictions = {}
    threshold_values = parse_thresholds(args.thresholds)
    all_threshold_rows = []
    for split in split_order:
        print(f"collecting split={split}", flush=True)
        loader = make_loader(cfg, split, shuffle=False)
        predictions[split] = collect_predictions(
            model,
            loader,
            device,
            symmetric_eval=symmetric_eval,
            log_every=args.log_every,
        )
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        labels, probs, bce = predictions[split]
        all_threshold_rows.extend(threshold_rows(args.name, split, labels, probs, threshold_values, bce))

    eval_rows = []
    if args.select_threshold_on_valid:
        valid_rows = [row for row in all_threshold_rows if row["split"] == "valid"]
        selected_threshold = best_threshold(valid_rows, args.selection_metric)
        print(
            f"selected_threshold={selected_threshold:.10g} "
            f"selection_split=valid selection_metric={args.selection_metric}",
            flush=True,
        )
        for split in split_order:
            labels, probs, bce = predictions[split]
            metrics = binary_metrics(labels, probs, threshold=selected_threshold)
            eval_rows.append(
                {
                    "model": args.name,
                    "split": split,
                    "threshold": selected_threshold,
                    "threshold_source": "valid",
                    "selection_metric": args.selection_metric,
                    **metrics,
                    "bce": bce,
                }
            )
    else:
        labels, probs, bce = predictions[args.split]
        eval_rows.append({"model": args.name, "split": args.split, **binary_metrics(labels, probs), "bce": bce})

    write_rows(out_dir / "evaluation_results.tsv", eval_rows)
    write_rows(out_dir / "evaluation_results_thresholds.tsv", all_threshold_rows)
    for row in eval_rows:
        print(row, flush=True)
    print(f"wrote {out_dir / 'evaluation_results.tsv'}", flush=True)
    print(f"wrote {out_dir / 'evaluation_results_thresholds.tsv'}", flush=True)


if __name__ == "__main__":
    main()
