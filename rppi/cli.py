import argparse
import json

import torch

from .inference import load_model, predict_pairs


def main():
    parser = argparse.ArgumentParser(description="RPPI symmetric inference from residue embeddings")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict")
    for key in ("model-dir", "pairs", "embeddings", "output"):
        predict.add_argument("--" + key, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--threshold", type=float, help="Explicit decision threshold; otherwise scores only")
    predict.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    model, config = load_model(args.model_dir, args.device)
    result = predict_pairs(
        model, config, args.pairs, args.embeddings, args.output, args.batch_size, args.threshold
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
