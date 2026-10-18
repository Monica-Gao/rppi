#!/usr/bin/env python3
"""Generate residue-level ProtT5 inputs for RPPI from a protein FASTA."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

PROTT5_REPOSITORY = "Rostlab/prot_t5_xl_uniref50"
PROTT5_REVISION = "973be27c52ee6474de9c945952a8008aeb2a1a73"
WEIGHT_SHA256 = "c06141d42e93c12b5f6d95c701952156bd4079661f5f2d981d0d2036ba96dae9"
TOKENIZER_SHA256 = "74da7b4afcde53faa570114b530c726135bdfcdb813dec3abfb27f9d44db7324"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    current_id = None
    sequence = []
    seen = set()
    with path.open() as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(sequence)))
                header = line[1:].split()
                current_id = header[0] if header else ""
                if (not current_id or current_id in {".", ".."}
                        or Path(current_id).name != current_id
                        or chr(92) in current_id or current_id in seen):
                    raise ValueError(f"Invalid or duplicate FASTA identifier at line {line_number}: {current_id!r}")
                seen.add(current_id)
                sequence = []
            else:
                if current_id is None:
                    raise ValueError("Sequence appeared before a FASTA header")
                sequence.append(line.upper())
    if current_id is not None:
        records.append((current_id, "".join(sequence)))
    if not records:
        raise ValueError("FASTA contains no records")
    for pid, seq in records:
        if not seq or re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYXBZOU]+", seq) is None:
            raise ValueError(f"{pid}: expected a nonempty, unaligned protein sequence")
    return records


def preprocess(sequence: str, max_residues: int = 900) -> str:
    return re.sub(r"[UZOB]", "X", sequence.upper()[:max_residues])


def load_encoder(model_dir: Path, device: torch.device):
    """Map the original PyTorch file and materialize only its encoder."""
    from transformers import T5Config, T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        str(model_dir), do_lower_case=False, legacy=True, clean_up_tokenization_spaces=False, local_files_only=True
    )
    config = T5Config.from_pretrained(str(model_dir), local_files_only=True)
    with torch.device("meta"):
        model = T5EncoderModel(config)
    state = torch.load(
        model_dir / "pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True
    )
    encoder_state = {k: v for k, v in state.items() if k == "shared.weight" or k.startswith("encoder.")}
    model.load_state_dict(encoder_state, strict=True, assign=True)
    del state, encoder_state
    return tokenizer, model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-residues", type=int, default=900)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    if args.max_residues < 1 or args.threads < 1 or args.log_every < 0:
        parser.error("max-residues and threads must be positive; log-every must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    torch.set_num_threads(args.threads)
    records = read_fasta(args.fasta)[args.shard_index::args.num_shards]
    if not records:
        parser.error("This shard contains no records")
    manifest_path = args.output_dir / f"embedding_manifest_shard{args.shard_index}.json"
    targets = [args.output_dir / f"{pid}.npy" for pid, _ in records]
    if manifest_path.exists() or any(p.exists() for p in targets):
        parser.error("Output already contains this shard; use a fresh output directory")
    print("Verifying the original ProtT5 source weights and tokenizer...", flush=True)
    for name, expected in (("pytorch_model.bin", WEIGHT_SHA256), ("spiece.model", TOKENIZER_SHA256)):
        if sha256(args.model_dir / name) != expected:
            raise ValueError(f"{name} does not match the pinned ProtT5 source")
    device = torch.device(args.device)
    tokenizer, model = load_encoder(args.model_dir, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proteins = []
    out_dtype = np.float16 if args.dtype == "float16" else np.float32
    for index, (pid, raw_sequence) in enumerate(records, 1):
        sequence = preprocess(raw_sequence, args.max_residues)
        encoded = tokenizer(" ".join(sequence), add_special_tokens=True, return_tensors="pt")
        if int(encoded["attention_mask"].sum()) != len(sequence) + 1:
            raise ValueError(f"{pid}: tokenizer did not produce one token per residue plus EOS")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        precision = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with torch.inference_mode(), precision:
            hidden = model(**encoded).last_hidden_state
        residue = hidden[0, :len(sequence)].float().cpu().numpy().astype(out_dtype)
        if residue.shape != (len(sequence), 1024) or not np.isfinite(residue).all():
            raise ValueError(f"{pid}: invalid residue embeddings {residue.shape}")
        target = args.output_dir / f"{pid}.npy"
        temporary = target.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, residue, allow_pickle=False)
        temporary.replace(target)
        proteins.append({"protein": pid, "residues": len(sequence), "sha256": sha256(target)})
        if args.log_every and (index % args.log_every == 0 or index == len(records)):
            print(f"written={index}/{len(records)} elapsed={(time.monotonic() - started) / 60:.1f}m", flush=True)
    metadata = {
        "model_id": PROTT5_REPOSITORY, "model_revision": PROTT5_REVISION,
        "model_sha256": WEIGHT_SHA256, "tokenizer_sha256": TOKENIZER_SHA256,
        "fasta_sha256": sha256(args.fasta), "max_residues": args.max_residues,
        "storage_dtype": args.dtype, "compute_precision": "cuda-fp16-autocast" if device.type == "cuda" else "cpu-float32",
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "proteins": proteins,
    }
    manifest_path.write_text(json.dumps(metadata, indent=2) + chr(10))
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
