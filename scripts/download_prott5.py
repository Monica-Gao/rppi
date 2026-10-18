#!/usr/bin/env python3
"""Download the pinned upstream ProtT5 files used by the embedding script."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from generate_prott5_embeddings import PROTT5_REPOSITORY, PROTT5_REVISION, sha256

FILES = {
    "config.json": "899fe63bf0178fd23583eca238ade81d56a1779c344741f391c3902855c64b77",
    "pytorch_model.bin": "c06141d42e93c12b5f6d95c701952156bd4079661f5f2d981d0d2036ba96dae9",
    "spiece.model": "74da7b4afcde53faa570114b530c726135bdfcdb813dec3abfb27f9d44db7324",
    "tokenizer_config.json": "a5064475340d80a560463f49921407d901eef3e89667a931eb81a186633eff4b",
    "special_tokens_map.json": "4720c0fddbe4c5991334f85ad7073d9bd0a294a8ba4641a2f8dab614ca825949",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot_download(
        repo_id=PROTT5_REPOSITORY, revision=PROTT5_REVISION,
        allow_patterns=list(FILES), local_dir=args.output_dir,
    )
    for filename, expected in FILES.items():
        if sha256(args.output_dir / filename) != expected:
            raise ValueError(f"Checksum mismatch: {filename}")
    print(f"Verified ProtT5 source: {args.output_dir}")


if __name__ == "__main__":
    main()
