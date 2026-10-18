"""Relocate dataset and embedding paths without changing scientific settings."""
import argparse
from pathlib import Path
import yaml

p = argparse.ArgumentParser()
for key in ["template", "data-dir", "embeddings", "run-dir", "output"]:
    p.add_argument("--" + key, required=True)
p.add_argument("--packed-embeddings")
p.add_argument("--test-embeddings")
a = p.parse_args()
c = yaml.safe_load(Path(a.template).read_text())
for split in ["train", "valid", "test"]:
    filename = Path(c["data"][split]["tsv"]).name
    base = Path(a.data_dir).resolve()
    candidates = [base / filename, base / "pairs" / filename]
    target = next((q for q in candidates if q.is_file()), None)
    if target is None:
        p.error(f"Missing {split} pair file {filename} under {base}")
    c["data"][split]["tsv"] = str(target)
    c["data"][split].pop("packed_embedding_dir", None)
    if "embedding_dir" in c["data"][split]:
        c["data"][split]["embedding_dir"] = str(Path(a.test_embeddings if split == "test" and a.test_embeddings else a.embeddings).resolve())
c["data"]["embedding_dir"] = str(Path(a.embeddings).resolve())
c["data"].pop("packed_embedding_dir", None)
if a.packed_embeddings:
    c["data"]["packed_embedding_dir"] = str(Path(a.packed_embeddings).resolve())
if a.test_embeddings:
    c["data"]["test"]["embedding_dir"] = str(Path(a.test_embeddings).resolve())
c["output"]["dir"] = str(Path(a.run_dir).resolve())
c["output"]["checkpoint_dir"] = str(Path(a.run_dir).resolve() / "checkpoints")
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(yaml.safe_dump(c, sort_keys=False))
