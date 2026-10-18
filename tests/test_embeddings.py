"""Input safeguards that do not download the external language model."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/generate_prott5_embeddings.py"
SPEC = importlib.util.spec_from_file_location("embedding_script", SOURCE)
embedding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(embedding)


def test_fasta_and_training_preprocessing(tmp_path):
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">A description\nacu\nzob\n>B\nWXX\n")
    assert embedding.read_fasta(fasta) == [("A", "ACUZOB"), ("B", "WXX")]
    assert embedding.preprocess("acuZoBw", 6) == "ACXXXX"


@pytest.mark.parametrize("content", [
    ">\nAAA\n", ">..\nAAA\n", ">../x\nAAA\n", ">A\nAAA\n>A\nBBB\n",
    ">A\n", ">A\nACD-*\n", "ACD\n",
])
def test_bad_fasta_rejected_before_model_loading(tmp_path, content):
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(content)
    with pytest.raises(ValueError):
        embedding.read_fasta(fasta)
