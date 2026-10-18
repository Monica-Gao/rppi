from __future__ import annotations

import os
import json
import random
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


PairRecord = Tuple[str, str, int]


def read_pairs(tsv_file: str, max_samples: Optional[int] = None) -> List[PairRecord]:
    records: List[PairRecord] = []
    with open(tsv_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Expected at least 3 tab-separated columns in {tsv_file}: {line[:120]}")
            records.append((parts[0], parts[1], int(float(parts[2]))))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def _load_prob_vector(path: str) -> np.ndarray:
    if ":" in path and not path.endswith(".pt"):
        real_path, column = path.rsplit(":", 1)
        values = []
        with open(real_path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            if column not in header:
                raise ValueError(f"Column {column} not found in {real_path}")
            idx = header.index(column)
            for line in handle:
                if not line.strip():
                    continue
                values.append(float(line.rstrip("\n").split("\t")[idx]))
        return np.asarray(values, dtype=np.float32)
    if path.endswith(".pt"):
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("probs", "probabilities", "predictions", "prediction"):
                if key in payload:
                    payload = payload[key]
                    break
        if torch.is_tensor(payload):
            return payload.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return np.asarray(payload, dtype=np.float32).reshape(-1)
    values = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("proteinA\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            values.append(float(parts[-1]))
    return np.asarray(values, dtype=np.float32)


def load_teacher_probs(paths: Optional[Sequence[str]], max_samples: Optional[int] = None) -> Optional[np.ndarray]:
    if not paths:
        return None
    arrays = [_load_prob_vector(path) for path in paths]
    n = min(len(arr) for arr in arrays)
    if max_samples is not None:
        n = min(n, max_samples)
    stacked = np.stack([arr[:n] for arr in arrays], axis=0)
    return stacked.mean(axis=0).astype(np.float32)


def load_protein_embedding(embedding_dir: str, protein_id: str) -> torch.Tensor:
    npy_path = os.path.join(embedding_dir, f"{protein_id}.npy")
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    embd_path = os.path.join(embedding_dir, f"{protein_id}.embd")
    if not os.path.exists(embd_path):
        raise FileNotFoundError(f"Missing embedding for protein {protein_id}: {npy_path} / {embd_path}")

    rows: List[List[float]] = []
    with open(embd_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            _, vec = line.strip().split(":", 1)
            rows.append([float(x) for x in vec.strip().split()])
    return torch.tensor(rows, dtype=torch.float32)


class PackedEmbeddingStore:
    def __init__(self, packed_dir: str):
        metadata_path = os.path.join(packed_dir, "metadata.json")
        manifest_path = os.path.join(packed_dir, "manifest.json")
        self.index: Dict[str, Tuple[int, int]] = {}

        if os.path.exists(metadata_path):
            with open(metadata_path) as handle:
                self.metadata = json.load(handle)
            self.dtype = np.dtype(self.metadata["dtype"])
            self.dim = int(self.metadata["dim"])
            self.total_tokens = int(self.metadata["total_tokens"])
            data_path = os.path.join(packed_dir, self.metadata["data_file"])
            self.data = np.memmap(
                data_path,
                dtype=self.dtype,
                mode="r",
                shape=(self.total_tokens, self.dim),
            )
            index_path = os.path.join(packed_dir, "index.tsv")
            with open(index_path) as handle:
                header = handle.readline().rstrip("\n").split("\t")
                if header != ["protein_id", "offset", "length"]:
                    raise ValueError(f"Unexpected packed index header in {index_path}")
                for line in handle:
                    if not line.strip():
                        continue
                    pid, offset_text, length_text = line.rstrip("\n").split("\t")
                    self.index[pid] = (int(offset_text), int(length_text))
            self.format = "memmap_index_tsv_v1"
            return

        if os.path.exists(manifest_path):
            with open(manifest_path) as handle:
                self.metadata = json.load(handle)
            cache_format = self.metadata.get("format")
            if cache_format != "flat_npy_offsets_v1":
                raise ValueError(f"Unsupported packed manifest format in {manifest_path}: {cache_format!r}")
            self.dtype = np.dtype(self.metadata["dtype"])
            self.dim = int(self.metadata["protein_embedding_dim"])
            self.total_tokens = int(self.metadata["total_tokens"])
            data_path = os.path.join(packed_dir, "embeddings.npy")
            self.data = np.load(data_path, mmap_mode="r")
            if self.data.ndim != 2 or self.data.shape[1] != self.dim:
                raise ValueError(
                    f"Unexpected embedding cache shape in {data_path}: {self.data.shape}, expected [N, {self.dim}]"
                )
            offsets = np.load(os.path.join(packed_dir, "offsets.npy"), mmap_mode="r")
            with open(os.path.join(packed_dir, "protein_ids.json")) as handle:
                protein_ids = json.load(handle)
            if len(offsets) != len(protein_ids) + 1:
                raise ValueError(
                    f"Offsets/protein_ids mismatch in {packed_dir}: "
                    f"offsets={len(offsets)} proteins={len(protein_ids)}"
                )
            for idx, pid in enumerate(protein_ids):
                start = int(offsets[idx])
                end = int(offsets[idx + 1])
                self.index[pid] = (start, end - start)
            self.format = cache_format
            return

        raise FileNotFoundError(
            f"Packed embedding metadata not found in {packed_dir}: expected metadata.json or manifest.json"
        )

    def load(self, protein_id: str, dtype: np.dtype | str = np.float32) -> torch.Tensor:
        offset, length = self.index[protein_id]
        arr = self.data[offset : offset + length]
        return torch.from_numpy(np.array(arr, dtype=dtype, copy=True))


def pad_embedding(
    protein: torch.Tensor,
    sequence_length: int,
    embed_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    length = protein.shape[0]
    actual_dim = protein.shape[1] if protein.ndim > 1 else embed_dim
    if actual_dim != embed_dim:
        if actual_dim < embed_dim:
            pad = torch.zeros((length, embed_dim - actual_dim), dtype=protein.dtype)
            protein = torch.cat([protein, pad], dim=1)
        else:
            protein = protein[:, :embed_dim]

    valid_len = min(length, sequence_length)
    if length >= sequence_length:
        padded = protein[:sequence_length].contiguous()
    else:
        zeros = torch.zeros((sequence_length - length, embed_dim), dtype=protein.dtype)
        padded = torch.cat([protein, zeros], dim=0).contiguous()
    mask = torch.zeros(sequence_length, dtype=torch.bool)
    mask[:valid_len] = True
    return padded, mask


def infer_species(protein_id: str) -> str:
    if "." in protein_id:
        return protein_id.split(".", 1)[0]
    return "unknown"


def degree_bin(value: int) -> int:
    if value <= 1:
        return 0
    if value <= 3:
        return 1
    if value <= 7:
        return 2
    return 3


class PPIPairDataset(Dataset):
    def __init__(
        self,
        tsv_file: str,
        embedding_dir: str,
        packed_embedding_dir: Optional[str] = None,
        sequence_length: int = 795,
        embed_dim: int = 1024,
        preload: bool = False,
        max_samples: Optional[int] = None,
        teacher_prob_paths: Optional[Sequence[str]] = None,
        cache_embeddings: Optional[bool] = None,
        include_counterfactual: bool = True,
    ):
        self.tsv_file = tsv_file
        self.embedding_dir = embedding_dir
        self._packed_store = PackedEmbeddingStore(packed_embedding_dir) if packed_embedding_dir else None
        if self._packed_store is not None:
            print(f"Using packed embedding store: {packed_embedding_dir}", flush=True)
        if cache_embeddings is None:
            cache_embeddings = self._packed_store is None
        self.cache_embeddings = cache_embeddings
        if not self.cache_embeddings:
            print("Embedding cache disabled for this dataset", flush=True)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.records = read_pairs(tsv_file, max_samples=max_samples)
        self.include_counterfactual = include_counterfactual
        self.teacher_probs = load_teacher_probs(teacher_prob_paths, max_samples=max_samples)
        if self.teacher_probs is not None:
            if len(self.teacher_probs) != len(self.records):
                raise ValueError(
                    f"Teacher length mismatch for {tsv_file}: "
                    f"teacher={len(self.teacher_probs)} records={len(self.records)}"
                )
            print(f"Loaded teacher probabilities: {len(self.teacher_probs)} values", flush=True)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.max_cache_items = int(os.environ.get("RPPI_MAX_CACHE_ITEMS", "0"))
        self.cache_dtype = os.environ.get("RPPI_CACHE_DTYPE", "float32").lower()
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError(f"Unsupported RPPI_CACHE_DTYPE={self.cache_dtype}")
        if self.cache_embeddings:
            print(
                f"Embedding cache enabled: max_items={self.max_cache_items or 'unbounded'} "
                f"dtype={self.cache_dtype}",
                flush=True,
            )
        self._positives_by_anchor: Dict[str, List[str]] = defaultdict(list)
        self._negatives_by_anchor: Dict[str, List[str]] = defaultdict(list)
        self._all_negatives: List[Tuple[str, str]] = []
        self._species_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._protein_degree: Dict[str, int] = defaultdict(int)

        for idx, (p1, p2, label) in enumerate(self.records):
            self._species_to_indices[infer_species(p1)].append(idx)
            self._species_to_indices[infer_species(p2)].append(idx)
            self._protein_degree[p1] += 1
            self._protein_degree[p2] += 1
            if label == 1:
                self._positives_by_anchor[p1].append(p2)
                self._positives_by_anchor[p2].append(p1)
            else:
                self._negatives_by_anchor[p1].append(p2)
                self._negatives_by_anchor[p2].append(p1)
                self._all_negatives.append((p1, p2))

        if preload:
            unique = sorted({p for pair in self.records for p in pair[:2]})
            print(f"Preloading {len(unique)} protein embeddings from {embedding_dir}")
            for pid in unique:
                if self._packed_store is not None:
                    self._cache[pid] = self._packed_store.load(pid)
                else:
                    self._cache[pid] = load_protein_embedding(embedding_dir, pid)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def species(self) -> List[str]:
        return sorted(self._species_to_indices)

    def load(self, protein_id: str) -> torch.Tensor:
        if not self.cache_embeddings:
            if self._packed_store is not None:
                return self._packed_store.load(protein_id)
            return load_protein_embedding(self.embedding_dir, protein_id)
        if protein_id not in self._cache:
            if self._packed_store is not None:
                dtype = np.float16 if self.cache_dtype == "float16" else np.float32
                self._cache[protein_id] = self._packed_store.load(protein_id, dtype=dtype)
            else:
                self._cache[protein_id] = load_protein_embedding(self.embedding_dir, protein_id)
            if self.max_cache_items > 0:
                while len(self._cache) > self.max_cache_items:
                    self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(protein_id)
        return self._cache[protein_id]

    def sample_counterfactual(self, p1: str, p2: str, label: int) -> Tuple[str, str, int]:
        if label == 1:
            candidates = self._negatives_by_anchor.get(p1)
            if candidates:
                return p1, random.choice(candidates), 0
            candidates = self._negatives_by_anchor.get(p2)
            if candidates:
                return p2, random.choice(candidates), 0
        if self._all_negatives:
            n1, n2 = random.choice(self._all_negatives)
            return n1, n2, 0
        return p1, p2, 1 - label

    def __getitem__(self, idx: int) -> Dict[str, object]:
        p1, p2, label = self.records[idx]
        emb1, mask1 = pad_embedding(self.load(p1), self.sequence_length, self.embed_dim)
        emb2, mask2 = pad_embedding(self.load(p2), self.sequence_length, self.embed_dim)
        item = {
            "p1": p1,
            "p2": p2,
            "x1": emb1,
            "x2": emb2,
            "mask1": mask1,
            "mask2": mask2,
            "label": torch.tensor(label, dtype=torch.float32),
            "same_species": torch.tensor(
                1.0 if infer_species(p1) == infer_species(p2) else 0.0,
                dtype=torch.float32,
            ),
            "degree_bin": torch.tensor(
                degree_bin(max(self._protein_degree[p1], self._protein_degree[p2])),
                dtype=torch.long,
            ),
            "teacher_prob": torch.tensor(
                float(self.teacher_probs[idx]) if self.teacher_probs is not None else -1.0,
                dtype=torch.float32,
            ),
        }
        if self.include_counterfactual:
            cf1, cf2, cf_label = self.sample_counterfactual(p1, p2, label)
            cf_emb1, cf_mask1 = pad_embedding(self.load(cf1), self.sequence_length, self.embed_dim)
            cf_emb2, cf_mask2 = pad_embedding(self.load(cf2), self.sequence_length, self.embed_dim)
            item.update(
                {
                    "cf_x1": cf_emb1,
                    "cf_x2": cf_emb2,
                    "cf_mask1": cf_mask1,
                    "cf_mask2": cf_mask2,
                    "cf_label": torch.tensor(cf_label, dtype=torch.float32),
                }
            )
        return item


def ppi_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    tensor_keys = [
        "x1",
        "x2",
        "mask1",
        "mask2",
        "label",
        "same_species",
        "degree_bin",
        "teacher_prob",
    ]
    if "cf_x1" in batch[0]:
        tensor_keys.extend(["cf_x1", "cf_x2", "cf_mask1", "cf_mask2", "cf_label"])
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch])  # type: ignore[index]
    out["p1"] = [item["p1"] for item in batch]
    out["p2"] = [item["p2"] for item in batch]
    return out
