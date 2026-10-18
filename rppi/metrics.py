from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(labels, probs, threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
        "aupr": float(average_precision_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
    }

