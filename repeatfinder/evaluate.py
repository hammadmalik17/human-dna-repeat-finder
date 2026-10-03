"""Base-level evaluation of repeat detectors against the soft-mask (and simulation truth)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def confusion(pred: np.ndarray, ref: np.ndarray) -> dict:
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum()); tn = int((~pred & ~ref).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "iou": tp / max(tp + fp + fn, 1),
            "called_frac": (tp + fp) / (tp + fp + fn + tn), "TP": tp, "FP": fp, "FN": fn}


def auc(score: np.ndarray, ref: np.ndarray, max_n=2_000_000, seed=0) -> float:
    """Base-level ROC-AUC of a continuous score vs the soft-mask (rank-based, subsampled)."""
    s = score.ravel().astype(np.float64); y = ref.ravel()
    if len(s) > max_n:
        idx = np.random.default_rng(seed).choice(len(s), max_n, replace=False)
        s, y = s[idx], y[idx]
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def benchmark(masks: dict, ref: np.ndarray) -> pd.DataFrame:
    rows = {name: confusion(m, ref) for name, m in masks.items()}
    return pd.DataFrame(rows).T


def decomposition(masks: dict, ref: np.ndarray) -> pd.DataFrame:
    """What explains the soft-masked bases?  Which detector calls bases the mask does NOT have?"""
    names = list(masks)
    union = np.zeros_like(ref)
    for m in masks.values():
        union |= m
    tot = ref.sum()
    rows = []
    for n in names:
        others = np.zeros_like(ref)
        for o in names:
            if o != n:
                others |= masks[o]
        rows.append({"detector": n,
                     "share_of_mask_covered": (masks[n] & ref).sum() / tot,
                     "uniquely_explains_mask": (masks[n] & ref & ~others).sum() / tot,
                     "unmasked_bases_called_frac": (masks[n] & ~ref).sum() / (~ref).sum()})
    rows.append({"detector": "UNION (all detectors)", "share_of_mask_covered": (union & ref).sum() / tot,
                 "uniquely_explains_mask": np.nan,
                 "unmasked_bases_called_frac": (union & ~ref).sum() / (~ref).sum()})
    rows.append({"detector": "unexplained by any detector", "share_of_mask_covered": (ref & ~union).sum() / tot,
                 "uniquely_explains_mask": np.nan, "unmasked_bases_called_frac": np.nan})
    return pd.DataFrame(rows).set_index("detector")


def per_class_recall(masks: dict, truth: np.ndarray) -> pd.DataFrame:
    """Simulation only: recall of each detector on each true repeat class (ground truth, not mask)."""
    names = {1: "tandem", 2: "low-complexity", 3: "interspersed (young)", 4: "interspersed (old)"}
    rows = {}
    for c, nm in names.items():
        sel = truth == c
        rows[nm] = {d: (m & sel).sum() / max(sel.sum(), 1) for d, m in masks.items()}
    rows["unique (false-positive rate)"] = {d: (m & (truth == 0)).sum() / max((truth == 0).sum(), 1) for d, m in masks.items()}
    return pd.DataFrame(rows).T
