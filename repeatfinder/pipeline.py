"""End-to-end detection: tandem + low-complexity + corpus-level k-mer depth -> masks and features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .complexity import dust_mask, window_complexity
from .kmers import KmerDB, call_mask, seed_and_extend
from .tandem import find_tandem_repeats


@dataclass
class Detections:
    tandem: object
    masks: dict          # name -> (N,L) bool
    kmer_depth: np.ndarray
    kmer_threshold: int
    kmer_lambda: float
    kmerdb: KmerDB


def run_detectors(corpus, k=13, k_extend=11, kmer_threshold=None, tandem_kw=None, dust_kw=None,
                  db=None, db_ext=None) -> Detections:
    """k = seed k-mer length (precise); k_extend = shorter k used to extend seeds (sensitive); None disables."""
    codes = corpus.codes
    tr = find_tandem_repeats(codes, **(tandem_kw or {}))
    db = db or KmerDB(k).fit(codes)
    thr, lam = db.auto_threshold() if kmer_threshold is None else (kmer_threshold, float("nan"))
    depth = db.depth(codes)
    masks = {
        "tandem": tr.mask(),
        "low_complexity (DUST)": dust_mask(codes, **(dust_kw or {})),
        f"high-copy k-mer (k={db.k}, >={thr})": call_mask(depth, thr),
    }
    if k_extend:
        db_ext = db_ext or KmerDB(k_extend).fit(codes)
        thr_e, _ = db_ext.auto_threshold()
        masks[f"seed+extend (k={db.k}->{k_extend})"] = seed_and_extend(depth, thr, db_ext.depth(codes), thr_e)
    return Detections(tr, masks, depth, thr, lam, db)


def window_table(corpus, det: Detections) -> pd.DataFrame:
    f = window_complexity(corpus.codes)
    df = pd.DataFrame({"id": corpus.ids, "chrom": corpus.chrom, "start": corpus.start, "end": corpus.end, **f})
    df["softmask_frac"] = corpus.mask.mean(1)
    for n, m in det.masks.items():
        df[f"frac[{n}]"] = m.mean(1)
    df["n_tandem"] = det.tandem.cnt
    tab = det.tandem.table(corpus)
    if len(tab):
        g = tab.groupby("win")
        df["longest_tandem"] = g["length"].max().reindex(range(len(df))).fillna(0).to_numpy()
        df["max_copies"] = g["copies"].max().reindex(range(len(df))).fillna(0).to_numpy()
    else:
        df["longest_tandem"] = 0; df["max_copies"] = 0
    df["max_kmer_depth"] = det.kmer_depth.max(1)
    return df
