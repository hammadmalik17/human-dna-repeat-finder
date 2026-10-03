"""Mismatch-tolerant tandem-repeat finder (periods 1..maxp), numba-accelerated.

For each period p, positions i where s[i]==s[i+p] score +match, others +mismatch.
A maximal-scoring segment (Kadane) of that +/- sequence is an approximate tandem
repeat of period p covering [seg_start, seg_end+p].  Candidates from all periods are
resolved greedily by score (ties -> smaller period), so ACACAC is reported as period 2,
not 4/6/8.  This is the same idea as TRF's "tuple/alignment" scoring, simplified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from .io import decode, revcomp_codes

MAXREP = 8


@njit(cache=True)
def _tandem_one(c, maxp, match, mismatch, min_score, min_ident, dropoff, out):
    L = c.shape[0]
    cap = 512
    cs = np.empty(cap, np.int32)
    ce = np.empty(cap, np.int32)
    cp = np.empty(cap, np.int32)
    csc = np.empty(cap, np.int32)
    cm = np.empty(cap, np.int32)
    n = 0
    for p in range(1, maxp + 1):
        cur = 0
        mcur = 0
        start = 0
        best = 0
        best_end = -1
        best_start = 0
        mbest = 0
        last = L - p  # number of comparable positions
        for i in range(last + 1):
            flush = False
            if i < last:
                if c[i] == c[i + p] and c[i] < 4:
                    cur += match
                    mcur += 1
                else:
                    cur += mismatch
                if cur > best:
                    best = cur
                    best_end = i
                    best_start = start
                    mbest = mcur
                if cur <= 0 or (best - cur) >= dropoff:
                    flush = True
            else:
                flush = True
            if flush:
                if best >= min_score and best_end >= 0 and n < cap:
                    seg = best_end - best_start + 1
                    if seg >= p and mbest >= min_ident * seg:
                        cs[n] = best_start
                        ce[n] = best_end + p + 1
                        cp[n] = p
                        csc[n] = best
                        cm[n] = mbest
                        n += 1
                cur = 0
                mcur = 0
                start = i + 1
                best = 0
                best_end = -1
    # greedy resolution: score desc, period asc
    key = np.empty(n, np.int64)
    for j in range(n):
        key[j] = -np.int64(csc[j]) * 1000 + cp[j]
    order = np.argsort(key)
    k = 0
    for oi in range(n):
        j = order[oi]
        ln = ce[j] - cs[j]
        ok = True
        for a in range(k):
            ov = min(ce[j], out[a, 1]) - max(cs[j], out[a, 0])
            if ov > 0.5 * ln:
                ok = False
                break
        if ok:
            out[k, 0] = cs[j]
            out[k, 1] = ce[j]
            out[k, 2] = cp[j]
            out[k, 3] = csc[j]
            out[k, 4] = cm[j]
            k += 1
            if k == out.shape[0]:
                break
    return k


@njit(parallel=True, cache=True)
def _tandem_all(codes, maxp, match, mismatch, min_score, min_ident, dropoff, maxrep):
    N = codes.shape[0]
    out = np.zeros((N, maxrep, 5), np.int32)
    cnt = np.zeros(N, np.int32)
    for w in prange(N):
        cnt[w] = _tandem_one(codes[w], maxp, match, mismatch, min_score, min_ident, dropoff, out[w])
    return out, cnt


@njit(cache=True)
def _paint(out, cnt, L, min_copies):
    N = out.shape[0]
    m = np.zeros((N, L), np.bool_)
    for w in range(N):
        for r in range(cnt[w]):
            s, e, p = out[w, r, 0], out[w, r, 1], out[w, r, 2]
            if (e - s) >= min_copies * p:
                m[w, s:e] = True
    return m


class TandemResult:
    def __init__(self, out, cnt, L, params):
        self.out, self.cnt, self.L, self.params = out, cnt, L, params

    def mask(self, min_copies=2.0):
        return _paint(self.out, self.cnt, self.L, float(min_copies))

    def table(self, corpus) -> pd.DataFrame:
        """One row per detected repeat with unit, copies, identity and genomic coordinates."""
        w_idx, r_idx = np.nonzero(np.arange(MAXREP)[None, :] < self.cnt[:, None])
        rows = []
        for w, r in zip(w_idx, r_idx):
            s, e, p, sc, nm = (int(v) for v in self.out[w, r])
            seg = corpus.codes[w, s:e]
            unit = _consensus_unit(seg, p)
            rows.append((w, corpus.ids[w], corpus.chrom[w], s, e, e - s, p, (e - s) / p,
                         nm / max(e - s - p, 1), sc, unit, canonical_unit(unit),
                         float(corpus.mask[w, s:e].mean())))
        return pd.DataFrame(rows, columns=["win", "id", "chrom", "start", "end", "length", "period",
                                           "copies", "identity", "score", "unit", "unit_class",
                                           "softmask_overlap"])


def _consensus_unit(seg: np.ndarray, p: int) -> str:
    cols = []
    for j in range(p):
        v = seg[j::p]
        v = v[v < 4]
        cols.append(int(np.bincount(v, minlength=4).argmax()) if len(v) else 4)
    return decode(np.array(cols, dtype=np.uint8))


def canonical_unit(u: str) -> str:
    """Lexicographically smallest rotation over both strands (AC == CA == GT == TG)."""
    comp = str.maketrans("ACGT", "TGCA")
    rc = u.translate(comp)[::-1]
    cands = [x[i:] + x[:i] for x in (u, rc) for i in range(len(u))]
    return min(cands)


def find_tandem_repeats(codes, maxp=50, match=2, mismatch=-5, min_score=24,
                        min_identity=0.70, dropoff=40) -> TandemResult:
    codes = np.ascontiguousarray(codes)
    out, cnt = _tandem_all(codes, maxp, match, mismatch, min_score, min_identity, dropoff, MAXREP)
    return TandemResult(out, cnt, codes.shape[1],
                        dict(maxp=maxp, match=match, mismatch=mismatch, min_score=min_score,
                             min_identity=min_identity))
