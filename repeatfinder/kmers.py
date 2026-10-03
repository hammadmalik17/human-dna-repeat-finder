"""Corpus-wide canonical k-mer counting: the de novo detector for *interspersed* repeats.

Inside one 255 bp window, k-mer counting can only rediscover tandem structure.  An Alu, L1 or
LTR fragment is a repeat because the same k-mers recur in thousands of *different* windows.
So we count, for every canonical k-mer, the number of windows that contain it, then call a base
repetitive when it is covered by a k-mer whose window-count is far above the uniform-random null.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.stats import poisson


@njit(cache=True)
def _count(codes, k, table):
    N, L = codes.shape
    mask = (1 << (2 * k)) - 1
    shift = 2 * (k - 1)
    buf = np.empty(L - k + 1, np.int64)
    for w in range(N):
        fwd = 0
        rev = 0
        valid = 0
        m = 0
        for i in range(L):
            b = codes[w, i]
            if b > 3:
                valid = 0
                fwd = 0
                rev = 0
                continue
            fwd = ((fwd << 2) | b) & mask
            rev = (rev >> 2) | ((3 - b) << shift)
            valid += 1
            if valid >= k:
                buf[m] = fwd if fwd < rev else rev
                m += 1
        s = np.sort(buf[:m])
        prev = -1
        for j in range(m):
            if s[j] != prev:
                if table[s[j]] < 65535:
                    table[s[j]] += 1
                prev = s[j]


@njit(parallel=True, cache=True)
def _depth(codes, k, table):
    N, L = codes.shape
    mask = (1 << (2 * k)) - 1
    shift = 2 * (k - 1)
    depth = np.zeros((N, L), np.uint16)
    for w in prange(N):
        fwd = 0
        rev = 0
        valid = 0
        for i in range(L):
            b = codes[w, i]
            if b > 3:
                valid = 0
                fwd = 0
                rev = 0
                continue
            fwd = ((fwd << 2) | b) & mask
            rev = (rev >> 2) | ((3 - b) << shift)
            valid += 1
            if valid >= k:
                v = table[fwd if fwd < rev else rev]
                for j in range(i - k + 1, i + 1):
                    if v > depth[w, j]:
                        depth[w, j] = v
    return depth


@njit(parallel=True, cache=True)
def _bridge_filter(flag, max_gap, min_len):
    N, L = flag.shape
    out = flag.copy()
    for w in prange(N):
        # close short gaps
        i = 0
        while i < L:
            if not out[w, i]:
                j = i
                while j < L and not out[w, j]:
                    j += 1
                if i > 0 and j < L and (j - i) <= max_gap:
                    out[w, i:j] = True
                i = j
            else:
                i += 1
        # drop short islands
        i = 0
        while i < L:
            if out[w, i]:
                j = i
                while j < L and out[w, j]:
                    j += 1
                if (j - i) < min_len:
                    out[w, i:j] = False
                i = j
            else:
                i += 1
    return out


class KmerDB:
    def __init__(self, k=13):
        assert k <= 15, "table is 4^k entries"
        self.k = k
        self.table = np.zeros(4 ** k, dtype=np.uint16)
        self.n_windows = 0

    def fit(self, codes):
        _count(np.ascontiguousarray(codes), self.k, self.table)
        self.n_windows += codes.shape[0]
        return self

    def auto_threshold(self, expected_false=1.0, minimum=4):
        """Smallest count t with  n_canonical * P(Poisson(lambda) >= t) < expected_false."""
        n_can = 4 ** self.k / 2
        lam = self.table.sum(dtype=np.float64) / n_can
        t = minimum
        while n_can * poisson.sf(t - 1, lam) > expected_false:
            t += 1
        return int(t), float(lam)

    def depth(self, codes):
        """(N,L) uint16: for every base, the highest window-count among k-mers covering it."""
        return _depth(np.ascontiguousarray(codes), self.k, self.table)


def call_mask(depth, threshold, max_gap=8, min_len=20):
    return _bridge_filter(depth >= threshold, max_gap, min_len)


@njit(parallel=True, cache=True)
def _hysteresis(strong, weak):
    """Keep every run of `weak` bases that contains at least one `strong` base (seed-and-extend)."""
    N, L = strong.shape
    out = np.zeros((N, L), np.bool_)
    for w in prange(N):
        i = 0
        while i < L:
            if weak[w, i] or strong[w, i]:
                j = i
                seeded = False
                while j < L and (weak[w, j] or strong[w, j]):
                    if strong[w, j]:
                        seeded = True
                    j += 1
                if seeded:
                    out[w, i:j] = True
                i = j
            else:
                i += 1
    return out


def seed_and_extend(strong_depth, strong_thr, weak_depth, weak_thr, max_gap=8, min_len=20):
    """Dual-threshold call: high-precision seeds (long k) extended through sensitive hits (short k)."""
    strong = call_mask(strong_depth, strong_thr, max_gap, min_len)
    weak = _bridge_filter(weak_depth >= weak_thr, max_gap, 1)
    return _bridge_filter(_hysteresis(strong, weak), max_gap, min_len)
