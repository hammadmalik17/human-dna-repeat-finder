"""Low-complexity detection (symmetric-DUST style) plus per-window complexity features."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _dust_one(c, W, T, out):
    L = c.shape[0]
    ntri = L - 2
    tri = np.empty(ntri, np.int16)
    for i in range(ntri):
        if c[i] < 4 and c[i + 1] < 4 and c[i + 2] < 4:
            tri[i] = c[i] * 16 + c[i + 1] * 4 + c[i + 2]
        else:
            tri[i] = -1
    nwin = L - W + 1
    if nwin <= 0:
        return
    l = W - 2
    counts = np.zeros(64, np.int32)
    s = 0.0
    flagged = np.zeros(nwin, np.bool_)
    for i in range(l):
        t = tri[i]
        if t >= 0:
            s += counts[t]
            counts[t] += 1
    flagged[0] = (s / (l - 1)) > T
    for w in range(1, nwin):
        t = tri[w - 1]
        if t >= 0:
            counts[t] -= 1
            s -= counts[t]
        t = tri[w + l - 1]
        if t >= 0:
            s += counts[t]
            counts[t] += 1
        flagged[w] = (s / (l - 1)) > T
    # merge runs of flagged windows, then trim ends whose triplet is rare (< max(3, 25% of the dominant triplet)) inside the run
    w = 0
    while w < nwin:
        if not flagged[w]:
            w += 1
            continue
        a = w
        while w < nwin and flagged[w]:
            w += 1
        b = w - 1
        lo = a
        hi = b + W  # exclusive base end
        cnt = np.zeros(64, np.int32)
        for i in range(lo, hi - 2):
            if tri[i] >= 0:
                cnt[tri[i]] += 1
        mx = 0
        for t in range(64):
            if cnt[t] > mx:
                mx = cnt[t]
        thr = max(3, mx // 4)  # a triplet must be well represented relative to the dominant one
        left = lo
        while left < hi - 2 and (tri[left] < 0 or cnt[tri[left]] < thr):
            left += 1
        right = hi - 3
        while right > left and (tri[right] < 0 or cnt[tri[right]] < thr):
            right -= 1
        if right >= left:
            out[left:right + 3] = True


@njit(parallel=True, cache=True)
def _dust_all(codes, W, T):
    N, L = codes.shape
    out = np.zeros((N, L), np.bool_)
    for w in prange(N):
        _dust_one(codes[w], W, T, out[w])
    return out


def dust_mask(codes, window=32, threshold=1.0):
    """Boolean (N,L) mask of low-complexity bases (triplet-redundancy score > threshold)."""
    return _dust_all(np.ascontiguousarray(codes), window, float(threshold))


@njit(cache=True)
def _max_homopolymer(codes):
    N, L = codes.shape
    out = np.zeros(N, np.int32)
    for w in range(N):
        best = 1
        run = 1
        for i in range(1, L):
            if codes[w, i] == codes[w, i - 1] and codes[w, i] < 4:
                run += 1
                if run > best:
                    best = run
            else:
                run = 1
        out[w] = best
    return out


def window_complexity(codes):
    """Per-window features: GC, mononucleotide entropy, trinucleotide entropy, longest homopolymer."""
    N, L = codes.shape
    acgt = codes < 4
    n = np.maximum(acgt.sum(1), 1)
    freqs = np.stack([(codes == b).sum(1) for b in range(4)], 1) / n[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = -np.nansum(np.where(freqs > 0, freqs * np.log2(freqs), 0), 1)
    c = codes.astype(np.int32)
    tri = c[:, :-2] * 16 + c[:, 1:-1] * 4 + c[:, 2:]
    tri = np.where((c[:, :-2] < 4) & (c[:, 1:-1] < 4) & (c[:, 2:] < 4), tri, 64)
    tc = np.zeros((N, 65))
    for t in range(65):
        tc[:, t] = (tri == t).sum(1)
    tc = tc[:, :64]
    p = tc / np.maximum(tc.sum(1, keepdims=True), 1)
    h3 = -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0).sum(1)
    return {
        "gc": ((codes == 1) | (codes == 2)).sum(1) / n,
        "entropy_1": h1,
        "entropy_3": h3,
        "unique_3mers": (tc > 0).sum(1),
        "max_homopolymer": _max_homopolymer(np.ascontiguousarray(codes)),
    }
