"""Simulate 255 bp windows with planted repeats and a RepeatMasker-like soft-mask.

PURPOSE: method validation with known ground truth (like a unit test at scale).  The numbers
that come out of this are NOT biological findings.  The simulator deliberately includes
things a k-mer method cannot see (very old, ~30% diverged repeats that RepeatMasker would
still mask) so that the benchmark reports an honest ceiling.

Per-base truth classes:  0 unique, 1 tandem/microsatellite, 2 low-complexity,
                         3 young interspersed (div <= 22%), 4 old interspersed (div > 22%)
"""
from __future__ import annotations

import numpy as np

from .io import Corpus, revcomp_codes

T_NONE, T_TANDEM, T_LOWC, T_YOUNG, T_OLD = 0, 1, 2, 3, 4


def _rand_seq(rng, n, gc):
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return rng.choice(4, size=n, p=p).astype(np.uint8)


def _mutate(rng, seq, div, indel_frac=0.12):
    """Substitutions at rate div*(1-indel_frac); indels at div*indel_frac."""
    seq = seq.copy()
    sub = rng.random(len(seq)) < div * (1 - indel_frac)
    seq[sub] = (seq[sub] + rng.integers(1, 4, sub.sum())) % 4
    out = []
    for b in seq:
        r = rng.random()
        if r < div * indel_frac / 2:
            continue  # deletion
        out.append(b)
        if r > 1 - div * indel_frac / 2:
            out.append(rng.integers(0, 4))  # insertion
    return np.array(out, dtype=np.uint8)


def make_families(rng, n_young=6, n_old=2):
    fams = []
    for i in range(n_young):  # Alu/L1/LTR-like consensus sequences of different lengths
        fams.append(("young", _rand_seq(rng, int(rng.choice([300, 300, 450, 900, 2000])), rng.uniform(.4, .6))))
    for i in range(n_old):    # MIR/L2-like: old, highly diverged
        fams.append(("old", _rand_seq(rng, 260, 0.45)))
    return fams


def simulate(n=50000, seed=0, p_int=0.22, p_old=0.04, p_tandem=0.12, p_lowc=0.03, families=None) -> Corpus:
    rng = np.random.default_rng(seed)
    fams = families or make_families(rng)
    young = [f for k, f in fams if k == "young"]
    old = [f for k, f in fams if k == "old"]
    L = 255
    codes = np.zeros((n, L), np.uint8)
    mask = np.zeros((n, L), bool)
    truth = np.zeros((n, L), np.uint8)
    for w in range(n):
        gc = float(np.clip(rng.normal(0.47, 0.07), 0.3, 0.7))
        segs = []  # (codes, truth_class, masked)
        r = rng.random(5)
        if r[0] < p_int:
            fam = young[rng.integers(len(young))]
            flen = int(min(len(fam), rng.integers(50, 330)))
            st = rng.integers(0, len(fam) - flen + 1)
            div = rng.uniform(0.02, 0.22)
            frag = _mutate(rng, fam[st:st + flen], div)
            if rng.random() < .5:
                frag = revcomp_codes(frag)
            segs.append((frag, T_YOUNG, True))
        if r[1] < p_old:
            fam = old[rng.integers(len(old))]
            flen = int(rng.integers(60, 200))
            st = rng.integers(0, len(fam) - flen + 1)
            frag = _mutate(rng, fam[st:st + flen], rng.uniform(0.25, 0.34))
            if rng.random() < .5:
                frag = revcomp_codes(frag)
            segs.append((frag, T_OLD, True))
        if r[2] < p_tandem:
            if rng.random() < 0.8:
                period = int(rng.choice([1, 2, 2, 3, 4, 4, 5, 6]))
            else:
                period = int(rng.integers(7, 41))
            unit = _rand_seq(rng, period, 0.5)
            total = int(rng.integers(max(16, 2 * period + 2), 90))
            rep = np.tile(unit, total // period + 1)[:total]
            rep = _mutate(rng, rep, rng.uniform(0, 0.15))
            segs.append((rep, T_TANDEM, bool(rng.random() < 0.9)))  # TRF/RM mask most of them
        if r[3] < p_lowc:
            n_ = int(rng.integers(20, 60))
            comp = rng.dirichlet([0.4, 0.4, 0.4, 0.4])
            segs.append((rng.choice(4, size=n_, p=comp).astype(np.uint8), T_LOWC, bool(rng.random() < 0.7)))
        # lay out: unique background between inserts
        order = rng.permutation(len(segs))
        parts, tr, mk = [], [], []
        for idx in order:
            gap = int(rng.integers(5, 120))
            g = _rand_seq(rng, gap, gc)
            parts.append(g); tr.append(np.zeros(gap, np.uint8)); mk.append(np.zeros(gap, bool))
            s, cls, masked = segs[idx]
            # RepeatMasker-like boundary jitter on the mask
            m = np.full(len(s), masked)
            j = int(rng.integers(0, 4))
            if masked and len(s) > 12 and j:
                m[:j] = False if rng.random() < .5 else m[:j]
                m[len(s) - j:] = False if rng.random() < .5 else m[len(s) - j:]
            parts.append(s); tr.append(np.full(len(s), cls, np.uint8)); mk.append(m)
        tail = _rand_seq(rng, L + 100, gc)
        parts.append(tail); tr.append(np.zeros(len(tail), np.uint8)); mk.append(np.zeros(len(tail), bool))
        seq = np.concatenate(parts); t = np.concatenate(tr); m = np.concatenate(mk)
        off = int(rng.integers(0, max(1, len(seq) - L - 99)))
        codes[w], truth[w], mask[w] = seq[off:off + L], t[off:off + L], m[off:off + L]
    chrom = rng.choice([f"chr{i}" for i in range(1, 23)] + ["chrX"], size=n)
    start = (np.arange(n) * 1000).astype(np.int64)
    ids = np.array([f"SIM:{c}:{s}-{s + L}_+" for c, s in zip(chrom, start)])
    return Corpus(ids, chrom, start, start + L, np.full(n, "+"), codes, mask, truth,
                  {"simulated": True, "families": len(fams)})
