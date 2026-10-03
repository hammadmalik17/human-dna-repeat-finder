"""Loading, parsing, canonicalisation and QC for the marin-dna 255bp human windows.

Dataset facts (from the dataset card, not from the SOP):
  * id  = "<RefSeq accession>:<start>-<end>_<strand>"   (0-based half-open)
  * seq = 255 bp, lowercase = repeat soft-masked in the source assembly
  * every genomic window appears twice (+ and - strand); windows overlap by 50%
  * windows come from ENCODE cCRE regions with conserved elements, NOT the whole genome
"""
from __future__ import annotations

import glob
import io as _io
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_RE = re.compile(r"^(?P<acc>[^:]+):(?P<start>\d+)-(?P<end>\d+)_(?P<strand>[+-])$")

# RefSeq GRCh38 accession -> UCSC-style chromosome name
ACC_TO_CHR = {f"NC_0000{i:02d}": f"chr{i}" for i in range(1, 23)}
ACC_TO_CHR.update({"NC_000023": "chrX", "NC_000024": "chrY", "NC_012920": "chrM"})

BASES = np.array(list("ACGTN"))
_LUT = np.full(256, 5, dtype=np.uint8)  # 5 = invalid
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i
_LUT[ord("N")] = _LUT[ord("n")] = 4


def acc_to_chrom(acc: str) -> str:
    return ACC_TO_CHR.get(acc.split(".")[0], acc)


def parse_id(i: str):
    m = ID_RE.match(i)
    if not m:
        return None
    return acc_to_chrom(m["acc"]), int(m["start"]), int(m["end"]), m["strand"]


def decode(codes: np.ndarray) -> str:
    return "".join(BASES[np.minimum(codes, 4)])


def revcomp_codes(codes: np.ndarray) -> np.ndarray:
    """Reverse-complement along the last axis (N stays N)."""
    rc = np.where(codes < 4, 3 - codes, codes).astype(np.uint8)
    return rc[..., ::-1]


@dataclass
class Corpus:
    ids: np.ndarray            # (N,) str
    chrom: np.ndarray          # (N,) str
    start: np.ndarray          # (N,) int64
    end: np.ndarray            # (N,) int64
    strand: np.ndarray         # (N,) str, ORIGINAL strand of the row
    codes: np.ndarray          # (N,L) uint8, A0 C1 G2 T3 N4 (case folded)
    mask: np.ndarray           # (N,L) bool, True = lowercase (soft-masked)
    truth: np.ndarray | None = None   # simulation only: per-base class labels
    qc: dict = field(default_factory=dict)

    def __len__(self):
        return self.codes.shape[0]

    @property
    def L(self):
        return self.codes.shape[1]

    def subset(self, idx) -> "Corpus":
        return Corpus(self.ids[idx], self.chrom[idx], self.start[idx], self.end[idx],
                      self.strand[idx], self.codes[idx], self.mask[idx],
                      None if self.truth is None else self.truth[idx], dict(self.qc))


# ----------------------------------------------------------------------------- readers
def read_tsv(path):
    ids, seqs = [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            a, b = line.split("\t")[:2]
            ids.append(a)
            seqs.append(b)
    return ids, seqs


def read_jsonl_zst(path, limit=None):
    import zstandard as zstd
    ids, seqs = [], []
    with open(path, "rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        for n, line in enumerate(_io.TextIOWrapper(reader, encoding="utf-8")):
            if limit is not None and n >= limit:
                break
            r = json.loads(line)
            ids.append(r["id"])
            seqs.append(r["seq"])
    return ids, seqs


def read_any(pattern, limit_per_file=None):
    """Read one or many files (TSV or .jsonl.zst) matched by a glob pattern."""
    ids, seqs = [], []
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    for f in files:
        i, s = read_jsonl_zst(f, limit_per_file) if f.endswith(".zst") else read_tsv(f)
        ids += i
        seqs += s
    return ids, seqs


# ------------------------------------------------------------------------- build corpus
def build_corpus(ids, seqs, length=255, canonicalise=True) -> Corpus:
    """Validate, encode, orient to the + strand and de-duplicate +/- pairs.

    Returns a Corpus of unique genomic windows (one per coordinate) all in + orientation.
    ``corpus.qc`` holds the numbers for the Phase-1 dataset report.
    """
    qc = {"rows_in": len(ids)}
    parsed = [parse_id(i) for i in ids]
    bad_id = np.array([p is None for p in parsed])
    lens = np.array([len(s) for s in seqs])
    bad_len = lens != length
    qc["bad_id"] = int(bad_id.sum())
    qc["bad_length"] = int(bad_len.sum())

    keep = ~(bad_id | bad_len)
    keep_idx = np.flatnonzero(keep)
    ids = np.array(ids, dtype=object)[keep_idx]
    parsed = [parsed[i] for i in keep_idx]
    seqs = [seqs[i] for i in keep_idx]

    raw = np.frombuffer("".join(seqs).encode("ascii", "replace"), dtype=np.uint8).reshape(-1, length)
    codes = _LUT[raw]
    mask = raw >= 97  # lowercase letters
    invalid = (codes == 5).any(axis=1)
    qc["invalid_char_rows"] = int(invalid.sum())
    qc["rows_with_N"] = int((codes == 4).any(axis=1).sum())
    ok = ~invalid
    ids, parsed, codes, mask = ids[ok], [p for p, o in zip(parsed, ok) if o], codes[ok], mask[ok]

    chrom = np.array([p[0] for p in parsed])
    start = np.array([p[1] for p in parsed], dtype=np.int64)
    end = np.array([p[2] for p in parsed], dtype=np.int64)
    strand = np.array([p[3] for p in parsed])
    qc["rows_valid"] = len(ids)
    qc["strand_counts"] = {s: int((strand == s).sum()) for s in ("+", "-")}

    if canonicalise:
        neg = strand == "-"
        if neg.any():
            codes[neg] = revcomp_codes(codes[neg])
            mask[neg] = mask[neg][:, ::-1]
        key = pd.Series(chrom).astype(str) + ":" + pd.Series(start).astype(str) + "-" + pd.Series(end).astype(str)
        first = ~key.duplicated().to_numpy()
        # sanity check: duplicated coordinates should carry identical (oriented) sequence
        if (~first).any():
            dup_key = key[~first].to_numpy()
            ref = pd.Series(np.flatnonzero(first), index=key[first].to_numpy())
            ref_idx = ref.loc[dup_key].to_numpy()
            dup_idx = np.flatnonzero(~first)
            same = (codes[dup_idx] == codes[ref_idx]).all(axis=1)
            qc["strand_pairs_consistent"] = float(same.mean())
            qc["mask_pairs_consistent"] = float((mask[dup_idx] == mask[ref_idx]).all(axis=1).mean())
        qc["duplicate_coordinate_rows_removed"] = int((~first).sum())
        ids, chrom, start, end, strand = ids[first], chrom[first], start[first], end[first], strand[first]
        codes, mask = codes[first], mask[first]

    qc["windows_unique"] = len(ids)
    return Corpus(ids.astype(str), chrom, start, end, strand, np.ascontiguousarray(codes),
                  np.ascontiguousarray(mask), None, qc)


def qc_report(c: Corpus) -> dict:
    """Phase-1 dataset statistics on the (deduplicated) corpus."""
    acgt = c.codes < 4
    n_acgt = acgt.sum()
    counts = {b: int((c.codes == i).sum()) for i, b in enumerate("ACGT")}
    gc = ((c.codes == 1) | (c.codes == 2)).sum(1) / np.maximum(acgt.sum(1), 1)
    mfrac = c.mask.mean(1)
    seqhash = pd.util.hash_array(c.codes.view(np.dtype((np.void, c.L))).ravel()) if len(c) else np.array([])
    rep = dict(c.qc)
    rep.update({
        "total_bases": int(c.codes.size),
        "nucleotide_counts": counts,
        "gc_mean": float(gc.mean()), "gc_sd": float(gc.std()),
        "softmask_fraction_bases": float(c.mask.mean()),
        "windows_fully_unmasked": float((mfrac == 0).mean()),
        "windows_>50%_masked": float((mfrac > 0.5).mean()),
        "windows_fully_masked": float((mfrac == 1).mean()),
        "identical_sequence_duplicates": int(len(seqhash) - len(np.unique(seqhash))),
        "windows_per_chrom": pd.Series(c.chrom).value_counts().to_dict(),
    })
    return rep
