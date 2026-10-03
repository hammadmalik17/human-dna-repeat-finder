import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from repeatfinder import io
from repeatfinder.tandem import find_tandem_repeats, canonical_unit
from repeatfinder.complexity import dust_mask
from repeatfinder.kmers import KmerDB

def rc(s): return s.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_strand_pair_dedup():
    ids, seqs = io.read_tsv(f"{ROOT}/data/real_sample_10.tsv")
    # fabricate the opposite-strand twin of every row, as the real dataset has
    tw_ids = [i[:-1] + ("-" if i.endswith("+") else "+") for i in ids]
    c = io.build_corpus(ids + tw_ids, seqs + [rc(s) for s in seqs])
    assert len(c) == 10 and c.qc["duplicate_coordinate_rows_removed"] == 10
    assert c.qc["strand_pairs_consistent"] == 1.0 and c.qc["mask_pairs_consistent"] == 1.0

def test_revcomp_roundtrip():
    ids, seqs = io.read_tsv(f"{ROOT}/data/real_sample_10.tsv")
    c = io.build_corpus(ids, seqs, canonicalise=False)
    assert (io.revcomp_codes(io.revcomp_codes(c.codes)) == c.codes).all()

def test_tandem_exact():
    rng = np.random.default_rng(0)
    r = lambda n: "".join(rng.choice(list("ACGT"), n))
    s = r(60) + "AC" * 12 + r(171)
    c = io.build_corpus(["NC_000001.11:0-255_+"], [s])
    tab = find_tandem_repeats(c.codes).table(c)
    assert len(tab) == 1 and tab.period[0] == 2 and tab.start[0] == 60 and tab.end[0] == 84

def test_no_calls_on_random():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 4, (5000, 255)).astype(np.uint8)
    assert find_tandem_repeats(codes).mask().mean() < 0.005
    assert dust_mask(codes).mean() < 0.01

def test_canonical_unit():
    assert canonical_unit("CA") == canonical_unit("AC") == canonical_unit("GT") == "AC"

def test_kmer_canonical_counts_strand_independent():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 4, (300, 255)).astype(np.uint8)
    a = KmerDB(9).fit(codes); b = KmerDB(9).fit(io.revcomp_codes(codes))
    assert (a.table == b.table).all()
