# Human DNA Repeat-Finding System: milestone 1

Reframed from the SOP: **can an alignment-free, de novo repeat finder rediscover what RepeatMasker/TRF
annotated (the soft-mask), and what does it find or miss?** The soft-mask is an *independent* base-resolution
ground truth, so evaluation is not circular (the SOP's "labels from our own rules" were).

## Corrections to the SOP, from the dataset card
- The windows are **ENCODE cCRE enhancer regions with conserved elements, not the whole genome**. Conclusions are about those regions.
- **Every window appears twice** (+ and − strand, reverse-complemented) and neighbours overlap 50%. The loader orients to + and dedupes on coordinates (tested). Without this, all stats double-count.
- ids use RefSeq accessions (NC_000017.11), mapped to chr17 for chromosome-aware splits.

## What's built
| module | what it does |
|---|---|
| `io.py` | read `.jsonl.zst`/TSV, parse ids, validate, orient + dedupe strands, Phase-1 QC report |
| `tandem.py` | mismatch-tolerant tandem finder, periods 1–50 (max-scoring-segment per period, greedy resolution), unit/copies/identity table |
| `complexity.py` | DUST-style low-complexity mask, entropy, homopolymer features |
| `kmers.py` | **corpus-wide** canonical k-mer counts; auto-threshold from a Poisson null; **seed-and-extend** (k=13 seeds extended through k=11 hits) |
| `evaluate.py` | base-level precision/recall/F1/IoU/AUC vs soft-mask; mask decomposition; per-class recall (sim only) |
| `simulate.py` | windows with planted repeats + known truth, incl. old repeats a k-mer method can't see |

## Results so far (read the caveats)
**Simulated corpus, 376k windows (same size as the real unique set), planted truth. This validates the method; it is NOT biology.**

| detector | precision | recall | F1 |
|---|---|---|---|
| tandem | 0.86 | 0.09 | 0.17 |
| DUST | 0.58 | 0.10 | 0.17 |
| k-mer (k=13) | 0.93 | 0.52 | 0.67 |
| **seed+extend 13→11** | 0.88 | 0.57 | **0.69** |
| union | 0.82 | 0.60 | 0.70 |

- Tandem detector recovers 70% of planted tandem bases; k-mer seed+extend recovers 64% of young interspersed repeats.
- **Old (~30% diverged) repeats: 2% recall.** ~40% of the simulated mask is unexplained. This is the case for a learned model (milestone 2).
- k trade-off at corpus size 376k: k=10 → recall .67 / precision .24; k=13 → recall .52 / precision .93.
- Simulated repeat divergences, families and abundances are my assumptions, so absolute numbers will differ on real data.

**Real data: only 10 windows were reachable from my sandbox** (Hugging Face blocked). On them, DUST intervals line up with the real soft-mask where it fires (e.g. 203–236 vs 205–234). The corpus-level detector can't be assessed on 10 windows. Real-data numbers need the full run.

## Run on the real data
```
pip install -r requirements.txt
python scripts/download_shards.py 8                       # start with 8 shards (~94k rows), iterate fast
python scripts/run_real.py "data/raw/*.jsonl.zst"         # writes results/real/
python scripts/run_benchmark.py 376000                    # re-run simulation
python -m pytest tests
```
`run_real.py` writes the QC report, benchmark vs soft-mask, mask decomposition, a tandem-repeat table (unit, copies, identity,
soft-mask overlap), per-window features and figures. Detection on 376k simulated windows took ~1 min on 2 cores (RAM not measured; arrays are ~100 MB each, so budget a few GB).
`download_shards.py` was not tested here.

## Next milestone
1. Run on real data; replace the simulation-based claims with real numbers.
2. **Base-level CNN/BiGRU predicting the soft-mask from sequence**, split by chromosome (e.g. train chr1–17, test chr18–X). Compare to the unsupervised union above. Report recall on repeats the k-mer method misses (old repeats).
3. Stitch overlapping windows back into contiguous intervals (adjacent starts differ by 128) for longer repeat context.
4. Embeddings + UMAP only if the CNN works.
