"""Run the full pipeline on real marin-dna windows.
usage: python scripts/run_real.py "<glob of .jsonl.zst or .tsv>" [outdir=results/real] [limit_per_file]
Example (full data, after downloading the 64 shards):  python scripts/run_real.py "data/raw/*.jsonl.zst"
"""
import sys, os, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from repeatfinder import io, evaluate as ev, plots
from repeatfinder.pipeline import run_detectors, window_table

pattern = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "results/real"
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)
t = time.time(); ids, seqs = io.read_any(pattern, limit)
c = io.build_corpus(ids, seqs); print(f"loaded {len(c):,} unique windows in {time.time()-t:.0f}s")
qc = io.qc_report(c); json.dump(qc, open(f"{out}/qc_report.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in qc.items() if k != "windows_per_chrom"}, indent=1, default=str))

det = run_detectors(c)
m = dict(det.masks); names = list(m)
bench = ev.benchmark({**m, "UNION": np.logical_or.reduce(list(m.values()))}, c.mask); bench.to_csv(f"{out}/benchmark_vs_softmask.csv")
ev.decomposition(m, c.mask).to_csv(f"{out}/mask_decomposition.csv")
tab = det.tandem.table(c); tab.to_csv(f"{out}/tandem_repeats.csv", index=False)
wt = window_table(c, det); wt.to_csv(f"{out}/window_features.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(bench[["precision", "recall", "f1", "iou", "called_frac"]]); print(f"tandem repeats: {len(tab)}")

plots.hist(wt.gc, "GC content per window", f"{out}/fig_gc.png", "GC fraction")
plots.hist(wt.softmask_frac, "Soft-mask fraction per window", f"{out}/fig_softmask.png", "fraction lowercase")
if len(tab):
    plots.hbar(tab.unit_class.value_counts(), "Most frequent tandem-repeat units", f"{out}/fig_units.png", "repeats")
    plots.hbar(tab.period.value_counts().sort_index(), "Tandem repeats by period", f"{out}/fig_period.png", "repeats", top=50)
plots.hbar(pd.Series(qc["windows_per_chrom"]), "Windows per chromosome", f"{out}/fig_chrom.png", "unique windows", top=25)
