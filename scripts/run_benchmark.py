"""Method validation on simulated windows with planted ground truth.
usage: python scripts/run_benchmark.py [n_windows=376000]"""
import sys, time, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from repeatfinder.simulate import simulate
from repeatfinder.pipeline import run_detectors
from repeatfinder import evaluate as ev, plots

n = int(sys.argv[1]) if len(sys.argv) > 1 else 376000
out = "results/simulated"; os.makedirs(out, exist_ok=True)
t = time.time(); c = simulate(n, seed=0)
det = run_detectors(c); print(f"simulated+detected {n} windows in {time.time()-t:.0f}s; k-mer thr={det.kmer_threshold}")
m = dict(det.masks); m["UNION"] = np.logical_or.reduce(list(m.values()))
short = {k: k.split(" (")[0] if k.startswith(("low", "tandem")) else k for k in m}
m = {short[k]: v for k, v in m.items()}
bench = ev.benchmark(m, c.mask); bench.to_csv(f"{out}/benchmark_vs_softmask.csv")
pc = ev.per_class_recall(m, c.truth); pc.to_csv(f"{out}/recall_by_true_class.csv")
dec = ev.decomposition({k: v for k, v in m.items() if k != "UNION"}, c.mask); dec.to_csv(f"{out}/mask_decomposition.csv")
depth_auc = ev.auc(det.kmer_depth, c.mask); print("AUC of raw k-mer depth vs soft-mask:", round(depth_auc, 3))
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(bench[["precision", "recall", "f1", "iou"]]); print(pc); print(dec)
plots.grouped_bars(bench[["precision", "recall", "f1"]], "Base-level agreement with the soft-mask (simulated)", f"{out}/fig_benchmark.png")
plots.grouped_bars(pc.drop(columns=["UNION"]).T.rename(columns=lambda s: s).T.T if False else pc.drop(columns=["UNION"]).iloc[:4].T,
                   "Recall by TRUE repeat class (simulation ground truth)", f"{out}/fig_recall_by_class.png", ylim=(0, 1.1))
