"""Static figures. Palette: Okabe-Ito subset (colour-blind safe), fixed order, direct value labels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True, "figure.dpi": 130})


def grouped_bars(df, title, path, ylabel="", ylim=(0, 1.05), fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(df)), 4.2))
    cols = list(df.columns); n = len(cols); x = np.arange(len(df)); w = .8 / n
    for j, c in enumerate(cols):
        b = ax.bar(x + (j - (n - 1) / 2) * w, df[c].values, w * .92, label=c, color=COL[j % len(COL)])
        for r, v in zip(b, df[c].values):
            ax.text(r.get_x() + r.get_width() / 2, v + .01, fmt.format(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=0, fontsize=8)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=min(n, 4), loc="upper center", bbox_to_anchor=(.5, -.18))
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def hbar(series, title, path, xlabel="", top=20, color=COL[0]):
    s = series.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, .28 * len(s) + 1.2))
    ax.barh(s.index.astype(str), s.values, color=color)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:,.0f}" if v >= 10 else f" {v:.3g}", va="center", fontsize=7)
    ax.set_title(title, loc="left", fontweight="bold"); ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def hist(values, title, path, xlabel, bins=50, color=COL[0]):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(values, bins=bins, color=color); ax.set_xlabel(xlabel); ax.set_ylabel("windows")
    ax.set_title(title, loc="left", fontweight="bold"); fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
