"""
Schematic diagrams for the academic report (architecture, workflow, algorithm flowcharts).
Drawn with matplotlib boxes/arrows so the report has no external dependencies.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.config import FIGURES_DIR

DIAG_DIR = FIGURES_DIR / "diagrams"

FILL = {"data": "#dbe9f6", "proc": "#e8f3e1", "model": "#fdebd3", "out": "#efe3f5", "dec": "#fff6c9"}


def _box(ax, x, y, w, h, text, kind="proc", fs=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=FILL[kind], ec="#33415c", lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.1, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, text, fontsize=7.5, color="#33415c")


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def _save(fig, name):
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    p = DIAG_DIR / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def architecture():
    fig, ax = _canvas(10, 6.2)
    _box(ax, 1.2, 5.4, 2.0, 0.7, "UCI 554 CNNpred\n5 CSV files\n(1,984 x 84 each)", "data")
    _box(ax, 3.7, 5.4, 2.2, 0.7, "data_loader.py\nSHA-256 + shape checks", "proc")
    _box(ax, 6.3, 5.4, 2.4, 0.7, "preprocessing.py\nsentinels, warm-up,\nRep A / Rep B, split", "proc")
    _box(ax, 8.9, 5.4, 1.9, 0.7, "feature_engineering.py\nfeatures, Rep C1/C2", "proc", fs=7.5)
    _arrow(ax, 2.2, 5.4, 2.6, 5.4); _arrow(ax, 4.8, 5.4, 5.1, 5.4); _arrow(ax, 7.5, 5.4, 7.95, 5.4)
    row = [(1.0, "descriptive_\nanalytics.py"), (2.85, "visualization.py\n(14 figures)"), (4.7, "association_\nrules.py (FP-Growth)"),
           (6.55, "pincer_search.py\n(MFIs)"), (8.4, "classification.py\n(Naive Bayes)")]
    for x, t in row:
        _box(ax, x, 3.7, 1.7, 0.75, t, "model", fs=8)
        _arrow(ax, x, 4.65, x, 4.08)
    L = dict(color="#33415c", lw=1.1)
    ax.plot([8.9, 8.9], [5.05, 4.65], **L); ax.plot([1.0, 9.75], [4.65, 4.65], **L)
    ax.plot([9.75, 9.75], [4.65, 2.95], **L); ax.plot([2.2, 9.75], [2.95, 2.95], **L)
    _box(ax, 2.2, 2.1, 2.4, 0.75, "clustering.py\nDBSCAN, PCA, t-SNE, Ward", "model")
    _box(ax, 5.0, 2.1, 2.4, 0.75, "pyspark_pipeline.py\nSpark NB + FPGrowth", "model")
    _box(ax, 7.8, 2.1, 2.4, 0.75, "benchmark_engine.py\nscaling, Amdahl, RSS", "model")
    for x in (2.2, 5.0, 7.8):
        _arrow(ax, x, 2.95, x, 2.48)
    _box(ax, 2.2, 0.55, 2.6, 0.7, "outputs/tables (CSV)\noutputs/figures (PNG)", "out")
    _box(ax, 5.0, 0.55, 2.4, 0.7, "notebooks/ (11 executed)\ntests/ (pytest)", "out")
    _box(ax, 7.8, 0.55, 2.6, 0.7, "generate_pdf_report.py\nreports/*.pdf", "out")
    for x in (2.2, 5.0, 7.8):
        _arrow(ax, x, 1.72, x, 0.92)
    ax.text(5, 6.05, "run_all.py  ->  src/run_pipeline.py  ->  build_notebooks.py  ->  pytest", ha="center", fontsize=9, style="italic")
    return _save(fig, "diag_architecture.png")


def workflow():
    fig, ax = _canvas(10, 3.2)
    steps = [("Ingest &\nvalidate", "data"), ("Clean:\nsentinels, warm-up", "proc"), ("Select &\ntransform", "proc"),
             ("Descriptive\nanalytics + EDA", "proc"), ("FP-Growth &\nPincer Search", "model"), ("Naive Bayes\n& DBSCAN", "model"),
             ("PySpark +\nbenchmarks", "model"), ("Report &\ntests", "out")]
    xs = [0.65 + i * 1.25 for i in range(len(steps))]
    for x, (t, k) in zip(xs, steps):
        _box(ax, x, 1.9, 1.1, 0.95, t, k, fs=8)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + 0.55, 1.9, b - 0.55, 1.9)
    ax.text(5, 0.7, "Train-only fitting: thresholds, scalers, NB parameters, majority baseline (1,425 rows)  |  "
            "Test: 357 untouched rows after a 1-day embargo", ha="center", fontsize=8.5)
    return _save(fig, "diag_workflow.png")


def fp_growth_flow():
    fig, ax = _canvas(6.5, 7.2)
    s = [(6.6, "Binary transactions\n(N rows x items)", "data"), (5.6, "Scan 1: item counts;\ndrop items with count < ceil(minsup*N)", "proc"),
         (4.6, "Scan 2: insert each transaction\n(items in frequency order) into FP-tree", "proc"),
         (3.6, "For each item (bottom-up):\nconditional pattern base -> conditional FP-tree", "proc"),
         (2.6, "Recurse until tree is a single path;\nemit all combinations as frequent itemsets", "proc"),
         (1.6, "Rules X => Y from each itemset:\nkeep conf >= minconf; compute lift, conviction", "model"),
         (0.6, "Fisher exact test + BH-FDR;\npredictive rules re-scored on test set", "out")]
    for y, t, k in s:
        _box(ax, 3.25, y, 5.2, 0.72, t, k)
    for (y1, *_), (y2, *_) in zip(s[:-1], s[1:]):
        _arrow(ax, 3.25, y1 - 0.36, 3.25, y2 + 0.36)
    return _save(fig, "diag_fpgrowth_flow.png")


def pincer_flow():
    fig, ax = _canvas(7.5, 8.6)
    _box(ax, 3.75, 8.1, 5.6, 0.6, "k = 1;  C1 = all single items;  MFCS = {all items};  MFS = {}", "data")
    _box(ax, 3.75, 7.1, 5.6, 0.7, "Count support of C_k  AND  of every element of MFCS (one pass)", "proc")
    _box(ax, 3.75, 6.05, 5.6, 0.75, "Frequent MFCS elements -> MFS (top-down discovery)\nL_k = frequent C_k;  S_k = infrequent C_k", "proc")
    _box(ax, 3.75, 4.95, 5.6, 0.8, "MFCS-gen: for each s in S_k and each m in MFCS with s <= m:\nreplace m by {m - {e} : e in s}, keep only maximal (antichain)", "model")
    _box(ax, 3.75, 3.85, 5.6, 0.75, "C_{k+1} = Apriori join of L_k + subset prune\n+ keep only candidates contained in some MFCS element", "proc")
    _box(ax, 3.75, 2.75, 3.6, 0.7, "C_{k+1} empty  or  MFCS empty ?", "dec")
    _box(ax, 6.6, 1.6, 1.6, 0.6, "k = k + 1", "proc")
    _box(ax, 2.6, 1.3, 4.4, 0.8, "MFS := maximal( MFS  U  frequent sets\nnot covered by MFS );  return MFS", "out")
    for y1, y2 in [(7.8, 7.45), (6.75, 6.43), (5.67, 5.35), (4.55, 4.23), (3.47, 3.1)]:
        _arrow(ax, 3.75, y1, 3.75, y2)
    _arrow(ax, 3.0, 2.4, 2.6, 1.7, "yes"); _arrow(ax, 4.9, 2.6, 6.4, 1.9, "no")
    ax.plot([7.4, 7.4, 7.25], [1.6, 7.1, 7.1], color="#33415c", lw=1.1)
    ax.plot([6.6 + 0.8, 7.4], [1.6, 1.6], color="#33415c", lw=1.1)
    _arrow(ax, 7.25, 7.1, 6.57, 7.1)
    return _save(fig, "diag_pincer_flow.png")


def dbscan_flow():
    fig, ax = _canvas(6.5, 6.0)
    s = [(5.4, "Regime features (5) -> StandardScaler (fit on train rows)", "data"),
         (4.4, "MinPts = 2*d - 1 = 9; k-distance curve (9-NN);\nknee = max distance to chord -> eps", "proc"),
         (3.4, "Core point: >= MinPts neighbours within eps;\nexpand clusters by density-reachability", "model"),
         (2.4, "Points reachable from no core -> noise (-1)", "model"),
         (1.4, "Profile clusters; silhouette on non-noise points;\nPCA / t-SNE projections; Ward HAC dendrogram", "out")]
    for y, t, k in s:
        _box(ax, 3.25, y, 5.6, 0.75, t, k)
    for (y1, *_), (y2, *_) in zip(s[:-1], s[1:]):
        _arrow(ax, 3.25, y1 - 0.37, 3.25, y2 + 0.37)
    return _save(fig, "diag_dbscan_flow.png")


def build_all_diagrams():
    return {"architecture": architecture(), "workflow": workflow(), "fpgrowth": fp_growth_flow(),
            "pincer": pincer_flow(), "dbscan": dbscan_flow()}


if __name__ == "__main__":
    print(build_all_diagrams())
