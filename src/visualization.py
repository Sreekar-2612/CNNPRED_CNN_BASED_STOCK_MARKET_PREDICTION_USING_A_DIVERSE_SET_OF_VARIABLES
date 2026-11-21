"""
Data Visualization Module for CNNpred Big Data Pipeline.
Produces 12 curriculum-aligned, publication-quality figures at 300 DPI:
1. Multi-Index Normalized Trajectories
2. Rolling Volatility Clustering (Mandelbrot stylized facts)
3. Yield Curve & Credit Spread Regime Dynamics
4. Multi-Asset Spearman Rank Correlation Heatmap
5. Return Distribution & Heavy Tails (Fat Tails / Kurtosis / QQ-Plot)
6. Maximum Drawdown & Underwater Analysis
7. PCA Scree Plot & Cumulative Explained Variance
8. DBSCAN k-Distance Graph with Automated Knee Locator
9. Market Regime Clusters in 2D Latent Feature Space
10. Truncated Ward HAC Dendrogram (p=30)
11. Association Rules: Support vs Confidence vs Lift (Conviction Colormap)
12. Distributed PySpark Strong Scaling Speedup & Amdahl's Law Asymptote
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

from src.config import FIGURES_DIR, INDEX_NAMES, PROCESSED_DATA_DIR


# Styling standards
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_01_normalized_trajectories(dfs: Dict[str, pd.DataFrame]) -> Path:
    """1. Multi-Index Normalized Price Trajectories (Base 100)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"DJI": "#1f77b4", "NASDAQ": "#ff7f0e", "NYSE": "#2ca02c", "RUSSELL": "#d62728", "S&P": "#9467bd"}

    for name in INDEX_NAMES:
        df = dfs[name]
        dates = pd.to_datetime(df["Date"])
        base_val = df["Close"].iloc[0]
        norm_series = (df["Close"] / base_val) * 100.0
        ax.plot(dates, norm_series, label=name, color=colors.get(name, "black"), lw=1.8, alpha=0.9)

    ax.set_title("Multi-Index Normalized Price Trajectories (2010–2017, Base = 100)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Index Level (100 = Base)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_path = FIGURES_DIR / "fig_01_multi_index_normalized_trajectories.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_02_rolling_volatility(df_rep_b: pd.DataFrame, index_name: str = "S&P") -> Path:
    """2. 20-Day Annualized Rolling Volatility Clustering."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dates = pd.to_datetime(df_rep_b["Date"])
    ret = df_rep_b["Close"].pct_change().fillna(0.0)
    vol_20d = ret.rolling(20).std() * np.sqrt(252) * 100.0  # Annualized %

    ax.plot(dates, vol_20d, color="#d95f02", lw=1.5, label="20-Day Ann. Volatility (%)")
    ax.axhline(vol_20d.mean(), color="black", linestyle="--", lw=1.2, label=f"Mean Volatility ({vol_20d.mean():.1f}%)")
    ax.fill_between(dates, 0, vol_20d, color="#d95f02", alpha=0.25)

    ax.set_title(f"Volatility Clustering in {index_name} (20-Day Annualized Volatility)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_path = FIGURES_DIR / "fig_02_rolling_volatility_clustering.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_03_yield_curve_regimes(df_rep_b: pd.DataFrame) -> Path:
    """3. Yield Curve & Credit Spread Dynamics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    dates = pd.to_datetime(df_rep_b["Date"])

    # Term Spread (10Y - 3M)
    term_spread = df_rep_b["DGS10"] - df_rep_b["DTB3"]
    ax1.plot(dates, term_spread, color="#1b9e77", lw=1.5, label="Term Spread (10Y - 3M)")
    ax1.axhline(0, color="red", linestyle=":", lw=1.2, label="Inversion Level (0%)")
    ax1.set_ylabel("Term Spread (%)")
    ax1.set_title("Treasury Yield Curve & Corporate Credit Spread Regimes", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Credit Spread (Baa - 10Y)
    credit_spread = df_rep_b["DBAA"] - df_rep_b["DGS10"]
    ax2.plot(dates, credit_spread, color="#7570b3", lw=1.5, label="Credit Spread (Baa - 10Y)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Credit Spread (%)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    out_path = FIGURES_DIR / "fig_03_yield_curve_regimes.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_04_correlation_heatmap(df_rep_b: pd.DataFrame) -> Path:
    """4. Multi-Asset Spearman Rank Correlation Heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    # Stationary series only (returns / changes): correlating trending LEVELS such as Close
    # or DGS10 would produce spurious correlations.
    assets = ["mom", "ROC_5", "Volume", "Dollar index", "Gold", "Oil", "AAPL", "MSFT",
              "JPM", "XOM", "FTSE", "HSI", "DJI", "IXIC"]
    cols = [c for c in assets if c in df_rep_b.columns]
    corr_matrix = df_rep_b[cols].corr(method="spearman")

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman Correlation"},
        ax=ax,
    )
    ax.set_title("Spearman Rank Correlation Across Asset Classes", fontweight="bold")

    out_path = FIGURES_DIR / "fig_04_correlation_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_05_return_distribution_kurtosis(df_rep_a: pd.DataFrame, index_name: str = "S&P") -> Path:
    """5. Heavy Tails: Return Histogram vs Gaussian Fit and QQ-Plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ret = df_rep_a["Close"].pct_change().dropna()

    # Histogram & Gaussian PDF
    mu, sigma = ret.mean(), ret.std()
    x = np.linspace(ret.min(), ret.max(), 300)
    pdf = stats.norm.pdf(x, mu, sigma)

    ax1.hist(ret, bins=50, density=True, alpha=0.6, color="#4575b4", edgecolor="black", label="Empirical Returns")
    ax1.plot(x, pdf, "r-", lw=2, label="Gaussian Fit N(μ, σ²)")
    ax1.set_title(f"Empirical Returns vs Gaussian ({index_name})", fontweight="bold")
    ax1.set_xlabel("Daily Return")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # QQ-Plot
    stats.probplot(ret, dist="norm", plot=ax2)
    ax2.set_title("Normal Q-Q Plot (Fat Tail Evidence)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    out_path = FIGURES_DIR / "fig_05_return_distribution_kurtosis.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_06_maximum_drawdown(dfs: Dict[str, pd.DataFrame]) -> Path:
    """6. Underwater Drawdown Analysis Across Indices."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = {"DJI": "#1f77b4", "NASDAQ": "#ff7f0e", "NYSE": "#2ca02c", "RUSSELL": "#d62728", "S&P": "#9467bd"}

    for name in INDEX_NAMES:
        df = dfs[name]
        dates = pd.to_datetime(df["Date"])
        ret = df["Close"].pct_change().fillna(0.0)
        cum_ret = (1 + ret).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak * 100.0
        ax.plot(dates, drawdown, label=name, color=colors.get(name, "black"), lw=1.5, alpha=0.85)

    ax.set_title("Historical Underwater Drawdown Comparison (2010–2017)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown from Peak (%)")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_path = FIGURES_DIR / "fig_06_maximum_drawdown_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_07_pca_scree(df_features: pd.DataFrame) -> Path:
    """7. PCA Scree Plot & Cumulative Explained Variance."""
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    # PCA on STANDARDISED, STATIONARY features only (raw price/EMA/rate levels excluded:
    # unscaled levels such as Close (~2,000) would dominate every component).
    # Scaler and PCA are fitted on the 1,425 training rows only.
    from sklearn.preprocessing import StandardScaler
    from src.config import TRAIN_ROW_COUNT
    excluded = {"Close", "EMA_10", "EMA_20", "EMA_50", "EMA_200", "DTB4WK", "DTB3", "DTB6", "DGS5",
                "DGS10", "DAAA", "DBAA", "TE1", "TE2", "TE3", "TE5", "TE6", "DE1", "DE2", "DE4",
                "DE5", "DE6", "term_spread", "credit_spread"}
    numeric_df = df_features.select_dtypes(include=[np.number]).drop(columns=[c for c in excluded if c in df_features.columns])
    numeric_df = numeric_df.loc[:, numeric_df.iloc[:TRAIN_ROW_COUNT].std() > 0].fillna(0.0)
    Z = StandardScaler().fit(numeric_df.iloc[:TRAIN_ROW_COUNT]).transform(numeric_df.iloc[:TRAIN_ROW_COUNT])
    pca = PCA(n_components=min(30, Z.shape[1]))
    pca.fit(Z)
    var_ratio = pca.explained_variance_ratio_ * 100.0
    cum_var_ratio = np.cumsum(var_ratio)
    components = np.arange(1, len(var_ratio) + 1)

    ax1.bar(components, var_ratio, color="#4575b4", alpha=0.7, label="Individual Explained Variance (%)")
    ax2 = ax1.twinx()
    ax2.plot(components, cum_var_ratio, color="#d73027", marker="o", lw=2, label="Cumulative Explained Variance (%)")
    ax2.axhline(80, color="gray", linestyle="--", lw=1.2, label="80% Threshold")

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Individual Variance Ratio (%)", color="#4575b4")
    ax2.set_ylabel("Cumulative Variance Ratio (%)", color="#d73027")
    n80 = int(np.searchsorted(cum_var_ratio, 80.0) + 1)
    ax1.set_title(f"PCA Scree Plot ({Z.shape[1]} standardised stationary features; 80% needs {n80} PCs)", fontweight="bold")
    ax1.set_xticks(components[::2])
    ax1.grid(True, linestyle="--", alpha=0.6)

    out_path = FIGURES_DIR / "fig_07_pca_scree_explained_variance.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_08_dbscan_k_distance(distances: np.ndarray, optimal_eps: float, k: int = 9) -> Path:
    """8. k-Distance Graph with Automated Knee Locator."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sorted_dists = np.sort(distances)
    indices = np.arange(len(sorted_dists))

    ax.plot(indices, sorted_dists, color="#313695", lw=2, label=f"{k}-NN Distance")
    ax.axhline(optimal_eps, color="red", linestyle="--", lw=1.5, label=f"Optimal ε = {optimal_eps:.3f}")
    ax.set_title(f"DBSCAN Parameter Selection: {k}-Distance Graph Knee", fontweight="bold")
    ax.set_xlabel("Points Sorted by Distance")
    ax.set_ylabel(f"Distance to {k}-th Nearest Neighbor")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_path = FIGURES_DIR / "fig_08_dbscan_k_distance_knee.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_09_dbscan_clusters(
    X_pca: np.ndarray,
    labels: np.ndarray,
    dbcv_score: float,
    X_tsne: Optional[np.ndarray] = None,
) -> Path:
    """9. Market Regime Clusters in 2D Latent Feature Space (PCA vs t-SNE Comparison)."""
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", len(unique_labels))

    if X_tsne is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        # Panel 1: PCA 2D
        for idx, lbl in enumerate(unique_labels):
            mask = (labels == lbl)
            if lbl == -1:
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c="gray", marker="x", s=25, alpha=0.5, label="Noise / Stress Regime")
            else:
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[palette[idx]], s=40, alpha=0.8, label=f"Cluster {lbl}")
        ax1.set_title(f"(A) PCA Linear Projection (DBCV = {dbcv_score:.3f})", fontweight="bold")
        ax1.set_xlabel("Principal Component 1")
        ax1.set_ylabel("Principal Component 2")
        ax1.legend(loc="upper right", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Panel 2: t-SNE 2D Manifold
        for idx, lbl in enumerate(unique_labels):
            mask = (labels == lbl)
            if lbl == -1:
                ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c="gray", marker="x", s=25, alpha=0.5, label="Noise / Stress Regime")
            else:
                ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[palette[idx]], s=40, alpha=0.8, label=f"Cluster {lbl}")
        ax2.set_title("(B) t-SNE Non-Linear Manifold Projection", fontweight="bold")
        ax2.set_xlabel("t-SNE Dimension 1")
        ax2.set_ylabel("t-SNE Dimension 2")
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)
    else:
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        for idx, lbl in enumerate(unique_labels):
            mask = (labels == lbl)
            if lbl == -1:
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c="gray", marker="x", s=25, alpha=0.5, label="Noise / Stress Regime")
            else:
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[palette[idx]], s=40, alpha=0.8, label=f"Cluster {lbl}")
        ax1.set_title(f"DBSCAN Market Regime Clusters in Latent Space (DBCV = {dbcv_score:.3f})", fontweight="bold")
        ax1.set_xlabel("Principal Component 1")
        ax1.set_ylabel("Principal Component 2")
        ax1.legend(loc="upper right", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig_09_dbscan_regime_clusters_pca.png"
    fig.savefig(out_path)
    plt.close(fig)

    # Also save dedicated t-SNE figure if X_tsne is provided
    if X_tsne is not None:
        fig_tsne, ax_t = plt.subplots(figsize=(8.5, 5.5))
        for idx, lbl in enumerate(unique_labels):
            mask = (labels == lbl)
            if lbl == -1:
                ax_t.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c="gray", marker="x", s=25, alpha=0.5, label="Noise / Stress Regime")
            else:
                ax_t.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[palette[idx]], s=40, alpha=0.8, label=f"Cluster {lbl}")
        ax_t.set_title("t-SNE 2D Manifold Embedding of Financial Market Regimes", fontweight="bold")
        ax_t.set_xlabel("t-SNE Dimension 1")
        ax_t.set_ylabel("t-SNE Dimension 2")
        ax_t.legend(loc="upper right", frameon=True)
        ax_t.grid(True, linestyle="--", alpha=0.6)
        out_path_tsne = FIGURES_DIR / "fig_09b_tsne_manifold.png"
        fig_tsne.savefig(out_path_tsne)
        plt.close(fig_tsne)

    return out_path


def plot_10_ward_dendrogram(linkage_matrix: np.ndarray, p: int = 30) -> Path:
    """10. Truncated Ward HAC Dendrogram (p=30)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=p,
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=9,
        ax=ax,
    )
    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward Linkage, Truncated to p={p})", fontweight="bold")
    ax.set_xlabel("Cluster / Sample Index (Cluster Size)")
    ax.set_ylabel("Ward Cophenetic Distance")
    ax.grid(True, linestyle="--", alpha=0.6)

    out_path = FIGURES_DIR / "fig_10_ward_hac_dendrogram.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_11_association_rules(df_rules: pd.DataFrame) -> Path:
    """11. Association Rules: Support vs Confidence vs Lift (Conviction Colormap)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if len(df_rules) > 0 and "support" in df_rules.columns:
        sc = ax.scatter(
            df_rules["support"],
            df_rules["confidence"],
            c=df_rules.get("conviction", df_rules["lift"]),
            s=df_rules["lift"] * 40,
            cmap="viridis",
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Conviction Metric")
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_title("Discovered Association Rules (Bubble Size = Lift)", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No Association Rules Found Matching Filters", ha="center", va="center")

    ax.grid(True, linestyle="--", alpha=0.6)
    out_path = FIGURES_DIR / "fig_11_association_rules_support_confidence_lift.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_12_pyspark_strong_scaling(
    cores: List[int],
    speedups: List[float],
    amdahl_speedup: List[float],
    parallel_fraction: float,
) -> Path:
    """12. Distributed PySpark Strong Scaling Speedup & Amdahl's Law Asymptote."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(cores, speedups, "bo-", lw=2, label="Observed PySpark Speedup")
    ax.plot(cores, amdahl_speedup, "r--", lw=2, label=f"Amdahl's Law (f = {parallel_fraction:.2f})")
    ax.plot(cores, cores, "k:", lw=1.5, label="Ideal Linear Speedup")

    ax.set_title("PySpark Strong Scaling Efficiency & Theoretical Limit", fontweight="bold")
    ax.set_xlabel("Number of Cores / Partitions (p)")
    ax.set_ylabel("Speedup S(p)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_path = FIGURES_DIR / "fig_12_pyspark_strong_scaling_speedup.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_13_classification_confusion(conf_mats: Dict[str, np.ndarray]) -> Path:
    """13. Confusion matrices of the baseline and both Naive Bayes models on the 357 test days."""
    fig, axes = plt.subplots(1, len(conf_mats), figsize=(4.2 * len(conf_mats), 3.9))
    for ax, (name, cm) in zip(np.atleast_1d(axes), conf_mats.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred NotUp", "Pred Up"], yticklabels=["True NotUp", "True Up"])
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{name}  (accuracy = {acc:.3f})", fontweight="bold")
    fig.suptitle("Test-Period Confusion Matrices (N = 357, 2016-06-17 to 2017-11-14)", fontweight="bold")
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig_13_naive_bayes_confusion_matrices.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_14_local_vs_spark_scaling(records: List[Dict]) -> Path:
    """14. Local (mlxtend) vs PySpark FP-Growth latency as data volume grows."""
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df["N_Rows"], df["Local_mlxtend_Sec"], "o-", color="#d95f02", lw=2, label="Local Python (mlxtend fpgrowth)")
    ax.plot(df["N_Rows"], df["Spark_FPGrowth_Sec"], "s-", color="#1b9e77", lw=2, label="PySpark MLlib FPGrowth (local[*])")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of transactions (bootstrap + 5% item perturbation)")
    ax.set_ylabel("Steady-state fit time (s, log scale)")
    ax.set_title("Data Scaling: Local vs Spark FP-Growth (minsup = 0.10)", fontweight="bold")
    ax.legend(frameon=True); ax.grid(True, which="both", linestyle="--", alpha=0.5)
    out_path = FIGURES_DIR / "fig_14_local_vs_spark_data_scaling.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
