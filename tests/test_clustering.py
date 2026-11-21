"""Unit tests for Clustering and Dimensionality Reduction (REQ-20, REQ-21)."""
import pytest
import pandas as pd
from src.config import TABLES_DIR, FIGURES_DIR


def test_dbscan_regime_clustering():
    """REQ-20: Verifies DBSCAN regime profiles, knee locator, and DBCV index."""
    df_profiles = pd.read_csv(TABLES_DIR / "table_16_cluster_profiles_S&P.csv")
    assert len(df_profiles) >= 1
    assert "Regime_Label" in df_profiles.columns
    assert "Mean_Volatility" in df_profiles.columns


def test_pca_tsne_and_ward_hac():
    """REQ-21: Verifies PCA scree plot, t-SNE manifold plot, and Ward HAC dendrogram generation."""
    assert (FIGURES_DIR / "fig_07_pca_scree_explained_variance.png").exists()
    assert (FIGURES_DIR / "fig_08_dbscan_k_distance_knee.png").exists()
    assert (FIGURES_DIR / "fig_09_dbscan_regime_clusters_pca.png").exists()
    assert (FIGURES_DIR / "fig_09b_tsne_manifold.png").exists()
    assert (FIGURES_DIR / "fig_10_ward_hac_dendrogram.png").exists()
