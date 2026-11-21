"""Unit tests for Visualization Suite and Academic PDF Report (REQ-11, REQ-25)."""
import pytest
from src.config import FIGURES_DIR, REPORTS_DIR


def test_visualization_files_exist():
    """REQ-11: Verifies all 12 publication-quality figure files exist."""
    expected_figures = [
        "fig_01_multi_index_normalized_trajectories.png",
        "fig_02_rolling_volatility_clustering.png",
        "fig_03_yield_curve_regimes.png",
        "fig_04_correlation_heatmap.png",
        "fig_05_return_distribution_kurtosis.png",
        "fig_06_maximum_drawdown_comparison.png",
        "fig_07_pca_scree_explained_variance.png",
        "fig_08_dbscan_k_distance_knee.png",
        "fig_09_dbscan_regime_clusters_pca.png",
        "fig_09b_tsne_manifold.png",
        "fig_10_ward_hac_dendrogram.png",
        "fig_11_association_rules_support_confidence_lift.png",
        "fig_12_pyspark_strong_scaling_speedup.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 10000, f"Figure file {fig_name} is too small"


def test_academic_pdf_report_exists():
    """REQ-25: Verifies that the academic PDF report exists and is > 2 MB."""
    report_path = REPORTS_DIR / "Analytics_BigData_Endsem_Report.pdf"
    assert report_path.exists()
    size_mb = report_path.stat().st_size / (1024 * 1024)
    assert size_mb >= 2.0, f"Report size {size_mb:.2f} MB is less than required 2 MB"
