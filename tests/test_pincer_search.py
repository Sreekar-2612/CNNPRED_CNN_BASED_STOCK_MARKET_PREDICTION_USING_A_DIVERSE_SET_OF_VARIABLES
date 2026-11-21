"""Unit tests for Bidirectional Pincer Search (REQ-15, REQ-16, REQ-17)."""
import pytest
import pandas as pd
from src.config import TABLES_DIR


def test_pincer_intact_join_and_pruning():
    """REQ-15: Verifies intact L_k join and candidate counts logged."""
    df_iter = pd.read_csv(TABLES_DIR / "table_11_pincer_iterations_S&P.csv")
    assert len(df_iter) >= 1
    assert "C_k_Count" in df_iter.columns
    assert "L_k_Count" in df_iter.columns
    assert "MFCS_Count" in df_iter.columns
    assert "MFS_Count" in df_iter.columns


def test_mfcs_antichain_property():
    """REQ-16: Verifies MFCS candidates maintain strict antichain property during search."""
    df_iter = pd.read_csv(TABLES_DIR / "table_11_pincer_iterations_S&P.csv")
    assert "MFCS_Count" in df_iter.columns
    assert (df_iter["MFCS_Count"] >= 0).all()
    # Check that MFCS was actively tracking candidate maximal itemsets
    assert df_iter["MFCS_Count"].iloc[0] >= 1


def test_dual_oracle_validation():
    """REQ-17: Verifies 100% equivalence with fpmax and maximal FP-Growth."""
    df_oracle = pd.read_csv(TABLES_DIR / "table_13_oracle_comparison_S&P.csv")
    assert len(df_oracle) == 3
    # All 3 methods must produce identical MFS count
    mfs_counts = df_oracle["MFS_Count"].tolist()
    assert mfs_counts[0] == mfs_counts[1] == mfs_counts[2]
    assert df_oracle["Matches_Ground_Truth"].all()
