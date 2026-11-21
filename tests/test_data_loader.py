"""Unit tests for Data Loader module (REQ-01, REQ-10)."""
import pytest
import pandas as pd
from src.config import INDEX_NAMES, TABLES_DIR
from src.data_loader import verify_raw_datasets, load_raw_dataset, get_column_classification
from src.descriptive_analytics import compute_descriptive_statistics


def test_raw_file_integrity():
    """REQ-01: Verifies that all 5 raw CSV files exist and match SHA-256 hashes."""
    df_hashes = verify_raw_datasets()
    assert len(df_hashes) == 5
    assert df_hashes["Checksum_Match"].all()


def test_raw_dimensions_and_classification():
    """Verifies that each raw dataset has (1984, 84) shape and valid classification."""
    for name in INDEX_NAMES:
        df = load_raw_dataset(name)
        assert df.shape == (1984, 84)
        col_class = get_column_classification(df, name)
        assert "Date" in col_class["temporal"]
        assert "DGS10" in col_class["interest_rate_levels"]


def test_descriptive_analytics_skipna():
    """REQ-10: Verifies that descriptive analytics enforces skipna=True."""
    df_raw = load_raw_dataset("S&P")
    df_stats = compute_descriptive_statistics(df_raw, "S&P")
    assert len(df_stats) > 0
    assert "Mean" in df_stats.columns
    assert "Jarque_Bera_pVal" in df_stats.columns
    assert "Ljung_Box_Lag10_pVal" in df_stats.columns
