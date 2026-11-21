"""Unit tests for Feature Engineering and Transaction Discretization (REQ-07, REQ-08, REQ-09)."""
import pytest
import pandas as pd
from src.config import (
    CONTEMPORANEOUS_ITEMS,
    PREDICTIVE_TARGET_ITEMS,
    ALL_TRANSACTION_ITEMS,
    POST_WARMUP_ROW_COUNT,
    LABELED_ROW_COUNT,
    TABLES_DIR,
    PROCESSED_DATA_DIR,
)
from src.feature_engineering import (
    compute_stationary_features,
    verify_feature_stationarity,
    fit_transaction_thresholds,
    discretize_transactions,
)


def test_target_omission_and_alignment():
    """REQ-07: Verifies final row target omission and test majority baseline = 55.46%."""
    df_dist = pd.read_csv(TABLES_DIR / "table_05_target_distribution_S&P.csv")
    assert df_dist["Total_Labeled_Rows"].iloc[0] == LABELED_ROW_COUNT
    assert df_dist["Test_Rows"].iloc[0] == 357
    assert df_dist["Test_NextDay_Up"].iloc[0] == 198
    assert df_dist["Test_NextDay_NotUp"].iloc[0] == 159
    assert abs(df_dist["Test_Up_Ratio"].iloc[0] - 0.554622) < 1e-4


def test_stationarity_features():
    """REQ-08: Verifies stationary transformations and ADF p-values < 0.05."""
    df_adf = pd.read_csv(TABLES_DIR / "table_06_adf_stationarity.csv")
    stationary_features = ["return_idx", "dist_EMA50", "volatility_20", "delta_term_spread", "delta_credit_spread", "Volume"]
    for feat in stationary_features:
        row = df_adf[df_adf["Feature"] == feat]
        assert len(row) == 1, f"Missing feature {feat} in ADF table"
        assert bool(row["Is_Stationary_5%"].iloc[0]) is True


def test_transaction_vocabulary_exact():
    """REQ-09: Verifies exact 22 contemporaneous and 24 total tokens."""
    assert len(CONTEMPORANEOUS_ITEMS) == 22
    assert len(PREDICTIVE_TARGET_ITEMS) == 2
    assert len(ALL_TRANSACTION_ITEMS) == 24

    # Check binary matrices
    df_c1 = pd.read_parquet(PROCESSED_DATA_DIR / "binary_c1_S&P.parquet")
    df_c2 = pd.read_parquet(PROCESSED_DATA_DIR / "binary_c2_S&P.parquet")

    assert len(df_c1) == POST_WARMUP_ROW_COUNT
    assert len(df_c2) == LABELED_ROW_COUNT

    # Column tokens
    c1_cols = [c for c in df_c1.columns if c != "Date"]
    c2_cols = [c for c in df_c2.columns if c != "Date"]

    assert set(c1_cols) == set(CONTEMPORANEOUS_ITEMS)
    assert set(c2_cols) == set(ALL_TRANSACTION_ITEMS)
