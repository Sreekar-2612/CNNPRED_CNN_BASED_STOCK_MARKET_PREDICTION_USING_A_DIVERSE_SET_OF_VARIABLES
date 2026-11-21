"""Unit tests for Preprocessing module (REQ-02, REQ-03, REQ-04, REQ-05, REQ-06)."""
import pytest
import pandas as pd
import numpy as np
from src.config import (
    WARMUP_ROW_COUNT,
    POST_WARMUP_ROW_COUNT,
    LABELED_ROW_COUNT,
    TRAIN_ROW_COUNT,
    EMBARGO_ROW_COUNT,
    TEST_ROW_COUNT,
)
from src.data_loader import load_raw_dataset
from src.preprocessing import (
    clean_sentinels,
    truncate_warmup,
    build_representation_a,
    build_representation_b,
    create_chronological_splits,
)


def test_sentinel_conversion():
    """REQ-02: Verifies non-NaN sentinels are converted to np.nan."""
    df_raw = load_raw_dataset("S&P")
    df_clean, df_audit = clean_sentinels(df_raw)

    # Rate column sentinels
    # T-bill yields were genuinely 0.00% on some 2011-2015 days, so only long-maturity
    # and corporate yields (never legitimately zero) are checked here.
    rate_cols = ["DGS10", "DGS5", "DAAA", "DBAA"]
    for c in rate_cols:
        assert (df_clean[c] == 0.0).sum() == 0, f"Found 0.0 sentinel in rate column {c}"

    # Return sentinels <= -0.99
    commodity_cols = ["Gold", "Oil", "CTB3M"]
    for c in commodity_cols:
        if c in df_clean.columns:
            assert (df_clean[c] <= -0.99).sum() == 0, f"Found <= -0.99 sentinel in {c}"


def test_warmup_truncation():
    """REQ-03: Verifies warm-up truncation drops 200 rows leaving 1,784."""
    df_raw = load_raw_dataset("S&P")
    df_trunc = truncate_warmup(df_raw)
    assert len(df_trunc) == POST_WARMUP_ROW_COUNT
    assert df_trunc["Date"].iloc[0] == "2010-10-18"


def test_representation_a_properties():
    """REQ-04: Representation A has 1,784 rows and preserves NaNs."""
    df_raw = load_raw_dataset("S&P")
    df_rep_a = build_representation_a(df_raw, "S&P")
    assert len(df_rep_a) == POST_WARMUP_ROW_COUNT
    assert df_rep_a.isna().sum().sum() > 0, "Rep A must preserve missing values as NaNs"


def test_representation_b_properties():
    """REQ-05: Representation B has 1,784 rows and zero NaNs."""
    df_raw = load_raw_dataset("S&P")
    df_rep_b = build_representation_b(df_raw, "S&P")
    assert len(df_rep_b) == POST_WARMUP_ROW_COUNT
    assert df_rep_b.isna().sum().sum() == 0, "Rep B must have zero NaNs"


def test_chronological_embargo_split():
    """REQ-06: Verifies 1,425 train, 1 embargo, 357 test = 1,783 labeled rows."""
    df_raw = load_raw_dataset("S&P")
    df_rep_b = build_representation_b(df_raw, "S&P")
    df_train, df_embargo, df_test = create_chronological_splits(df_rep_b)

    assert len(df_train) == TRAIN_ROW_COUNT
    assert len(df_embargo) == EMBARGO_ROW_COUNT
    assert len(df_test) == TEST_ROW_COUNT
    assert len(df_train) + len(df_embargo) + len(df_test) == LABELED_ROW_COUNT

    # Invariant dates
    assert str(df_train["Date"].iloc[0]) == "2010-10-18"
    assert str(df_train["Date"].iloc[-1]) == "2016-06-15"
    assert str(df_embargo["Date"].iloc[0]) == "2016-06-16"
    assert str(df_test["Date"].iloc[0]) == "2016-06-17"
    assert str(df_test["Date"].iloc[-1]) == "2017-11-14"
