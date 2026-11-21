"""
Data Preprocessing Module for CNNpred Big Data Pipeline.
Handles:
1. Sentinel detection and elimination (DGS10 == 0.0, -1.0 return sentinels -> NaN).
2. Warm-up truncation (t < 200 removed, leaving 1,784 observations).
3. Creation of Representation A (Descriptive Analytics: NaNs preserved).
4. Creation of Representation B (Continuous Modeling: ffill for levels, 0.0 for returns).
5. Chronological 80/20 train/test partition with 1-day boundary embargo (1,425 + 1 + 357 = 1,783).
"""
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import (
    WARMUP_ROW_COUNT,
    POST_WARMUP_ROW_COUNT,
    LABELED_ROW_COUNT,
    TRAIN_ROW_COUNT,
    EMBARGO_ROW_COUNT,
    TEST_ROW_COUNT,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)
from src.data_loader import load_raw_dataset, get_column_classification


def clean_sentinels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies and replaces non-NaN sentinels with np.nan:
    - US interest rates ('DGS10', 'DTB3', 'DTB6', 'DTB4WK', 'DBAA'): 0.0 replaced with NaN.
    - Foreign returns & commodities ('Gold', 'Oil', 'CTB3M', 'CTB6M', 'CTB1Y'): <= -0.99 replaced with NaN.
    Returns:
        (df_cleaned, df_sentinel_audit)
    """
    df_clean = df.copy()
    sentinel_records = []

    # 1. Rate block sentinels (0.0): on US bond-market holidays the whole rate block
    #    (and every spread derived from it) is recorded as 0.0. Detect closure days via
    #    DGS10 == 0 and blank the complete block, including DGS5, DAAA, TE* and DE*.
    rate_cols = ["DGS10", "DTB3", "DTB6", "DTB4WK", "DBAA", "DGS5", "DAAA",
                 "TE1", "TE2", "TE3", "TE5", "TE6", "DE1", "DE2", "DE4", "DE5", "DE6"]
    closure = (df_clean["DGS10"] == 0.0) if "DGS10" in df_clean.columns else pd.Series(False, index=df_clean.index)
    # masks computed from the ORIGINAL values, before any column is blanked
    corp_zero = (df_clean["DAAA"] == 0.0) | (df_clean["DBAA"] == 0.0)
    dgs5_zero = df_clean["DGS5"] == 0.0
    for col in rate_cols:
        if col in df_clean.columns:
            # genuine 0.00% T-bill yields (2011-2015) are NOT sentinels; long-maturity and
            # corporate yields are never legitimately 0, so a zero there is a missing value
            # that also invalidates every spread built from it.
            mask = closure.copy()
            if col == "DGS5":
                mask |= dgs5_zero
            if col in ("DAAA", "DBAA") or col.startswith("DE"):
                mask |= corp_zero
            count = mask.sum()
            if count > 0:
                dates = df_clean.loc[mask, "Date"].tolist()
                for d in dates:
                    sentinel_records.append({
                        "Column": col,
                        "Sentinel_Type": "Zero_Rate",
                        "Sentinel_Value": 0.0,
                        "Date": d,
                    })
                df_clean.loc[mask, col] = np.nan

    # 2. Return & commodity sentinels (<= -0.99)
    commodity_and_foreign_cols = ["Gold", "Oil", "CTB3M", "CTB6M", "CTB1Y"]
    for col in commodity_and_foreign_cols:
        if col in df_clean.columns:
            mask = (df_clean[col] <= -0.99)
            count = mask.sum()
            if count > 0:
                dates = df_clean.loc[mask, "Date"].tolist()
                for d in dates:
                    sentinel_records.append({
                        "Column": col,
                        "Sentinel_Type": "Holiday_Negative_Sentinel",
                        "Sentinel_Value": -1.0,
                        "Date": d,
                    })
                df_clean.loc[mask, col] = np.nan

    df_audit = pd.DataFrame(sentinel_records)
    return df_clean, df_audit


def truncate_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Truncates the first 200 warm-up rows (indices 0 to 199).
    Returns 1,784 clean rows with reset index (0 to 1,783).
    """
    if len(df) < WARMUP_ROW_COUNT:
        raise ValueError(f"Dataset length {len(df)} is less than warm-up count {WARMUP_ROW_COUNT}")
    df_truncated = df.iloc[WARMUP_ROW_COUNT:].copy().reset_index(drop=True)
    if len(df_truncated) != POST_WARMUP_ROW_COUNT:
        raise ValueError(f"Expected {POST_WARMUP_ROW_COUNT} post-warmup rows, got {len(df_truncated)}")
    return df_truncated


def build_representation_a(df_raw: pd.DataFrame, target_index: str) -> pd.DataFrame:
    """
    Representation A: Descriptive Analytics Dataset.
    - Warmup truncated (N = 1,784).
    - Sentinels converted to np.nan.
    - Missing returns preserved as NaN.
    """
    df_clean, _ = clean_sentinels(df_raw)
    df_rep_a = truncate_warmup(df_clean)
    return df_rep_a


def build_representation_b(df_raw: pd.DataFrame, target_index: str) -> pd.DataFrame:
    """
    Representation B: Continuous Modeling Dataset.
    - Warmup truncated (N = 1,784).
    - Sentinels converted to np.nan.
    - Levels (price and rates) forward-filled (then bfill for any leading).
    - Stationary returns and relative spreads imputed with 0.0 (neutral market).
    - Zero NaNs guaranteed.
    """
    df_clean, _ = clean_sentinels(df_raw)
    df_post = truncate_warmup(df_clean)
    col_class = get_column_classification(df_post, target_index)

    df_rep_b = df_post.copy()

    # Forward fill levels
    level_cols = col_class["price_levels"] + col_class["interest_rate_levels"]
    for col in level_cols:
        if col in df_rep_b.columns:
            df_rep_b[col] = df_rep_b[col].ffill().bfill()

    # Impute stationary returns and spreads with 0.0
    return_cols = (
        col_class["yield_spreads"]
        + col_class["canadian_rate_changes"]
        + col_class["commodities_currencies"]
        + col_class["volume_features"]
        + col_class["technical_indicators"]
        + col_class["peer_stock_returns"]
    )
    for col in return_cols:
        if col in df_rep_b.columns:
            df_rep_b[col] = df_rep_b[col].fillna(0.0)

    # Check that no NaNs remain (except potentially target on unobserved rows, but target is binary)
    nan_count = df_rep_b.isna().sum().sum()
    if nan_count > 0:
        # Check which column has NaN
        remaining_nans = df_rep_b.columns[df_rep_b.isna().any()].tolist()
        raise ValueError(f"Representation B contains {nan_count} NaNs in columns: {remaining_nans}")

    return df_rep_b


def create_chronological_splits(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the dataset chronologically into Train (1,425), Embargo (1), and Test (357).
    Total labeled rows = 1,783.
    Note: If df has 1,784 rows (post-warmup), the final row (index 1,783) is omitted
    from predictive evaluation since its forward target is unobserved.
    """
    if len(df) >= POST_WARMUP_ROW_COUNT:
        # Exclude the final row for labeled partitions
        df_labeled = df.iloc[:LABELED_ROW_COUNT].copy()
    elif len(df) == LABELED_ROW_COUNT:
        df_labeled = df.copy()
    else:
        raise ValueError(f"Unexpected dataframe length {len(df)} for partitioning.")

    df_train = df_labeled.iloc[:TRAIN_ROW_COUNT].copy()
    df_embargo = df_labeled.iloc[TRAIN_ROW_COUNT : TRAIN_ROW_COUNT + EMBARGO_ROW_COUNT].copy()
    df_test = df_labeled.iloc[TRAIN_ROW_COUNT + EMBARGO_ROW_COUNT :].copy()

    assert len(df_train) == TRAIN_ROW_COUNT, f"Expected {TRAIN_ROW_COUNT} train rows, got {len(df_train)}"
    assert len(df_embargo) == EMBARGO_ROW_COUNT, f"Expected {EMBARGO_ROW_COUNT} embargo rows, got {len(df_embargo)}"
    assert len(df_test) == TEST_ROW_COUNT, f"Expected {TEST_ROW_COUNT} test rows, got {len(df_test)}"
    assert len(df_train) + len(df_embargo) + len(df_test) == LABELED_ROW_COUNT

    return df_train, df_embargo, df_test


def execute_preprocessing_pipeline() -> Dict[str, pd.DataFrame]:
    """
    Executes full preprocessing pipeline across all 5 indices.
    Saves:
    - Sentinel audit to outputs/tables/table_02_sentinel_audit.csv
    - Truncation summary to outputs/tables/table_03_truncation_summary.csv
    - Split partitions to outputs/tables/table_04_split_partitions.csv
    - Processed DataFrames to data/processed/ (both Parquet and CSV)
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_audits = []
    trunc_records = []
    split_records = []
    processed_dfs = {}

    from src.config import INDEX_NAMES

    for name in INDEX_NAMES:
        df_raw = load_raw_dataset(name)
        df_clean, df_audit = clean_sentinels(df_raw)
        df_audit["Index"] = name
        all_audits.append(df_audit)

        df_rep_a = build_representation_a(df_raw, name)
        df_rep_b = build_representation_b(df_raw, name)

        trunc_records.append({
            "Index": name,
            "Raw_Rows": len(df_raw),
            "Warmup_Truncated": WARMUP_ROW_COUNT,
            "Rep_A_Rows": len(df_rep_a),
            "Rep_A_NaNs": int(df_rep_a.isna().sum().sum()),
            "Rep_B_Rows": len(df_rep_b),
            "Rep_B_NaNs": int(df_rep_b.isna().sum().sum()),
            "Rep_A_Start_Date": str(df_rep_a["Date"].iloc[0]),
            "Rep_A_End_Date": str(df_rep_a["Date"].iloc[-1]),
        })

        train_b, emb_b, test_b = create_chronological_splits(df_rep_b)
        split_records.append({
            "Index": name,
            "Train_Rows": len(train_b),
            "Train_Start": str(train_b["Date"].iloc[0]),
            "Train_End": str(train_b["Date"].iloc[-1]),
            "Embargo_Rows": len(emb_b),
            "Embargo_Date": str(emb_b["Date"].iloc[0]),
            "Test_Rows": len(test_b),
            "Test_Start": str(test_b["Date"].iloc[0]),
            "Test_End": str(test_b["Date"].iloc[-1]),
        })

        # Save to parquet and csv
        df_rep_a.to_parquet(PROCESSED_DATA_DIR / f"df_descriptive_{name}.parquet", index=False)
        df_rep_a.to_csv(PROCESSED_DATA_DIR / f"df_descriptive_{name}.csv", index=False)
        df_rep_b.to_parquet(PROCESSED_DATA_DIR / f"df_modeling_{name}.parquet", index=False)
        df_rep_b.to_csv(PROCESSED_DATA_DIR / f"df_modeling_{name}.csv", index=False)

        processed_dfs[f"{name}_rep_a"] = df_rep_a
        processed_dfs[f"{name}_rep_b"] = df_rep_b

    # Combine audits and save
    if all_audits:
        df_full_audit = pd.concat(all_audits, ignore_index=True)
        df_full_audit.to_csv(TABLES_DIR / "table_02_sentinel_audit.csv", index=False)

    df_trunc_summary = pd.DataFrame(trunc_records)
    df_trunc_summary.to_csv(TABLES_DIR / "table_03_truncation_summary.csv", index=False)

    df_split_summary = pd.DataFrame(split_records)
    df_split_summary.to_csv(TABLES_DIR / "table_04_split_partitions.csv", index=False)

    return processed_dfs
