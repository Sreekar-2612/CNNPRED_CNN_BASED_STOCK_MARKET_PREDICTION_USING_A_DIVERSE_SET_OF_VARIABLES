"""
Feature Engineering and Transaction Discretization Module.
Implements:
1. Stationarity transformations (relative spreads, distance to EMA, normalized returns).
2. ADF Stationarity validation on modeling features.
3. Quantile threshold fitting on TRAINING set only (preventing leakage).
4. Strict 24-item transaction generation:
   - Representation C1: Contemporaneous transactions (N = 1,784, 22 tokens).
   - Representation C2: Predictive transactions (N = 1,783, 24 tokens).
5. Target validation and distribution tracking.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.config import (
    CONTEMPORANEOUS_ITEMS,
    PREDICTIVE_TARGET_ITEMS,
    ALL_TRANSACTION_ITEMS,
    TRAIN_ROW_COUNT,
    LABELED_ROW_COUNT,
    POST_WARMUP_ROW_COUNT,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)


def compute_stationary_features(df_modeling: pd.DataFrame, target_index: str) -> pd.DataFrame:
    """
    Computes stationary continuous features for machine learning and clustering:
    - Index return: r_t = (Close_t - Close_{t-1}) / Close_{t-1}
    - Distance to EMA: dist_EMA50 = (Close_t - EMA50_t) / EMA50_t
    - Volatility proxy: 20-day rolling std of returns (using only past data)
    - Term spread: DGS10 - DTB3
    - Credit spread: DBAA - DGS10
    - Changes in spreads: delta_term_spread, delta_credit_spread
    """
    df = df_modeling.copy()

    # Index return
    # Index return = dataset column `mom` (already (C_t - C_{t-1}) / C_{t-1}); defined for row 0 too
    df["return_idx"] = df["mom"].astype(float) if "mom" in df.columns else df["Close"].pct_change().fillna(0.0)

    # Normalized distance to EMA50
    ema50_col = "EMA_50" if "EMA_50" in df.columns else ("EMA50" if "EMA50" in df.columns else None)
    if ema50_col is not None:
        df["dist_EMA50"] = (df["Close"] - df[ema50_col]) / df[ema50_col]
    else:
        df["dist_EMA50"] = 0.0

    # 20-day rolling volatility proxy (past data only)
    # Causal 20-day rolling volatility; the first rows use an expanding window (past data only)
    roll = df["return_idx"].rolling(window=20, min_periods=20).std()
    expd = df["return_idx"].expanding(min_periods=2).std()
    df["volatility_20"] = roll.fillna(expd).fillna(0.0)

    # Yield & Credit spreads
    if "DGS10" in df.columns and "DTB3" in df.columns:
        df["term_spread"] = df["DGS10"] - df["DTB3"]
        df["delta_term_spread"] = df["term_spread"].diff().fillna(0.0)
    else:
        df["term_spread"] = 0.0
        df["delta_term_spread"] = 0.0

    if "DBAA" in df.columns and "DGS10" in df.columns:
        df["credit_spread"] = df["DBAA"] - df["DGS10"]
        df["delta_credit_spread"] = df["credit_spread"].diff().fillna(0.0)
    else:
        df["credit_spread"] = 0.0
        df["delta_credit_spread"] = 0.0

    return df


def verify_feature_stationarity(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Augmented Dickey-Fuller (ADF) test on key modeling features to confirm stationarity.
    Returns summary DataFrame with test statistic, p-value, and stationarity status.
    """
    test_cols = [
        "return_idx",
        "dist_EMA50",
        "volatility_20",
        "term_spread",
        "delta_term_spread",
        "credit_spread",
        "delta_credit_spread",
        "Volume",
    ]
    records = []
    for col in test_cols:
        if col in df_features.columns:
            series = df_features[col].dropna()
            if series.nunique() <= 1 or series.std() < 1e-8:
                continue
            result = adfuller(series, autolag="AIC")
            records.append({
                "Feature": col,
                "ADF_Statistic": round(result[0], 4),
                "p_value": round(result[1], 6),
                "Used_Lag": result[2],
                "N_Obs": result[3],
                "Critical_1%": round(result[4]["1%"], 4),
                "Critical_5%": round(result[4]["5%"], 4),
                "Is_Stationary_5%": result[1] < 0.05,
            })

    df_adf = pd.DataFrame(records)
    df_adf.to_csv(TABLES_DIR / "table_06_adf_stationarity.csv", index=False)
    return df_adf


def fit_transaction_thresholds(df_train: pd.DataFrame) -> Dict[str, float]:
    """
    Fits quantile discretization thresholds strictly on the TRAINING set (1,425 rows).
    Never uses embargo or test rows to fit quantiles.
    """
    thresholds = {}

    # Volume Jump threshold: 75th percentile of Volume
    thresholds["volume_q75"] = float(df_train["Volume"].quantile(0.75))

    # Term Spread quantiles
    if "DGS10" in df_train.columns and "DTB3" in df_train.columns:
        ts = df_train["DGS10"] - df_train["DTB3"]
        thresholds["term_spread_q66"] = float(ts.quantile(0.66))
        thresholds["term_spread_q33"] = float(ts.quantile(0.33))
    else:
        thresholds["term_spread_q66"] = 0.0
        thresholds["term_spread_q33"] = 0.0

    # Credit Spread quantiles
    if "DBAA" in df_train.columns and "DGS10" in df_train.columns:
        cs = df_train["DBAA"] - df_train["DGS10"]
        thresholds["credit_spread_q66"] = float(cs.quantile(0.66))
        thresholds["credit_spread_q33"] = float(cs.quantile(0.33))
    else:
        thresholds["credit_spread_q66"] = 0.0
        thresholds["credit_spread_q33"] = 0.0

    return thresholds


def discretize_transactions(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    target_index: str,
    include_target: bool = True,
) -> Tuple[List[List[str]], pd.DataFrame]:
    """
    Discretizes a financial DataFrame into transactions according to the 24-item vocabulary.
    - If include_target is False: Produces Representation C1 (22 contemporaneous tokens, N = len(df)).
    - If include_target is True: Produces Representation C2 (24 tokens, N = len(df) - 1 if df has final unobserved row).
    Returns:
        (transactions_list, binary_dataframe)
    """
    transactions = []
    binary_records = []

    # Identify stock peer columns
    tech_cols = [c for c in ["AAPL", "MSFT"] if c in df.columns]
    europe_cols = [c for c in ["FTSE", "DAX-F"] if c in df.columns]
    asia_cols = [c for c in ["HSI", "Nikkei-F"] if c in df.columns]

    n_rows = len(df)
    limit = (n_rows - 1) if (include_target and n_rows >= POST_WARMUP_ROW_COUNT) else n_rows

    for t in range(limit):
        items = []
        row = df.iloc[t]
        close = row["Close"]

        # 1-2. SameDay Direction: sign of the dataset's own 1-day return `mom`
        # (defined for every post-warm-up row). A zero return produces no item.
        mom = row.get("mom", np.nan)
        if pd.notna(mom) and mom > 0:
            items.append("SameDay_Up")
        elif pd.notna(mom) and mom < 0:
            items.append("SameDay_Down")

        # 3. Volume Jump
        if row["Volume"] > thresholds["volume_q75"]:
            items.append("Volume_Jump")

        # 4-5. Short-term Momentum (mom5)
        mom5 = row.get("ROC_5", np.nan)  # 5-day rate of change (%), column ROC_5
        if pd.notna(mom5):
            if mom5 > 0:
                items.append("Mom5_Pos")
            elif mom5 < 0:
                items.append("Mom5_Neg")

        # 6-7. EMA50 Filter
        ema50 = row.get("EMA_50", row.get("EMA50", np.nan))
        if pd.notna(ema50):
            if close > ema50:
                items.append("Above_EMA50")
            else:
                items.append("Below_EMA50")

        # 8-10. Term Spread
        if "DGS10" in row and "DTB3" in row and pd.notna(row["DGS10"]) and pd.notna(row["DTB3"]):
            ts_val = row["DGS10"] - row["DTB3"]
            if ts_val > thresholds["term_spread_q66"]:
                items.append("TermSpread_Steep")
            elif ts_val <= thresholds["term_spread_q33"]:
                items.append("TermSpread_Flat")

            if t > 0 and pd.notna(df.iloc[t - 1]["DGS10"]) and pd.notna(df.iloc[t - 1]["DTB3"]):
                prev_ts = df.iloc[t - 1]["DGS10"] - df.iloc[t - 1]["DTB3"]
                if (ts_val - prev_ts) > 0:
                    items.append("TermSpread_Widening")

        # 11-13. Credit Spread
        if "DBAA" in row and "DGS10" in row and pd.notna(row["DBAA"]) and pd.notna(row["DGS10"]):
            cs_val = row["DBAA"] - row["DGS10"]
            if cs_val > thresholds["credit_spread_q66"]:
                items.append("CreditSpread_High")
            elif cs_val <= thresholds["credit_spread_q33"]:
                items.append("CreditSpread_Low")

            if t > 0 and pd.notna(df.iloc[t - 1]["DBAA"]) and pd.notna(df.iloc[t - 1]["DGS10"]):
                prev_cs = df.iloc[t - 1]["DBAA"] - df.iloc[t - 1]["DGS10"]
                if (cs_val - prev_cs) > 0:
                    items.append("CreditSpread_Widening")

        # 14-15. USD Direction
        # Daily return; zero (no change / closed market) and NaN produce no item
        usd = row.get("Dollar index", np.nan)
        if pd.notna(usd):
            if usd > 0:
                items.append("USD_Up")
            elif usd < 0:
                items.append("USD_Down")

        # 16-17. Gold Direction
        # Daily return; zero (no change / closed market) and NaN produce no item
        gold = row.get("Gold", np.nan)
        if pd.notna(gold):
            if gold > 0:
                items.append("Gold_Up")
            elif gold < 0:
                items.append("Gold_Down")

        # 18-19. Oil Direction
        # Daily return; zero (no change / closed market) and NaN produce no item
        oil = row.get("Oil", np.nan)
        if pd.notna(oil):
            if oil > 0:
                items.append("Oil_Up")
            elif oil < 0:
                items.append("Oil_Down")

        # 20. Tech Leaders
        if tech_cols:
            tech_vals = [row[c] for c in tech_cols]
            if len(tech_cols) == 2 and all(pd.notna(v) and v > 0 for v in tech_vals):
                items.append("Tech_Leaders_Up")

        # 21. Europe Direction
        if europe_cols:
            eu_vals = [row[c] for c in europe_cols]
            if len(europe_cols) == 2 and all(pd.notna(v) and v > 0 for v in eu_vals):
                items.append("Europe_Up")

        # 22. Asia Direction
        if asia_cols:
            as_vals = [row[c] for c in asia_cols]
            if len(asia_cols) == 2 and all(pd.notna(v) and v > 0 for v in as_vals):
                items.append("Asia_Up")

        # 23-24. Predictive Forward Target (if requested and t < n_rows - 1)
        if include_target:
            if t < n_rows - 1:
                next_close = df.iloc[t + 1]["Close"]
                if next_close > close:
                    items.append("NextDay_Up")
                else:
                    items.append("NextDay_NotUp")

        transactions.append(items)

        # Build binary indicator row
        vocab = ALL_TRANSACTION_ITEMS if include_target else CONTEMPORANEOUS_ITEMS
        bin_row = {item: (1 if item in items else 0) for item in vocab}
        bin_row["Date"] = row["Date"]
        binary_records.append(bin_row)

    df_binary = pd.DataFrame(binary_records)
    return transactions, df_binary


def execute_feature_engineering_pipeline(
    df_rep_a: pd.DataFrame,
    df_rep_b: pd.DataFrame,
    target_index: str,
) -> Tuple[pd.DataFrame, List[List[str]], List[List[str]]]:
    """
    Executes feature engineering and transaction generation for an index.
    Saves:
    - ADF stationarity table
    - Binary transaction matrices
    - JSON transaction lists
    Returns:
        (df_features, contemporaneous_transactions, predictive_transactions)
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Compute continuous stationary features on Rep B
    df_features = compute_stationary_features(df_rep_b, target_index)
    verify_feature_stationarity(df_features)

    # 2. Fit thresholds strictly on training partition (Rep A: sentinel days stay NaN and are ignored)
    df_train = df_rep_a.iloc[:TRAIN_ROW_COUNT]
    thresholds = fit_transaction_thresholds(df_train)

    # 3. Discretize Representation C1 (Contemporaneous: N = 1,784, 22 tokens)
    tx_c1, df_bin_c1 = discretize_transactions(df_rep_a, thresholds, target_index, include_target=False)

    # 4. Discretize Representation C2 (Predictive: N = 1,783, 24 tokens)
    tx_c2, df_bin_c2 = discretize_transactions(df_rep_a, thresholds, target_index, include_target=True)

    # Save transactions
    with open(PROCESSED_DATA_DIR / f"transactions_contemporaneous_{target_index}.json", "w") as f:
        json.dump(tx_c1, f)
    with open(PROCESSED_DATA_DIR / f"transactions_predictive_{target_index}.json", "w") as f:
        json.dump(tx_c2, f)

    df_bin_c1.to_parquet(PROCESSED_DATA_DIR / f"binary_c1_{target_index}.parquet", index=False)
    df_bin_c2.to_parquet(PROCESSED_DATA_DIR / f"binary_c2_{target_index}.parquet", index=False)
    df_features.to_parquet(PROCESSED_DATA_DIR / f"df_features_{target_index}.parquet", index=False)

    # Validate target distribution for test split
    y_test_counts = df_bin_c2.iloc[TRAIN_ROW_COUNT + 1 :]["NextDay_Up"].value_counts()
    up_count = y_test_counts.get(1, 0)
    not_up_count = y_test_counts.get(0, 0)
    total_test = up_count + not_up_count
    up_ratio = up_count / total_test if total_test > 0 else 0.0

    target_dist = pd.DataFrame([{
        "Index": target_index,
        "Total_Labeled_Rows": len(df_bin_c2),
        "Test_Rows": total_test,
        "Test_NextDay_Up": int(up_count),
        "Test_NextDay_NotUp": int(not_up_count),
        "Test_Up_Ratio": round(up_ratio, 6),
        "Expected_Majority_Baseline": "55.46%",
    }])
    target_dist.to_csv(TABLES_DIR / f"table_05_target_distribution_{target_index}.csv", index=False)

    return df_features, tx_c1, tx_c2
