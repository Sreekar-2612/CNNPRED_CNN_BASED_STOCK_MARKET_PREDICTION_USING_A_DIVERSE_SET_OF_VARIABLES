"""
Descriptive Analytics Module for CNNpred Big Data Pipeline.
Calculates statistical moments, distributional properties, and time series tests
on Representation A (with sentinels converted to NaN, preserving missing values via skipna=True).
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.config import TABLES_DIR, PROCESSED_DATA_DIR, INDEX_NAMES


def compute_descriptive_statistics(df_rep_a: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """
    Computes rigorous descriptive analytics for a specified index dataset:
    - Sample size (total rows, valid non-null rows, null count, null %)
    - Statistical moments: Mean, Standard Deviation, Variance, Skewness, Kurtosis (Fisher-Pearson)
    - Percentiles: Min, 25%, Median (50%), 75%, Max
    - Normality: Jarque-Bera test statistic and p-value
    - Autocorrelation: Ljung-Box test statistic (lag 10) and p-value
    All calculations explicitly enforce skipna=True.
    """
    # Select representative columns across asset classes
    candidate_cols = [
        "Close", "Volume", "mom5", "RSI_14", "DGS10", "DTB3", "DBAA",
        "TE1", "DE1", "USD", "Gold", "Oil", "CTB3M", "AAPL", "FTSE", "N225"
    ]
    cols = [c for c in candidate_cols if c in df_rep_a.columns]

    records = []
    total_n = len(df_rep_a)

    for col in cols:
        series = df_rep_a[col].dropna()
        valid_n = len(series)
        missing_n = total_n - valid_n
        missing_pct = (missing_n / total_n) * 100.0

        if valid_n < 10:
            continue

        mean_val = float(series.mean())
        std_val = float(series.std(ddof=1))
        var_val = float(series.var(ddof=1))
        skew_val = float(stats.skew(series, bias=False))
        kurt_val = float(stats.kurtosis(series, bias=False))  # Excess kurtosis

        min_val = float(series.min())
        q25_val = float(series.quantile(0.25))
        median_val = float(series.median())
        q75_val = float(series.quantile(0.75))
        max_val = float(series.max())

        # Jarque-Bera test for normality
        jb_stat, jb_pval = stats.jarque_bera(series)

        # Ljung-Box test for serial autocorrelation (lag 10)
        try:
            lb_res = acorr_ljungbox(series, lags=[10], return_df=True)
            lb_stat = float(lb_res["lb_stat"].iloc[0])
            lb_pval = float(lb_res["lb_pvalue"].iloc[0])
        except Exception:
            lb_stat, lb_pval = np.nan, np.nan

        records.append({
            "Index": index_name,
            "Feature": col,
            "Total_N": total_n,
            "Valid_N": valid_n,
            "Missing_Count": missing_n,
            "Missing_Pct": round(missing_pct, 2),
            "Mean": round(mean_val, 4),
            "Std_Dev": round(std_val, 4),
            "Variance": round(var_val, 4),
            "Skewness": round(skew_val, 4),
            "Excess_Kurtosis": round(kurt_val, 4),
            "Min": round(min_val, 4),
            "Q25": round(q25_val, 4),
            "Median": round(median_val, 4),
            "Q75": round(q75_val, 4),
            "Max": round(max_val, 4),
            "Jarque_Bera_Stat": round(jb_stat, 2),
            "Jarque_Bera_pVal": round(jb_pval, 6),
            "Ljung_Box_Lag10_Stat": round(lb_stat, 2) if not np.isnan(lb_stat) else np.nan,
            "Ljung_Box_Lag10_pVal": round(lb_pval, 6) if not np.isnan(lb_pval) else np.nan,
        })

    return pd.DataFrame(records)


def execute_descriptive_pipeline() -> pd.DataFrame:
    """
    Runs descriptive analytics across all 5 index datasets on Representation A.
    Consolidates results and writes to outputs/tables/table_07_descriptive_stats.csv.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []

    for name in INDEX_NAMES:
        path = PROCESSED_DATA_DIR / f"df_descriptive_{name}.parquet"
        if not path.exists():
            # Fallback to CSV if parquet not yet generated
            path = PROCESSED_DATA_DIR / f"df_descriptive_{name}.csv"
        df_rep_a = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
        df_stats = compute_descriptive_statistics(df_rep_a, name)
        dfs.append(df_stats)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(TABLES_DIR / "table_07_descriptive_stats.csv", index=False)
    return df_all
