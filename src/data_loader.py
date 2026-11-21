"""
Data Loader Module for CNNpred Big Data Pipeline.
Handles raw dataset ingestion, SHA-256 verification, column classification,
and structural validation across all five index datasets.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import (
    RAW_CSV_FILES,
    EXPECTED_SHA256,
    RAW_ROW_COUNT,
    RAW_COL_COUNT,
    TABLES_DIR,
    INDEX_NAMES,
)


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower()


def verify_raw_datasets() -> pd.DataFrame:
    """
    Verifies that all five raw CSV files exist and match their authoritative SHA-256 checksums.
    Generates and saves the verification table to outputs/tables/table_01_dataset_hashes.csv.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    all_valid = True

    for name in INDEX_NAMES:
        path = RAW_CSV_FILES[name]
        exists = path.exists()
        file_size_bytes = path.stat().st_size if exists else 0
        actual_hash = compute_sha256(path) if exists else "MISSING"
        expected_hash = EXPECTED_SHA256[name].lower()
        match = (actual_hash == expected_hash)
        if not match:
            all_valid = False

        records.append({
            "Index": name,
            "File": path.name,
            "Size_Bytes": file_size_bytes,
            "Expected_SHA256": expected_hash,
            "Actual_SHA256": actual_hash,
            "Checksum_Match": match,
        })

    df_hashes = pd.DataFrame(records)
    out_path = TABLES_DIR / "table_01_dataset_hashes.csv"
    df_hashes.to_csv(out_path, index=False)
    if not all_valid:
        raise ValueError(f"Raw dataset checksum mismatch detected! See {out_path}")
    return df_hashes


def load_raw_dataset(index_name: str) -> pd.DataFrame:
    """
    Loads raw CSV dataset for the specified index and validates its dimensions.
    """
    if index_name not in RAW_CSV_FILES:
        raise ValueError(f"Unknown index '{index_name}'. Must be one of {list(RAW_CSV_FILES.keys())}")
    path = RAW_CSV_FILES[index_name]
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found at {path}")

    df = pd.read_csv(path)
    if df.shape != (RAW_ROW_COUNT, RAW_COL_COUNT):
        raise ValueError(
            f"Dataset {index_name} shape mismatch: expected ({RAW_ROW_COUNT}, {RAW_COL_COUNT}), got {df.shape}"
        )
    return df


def get_column_classification(df: pd.DataFrame, target_index: str) -> Dict[str, List[str]]:
    """
    Classifies all columns into financial and analytical categories:
    - temporal: ['Date']
    - target_label: [target_index]
    - price_levels: ['Close', 'EMA10', 'EMA20', 'EMA50', etc.]
    - interest_rate_levels: ['DGS10', 'DTB3', 'DTB6', 'DTB4WK', 'DBAA']
    - yield_spreads: ['TE1', 'TE2', 'TE3', 'TE5', 'TE6', 'DE1', 'DE2', 'DE4', 'DE5', 'DE6']
    - canadian_rate_changes: ['CTB3M', 'CTB6M', 'CTB1Y']
    - commodities_currencies: ['Gold', 'Oil', 'USD']
    - technical_indicators: momentum, ROC, MACD, RSI, etc.
    - volume_features: ['Volume']
    - peer_stock_returns: individual stock returns and foreign index returns
    """
    cols = list(df.columns)
    classification: Dict[str, List[str]] = {
        "temporal": ["Date"],
        "target_label": [target_index],
        "price_levels": [],
        "interest_rate_levels": ["DGS10", "DTB3", "DTB6", "DTB4WK", "DBAA", "DGS5", "DAAA",
                                 "TE1", "TE2", "TE3", "TE5", "TE6", "DE1", "DE2", "DE4", "DE5", "DE6"],
        "yield_spreads": [],  # spreads are rate LEVELS (forward-filled), listed above
        "canadian_rate_changes": ["CTB3M", "CTB6M", "CTB1Y"],
        "commodities_currencies": ["Gold", "Oil", "Dollar index"],
        "volume_features": ["Volume"],
        "technical_indicators": [],
        "peer_stock_returns": [],
    }

    known_sets = set(
        classification["temporal"]
        + classification["target_label"]
        + classification["interest_rate_levels"]
        + classification["yield_spreads"]
        + classification["canadian_rate_changes"]
        + classification["commodities_currencies"]
        + classification["volume_features"]
    )

    for col in cols:
        if col in known_sets:
            continue
        if col == "Close" or col.startswith("EMA") or col.startswith("SMA"):
            classification["price_levels"].append(col)
        elif any(col.startswith(p) for p in ["mom", "ROC", "RSI", "MACD", "Bollinger", "ATR", "CCI"]):
            classification["technical_indicators"].append(col)
        else:
            classification["peer_stock_returns"].append(col)

    return classification
