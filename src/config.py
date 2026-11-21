"""
Central Configuration Module
Defines project paths, dataset metadata, constants, indexing invariants,
and transaction vocabulary for the CNNpred Big Data Analytics pipeline.
"""
import os
import sys
from pathlib import Path

# Enforce Python executable for PySpark worker processes on Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Windows + Java 17: the JVM's NIO selector creates a Unix-domain socket in the temp
# directory. If that path is long (AF_UNIX limit ~108 chars) Spark start-up fails with
# "Unable to establish loopback connection". Point the JVM at a short directory.
if sys.platform == "win32" and "jdk.net.unixdomain.tmpdir" not in os.environ.get("JAVA_TOOL_OPTIONS", ""):
    _short_tmp = Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "sparktmp"
    try:
        _short_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["JAVA_TOOL_OPTIONS"] = (os.environ.get("JAVA_TOOL_OPTIONS", "")
                                          + f" -Djdk.net.unixdomain.tmpdir={_short_tmp}").strip()
    except OSError:
        pass

# Workspace Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output Directories
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = ROOT_DIR / "reports"

# Dataset Identifiers & File Mapping
INDEX_NAMES = ["DJI", "NASDAQ", "NYSE", "RUSSELL", "S&P"]
RAW_CSV_FILES = {
    "DJI": RAW_DATA_DIR / "Processed_DJI.csv",
    "NASDAQ": RAW_DATA_DIR / "Processed_NASDAQ.csv",
    "NYSE": RAW_DATA_DIR / "Processed_NYSE.csv",
    "RUSSELL": RAW_DATA_DIR / "Processed_RUSSELL.csv",
    "S&P": RAW_DATA_DIR / "Processed_S&P.csv",
}

# Verified SHA-256 Hashes
EXPECTED_SHA256 = {
    "DJI": "6c0509451c784f117ce96e90b61b8e21a3f96a7dbc32f4a8427a146202c559eb",
    "NASDAQ": "fe5b09089d1e0c1f11aa66f512d6edc740489452cdd90aec48f760df90c51b01",
    "NYSE": "569a4e44e9fa258acbc799ed43462e93d38b3ceb8ae075474bcfe60f5ef5c11f",
    "RUSSELL": "c3c0e5eaaf60cc06047d4d79ba7288d744365642f6a6b7ad4963eaca53967857",
    "S&P": "62f39a586e101390c915f65af3c37caa0e19ef6a979de9d20fb2e006ec52053e",
}

# Row Partitioning Parameters
RAW_ROW_COUNT = 1984
RAW_COL_COUNT = 84
WARMUP_ROW_COUNT = 200

# Post-Warmup Dimensions
POST_WARMUP_ROW_COUNT = 1784   # Rows 200 to 1983 (index 0 to 1783)
LABELED_ROW_COUNT = 1783       # Rows 200 to 1982 (index 0 to 1782)

# Chronological Train-Embargo-Test Partition
TRAIN_ROW_COUNT = 1425         # Post-indices 0 to 1424 (Dates: 2010-10-18 to 2016-06-15)
EMBARGO_ROW_COUNT = 1          # Post-index 1425 (Date: 2016-06-16)
TEST_ROW_COUNT = 357           # Post-indices 1426 to 1782 (Dates: 2016-06-17 to 2017-11-14)
# Invariant: TRAIN_ROW_COUNT + EMBARGO_ROW_COUNT + TEST_ROW_COUNT == LABELED_ROW_COUNT (1425 + 1 + 357 == 1783)

TEST_MAJORITY_CLASS_RATIO = 0.55462745  # 198 Up / 357 Test = 55.46% baseline

# Transaction Vocabulary Specification (Strict 24 Tokens)
CONTEMPORANEOUS_ITEMS = [
    "SameDay_Up",
    "SameDay_Down",
    "Volume_Jump",
    "Mom5_Pos",
    "Mom5_Neg",
    "Above_EMA50",
    "Below_EMA50",
    "TermSpread_Steep",
    "TermSpread_Flat",
    "TermSpread_Widening",
    "CreditSpread_High",
    "CreditSpread_Low",
    "CreditSpread_Widening",
    "USD_Up",
    "USD_Down",
    "Gold_Up",
    "Gold_Down",
    "Oil_Up",
    "Oil_Down",
    "Tech_Leaders_Up",
    "Europe_Up",
    "Asia_Up",
]

PREDICTIVE_TARGET_ITEMS = [
    "NextDay_Up",
    "NextDay_NotUp",
]

ALL_TRANSACTION_ITEMS = CONTEMPORANEOUS_ITEMS + PREDICTIVE_TARGET_ITEMS
assert len(CONTEMPORANEOUS_ITEMS) == 22, "Contemporaneous vocabulary must have exactly 22 items"
assert len(ALL_TRANSACTION_ITEMS) == 24, "Full transaction vocabulary must have exactly 24 items"

# Global Random State
RANDOM_SEED = 42
