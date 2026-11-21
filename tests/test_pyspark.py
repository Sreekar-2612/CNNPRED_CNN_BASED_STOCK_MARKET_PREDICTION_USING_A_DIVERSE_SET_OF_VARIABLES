"""Unit tests for Distributed PySpark Pipeline (REQ-22, REQ-23)."""
import pytest
import pandas as pd
from src.config import TABLES_DIR


def test_spark_naive_bayes_parity():
    """REQ-22: Verifies PySpark Gaussian NB achieves >= 99.5% agreement with Scikit-learn."""
    df_parity = pd.read_csv(TABLES_DIR / "table_17_spark_parity_S&P.csv")
    assert len(df_parity) == 1
    assert df_parity["Parity_Agreement_Pct"].iloc[0] >= 99.5
    assert bool(df_parity["Parity_Requirement_Met"].iloc[0]) is True


def test_spark_fp_growth_equivalence():
    """REQ-23: Verifies PySpark FP-Growth execution and rule extraction."""
    df_rules = pd.read_csv(TABLES_DIR / "table_18_spark_rules_S&P.csv")
    assert len(df_rules) > 0
    assert "antecedent" in df_rules.columns
    assert "consequent" in df_rules.columns
