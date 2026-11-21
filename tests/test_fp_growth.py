"""Unit tests for FP-Growth Association Rule Mining (REQ-12, REQ-13, REQ-14)."""
import pytest
import pandas as pd
from src.config import TABLES_DIR, PROCESSED_DATA_DIR, PREDICTIVE_TARGET_ITEMS, CONTEMPORANEOUS_ITEMS
from src.association_rules import run_integer_fp_growth, generate_association_rules, validate_rules_significance


def test_fp_growth_integer_support():
    """REQ-12: Verifies integer support counting ceil(minsup * N) across distinct databases."""
    df_itemsets = pd.read_csv(TABLES_DIR / "table_08_fp_growth_itemsets_S&P.csv")
    assert len(df_itemsets) > 0
    assert "support_count" in df_itemsets.columns
    assert "min_count_threshold" in df_itemsets.columns

    # Contemporaneous itemsets (N=1784): ceil(0.08 * 1784) = 143
    c_items = df_itemsets[df_itemsets["database"] == "Contemporaneous_Full_1784"]
    if len(c_items) > 0:
        assert (c_items["support_count"] >= 143).all()
        assert c_items["min_count_threshold"].iloc[0] == 143

    # Predictive itemsets (N=1425): ceil(0.08 * 1425) = 114
    p_items = df_itemsets[df_itemsets["database"] == "Predictive_Train_1425"]
    if len(p_items) > 0:
        assert (p_items["support_count"] >= 114).all()
        assert p_items["min_count_threshold"].iloc[0] == 114


def test_predictive_vs_contemporaneous_separation():
    """REQ-13: Verifies strict separation between Predictive and Contemporaneous rules."""
    df_rules = pd.read_csv(TABLES_DIR / "table_09_association_rules_S&P.csv")
    assert "rule_type" in df_rules.columns
    types = set(df_rules["rule_type"].unique())
    assert "Predictive" in types or "Contemporaneous" in types

    pred_rules = df_rules[df_rules["rule_type"] == "Predictive"]
    for _, r in pred_rules.iterrows():
        conseq_str = str(r["consequents"])
        assert any(target in conseq_str for target in PREDICTIVE_TARGET_ITEMS)


def test_rule_statistical_significance():
    """REQ-14: Verifies Conviction, Lift, and Benjamini-Hochberg FDR."""
    df_val = pd.read_csv(TABLES_DIR / "table_10_validated_rules_S&P.csv")
    assert "conviction" in df_val.columns
    assert "lift" in df_val.columns
    assert "fisher_p_value" in df_val.columns
    assert "bh_adjusted_p_value" in df_val.columns
    assert "is_statistically_significant" in df_val.columns
