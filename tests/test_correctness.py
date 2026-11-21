"""
Independent correctness tests (added in the final verification pass).
Unlike the artefact-existence tests, these recompute results from first principles.
"""
import json
import math
import random
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DATA_DIR, TABLES_DIR, TRAIN_ROW_COUNT, EMBARGO_ROW_COUNT, TEST_ROW_COUNT
from src.pincer_search import PincerSearch
from src.data_loader import load_raw_dataset
from src.preprocessing import build_representation_a


# ---------------------------------------------------------------- Pincer Search
def brute_force_mfi(transactions, min_count):
    items = sorted({i for t in transactions for i in t})
    tsets = [set(t) for t in transactions]
    frequent = []
    for k in range(1, len(items) + 1):
        level = [frozenset(c) for c in combinations(items, k)
                 if sum(1 for t in tsets if set(c) <= t) >= min_count]
        if not level:
            break
        frequent.extend(level)
    return {f for f in frequent if not any(f < g for g in frequent)}


@pytest.mark.parametrize("seed", range(60))
def test_pincer_equals_brute_force_on_random_databases(seed):
    rng = random.Random(seed)
    n_items = rng.randint(4, 9)
    items = [f"i{j}" for j in range(n_items)]
    density = rng.uniform(0.3, 0.8)
    db = [[i for i in items if rng.random() < density] for _ in range(rng.randint(8, 40))]
    db = [t for t in db if t] or [[items[0]]]
    minsup = rng.uniform(0.1, 0.6)
    ps = PincerSearch(db, min_support=minsup)
    mfs = ps.fit()
    assert mfs == brute_force_mfi(db, ps.min_count)
    # antichain
    assert not any(a < b for a in mfs for b in mfs)


def test_pincer_textbook_example():
    db = [["A", "B", "C"], ["A", "B", "C", "D"], ["B", "C", "D"], ["A", "C", "D"], ["A", "B", "D"]]
    mfs = PincerSearch(db, min_support=0.6).fit()  # min_count = 3
    assert mfs == {frozenset("AB"), frozenset("AC"), frozenset("AD"), frozenset("BC"),
                   frozenset("BD"), frozenset("CD")}


# ---------------------------------------------------------------- preprocessing
def test_sentinels_removed_in_representation_a():
    rep_a = build_representation_a(load_raw_dataset("S&P"), "S&P")
    assert not (rep_a["Gold"] <= -0.99).any()
    assert not (rep_a["Oil"] <= -0.99).any()
    assert not (rep_a["DGS10"] == 0).any()
    # spreads on bond-holidays must also be blanked (previously left at 0.0)
    assert not (rep_a["TE2"] == 0).any()
    assert not (rep_a["DE1"] == 0).any()


# ---------------------------------------------------------------- transactions
@pytest.fixture(scope="module")
def tx():
    c1 = json.load(open(PROCESSED_DATA_DIR / "transactions_contemporaneous_S&P.json"))
    c2 = json.load(open(PROCESSED_DATA_DIR / "transactions_predictive_S&P.json"))
    return c1, c2


def test_every_vocabulary_item_actually_occurs(tx):
    c1, _ = tx
    counts = pd.Series([i for t in c1 for i in t]).value_counts()
    for item in ["Mom5_Pos", "Mom5_Neg", "USD_Up", "USD_Down", "Asia_Up", "Tech_Leaders_Up"]:
        assert counts.get(item, 0) > 100, f"{item} occurs only {counts.get(item, 0)} times"


def test_mutually_exclusive_pairs(tx):
    c1, _ = tx
    for t in c1:
        s = set(t)
        for a, b in [("SameDay_Up", "SameDay_Down"), ("Gold_Up", "Gold_Down"),
                     ("USD_Up", "USD_Down"), ("Mom5_Pos", "Mom5_Neg")]:
            assert not (a in s and b in s)


def test_target_flat_day_is_notup(tx):
    """Unchanged close must be NextDay_NotUp (never a separate/Down class)."""
    _, c2 = tx
    rep_a = build_representation_a(load_raw_dataset("S&P"), "S&P")
    close = rep_a["Close"].values
    for t in range(len(c2)):
        expected = "NextDay_Up" if close[t + 1] > close[t] else "NextDay_NotUp"
        assert expected in c2[t]


# ---------------------------------------------------------------- FP-Growth thresholds
def test_fp_growth_database_sizes_and_thresholds():
    fi = pd.read_csv(TABLES_DIR / "table_08_fp_growth_itemsets_S&P.csv")
    c = fi[fi["database"] == "Contemporaneous_Full_1784"]
    p = fi[fi["database"] == "Predictive_Train_1425"]
    assert c["N_transactions"].iloc[0] == 1784 and c["min_count_threshold"].iloc[0] == math.ceil(0.08 * 1784) == 143
    assert p["N_transactions"].iloc[0] == 1425 and p["min_count_threshold"].iloc[0] == math.ceil(0.08 * 1425) == 114
    assert (c["support_count"] >= 143).all() and (p["support_count"] >= 114).all()


def test_predictive_rules_have_target_consequent_and_test_metrics():
    r = pd.read_csv(TABLES_DIR / "table_10_validated_rules_S&P.csv")
    pred = r[r["rule_type"] == "Predictive"]
    assert len(pred) > 0
    assert pred["consequents"].str.contains("NextDay_").all()
    assert not pred["antecedents"].str.contains("NextDay_").any()
    assert pred["test_confidence"].notna().all()


# ---------------------------------------------------------------- classification leakage
def test_majority_baseline_from_training_labels_only():
    m = pd.read_csv(TABLES_DIR / "table_14_classification_metrics_S&P.csv")
    rep_a = build_representation_a(load_raw_dataset("S&P"), "S&P")
    close = rep_a["Close"].values
    y = (close[1:1784] > close[:1783]).astype(int)
    y_train = y[:TRAIN_ROW_COUNT]
    y_test = y[TRAIN_ROW_COUNT + EMBARGO_ROW_COUNT:TRAIN_ROW_COUNT + EMBARGO_ROW_COUNT + TEST_ROW_COUNT]
    assert len(y_train) == 1425 and len(y_test) == 357
    maj = int(y_train.mean() >= 0.5)
    assert m["Train_Majority_Class"].iloc[0] == maj == 1
    base = m[m["Model"] == "Majority_Baseline"]["Accuracy"].iloc[0]
    assert abs(base - (y_test == maj).mean()) < 1e-4


def test_spark_fpgrowth_identical_to_mlxtend():
    df = pd.read_csv(TABLES_DIR / "table_19_spark_mlxtend_fpgrowth_parity_S&P.csv")
    assert bool(df["Itemsets_Identical"].iloc[0])
    assert bool(df["Rules_Identical"].iloc[0])
    assert df["Max_Confidence_Abs_Diff"].iloc[0] < 1e-9
