"""
Association Rule Mining Module for CNNpred Big Data Pipeline.
Implements:
1. Contemporaneous Mining (Descriptive Intra-Day Market Structure):
   - Database: Representation C1 (N = 1,784, 22 items).
   - Denominator: 1,784.
   - Support Count: ceil(0.08 * 1,784) = 143.
2. Predictive Rule Discovery (Forward Market Prediction):
   - Training Database: Representation C2 Train (N = 1,425, 24 items).
   - Training Denominator: 1,425.
   - Training Support Count: ceil(0.08 * 1,425) = 114.
   - Consequent: Strictly NextDay_Up or NextDay_NotUp.
3. Predictive Out-of-Sample Test Evaluation:
   - Test Database: Representation C2 Test (N = 357, rows 1,426 to 1,782).
   - Test Denominator: 357.
   - Out-of-sample support, confidence, lift, conviction, and Fisher test.
4. Statistical significance testing: Fisher's Exact Test + Benjamini-Hochberg FDR correction.
"""
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    CONTEMPORANEOUS_ITEMS,
    PREDICTIVE_TARGET_ITEMS,
    TRAIN_ROW_COUNT,
    EMBARGO_ROW_COUNT,
    TEST_ROW_COUNT,
    TABLES_DIR,
)


def run_integer_fp_growth(
    df_binary: pd.DataFrame,
    minsup: float = 0.08,
    database_label: str = "Dataset",
) -> pd.DataFrame:
    """
    Executes FP-Growth on binary transaction DataFrame using integer ceiling support count.
    sigma_min = ceil(minsup * N).
    Returns DataFrame of frequent itemsets with columns:
    ['itemsets', 'support', 'support_count', 'length', 'database', 'N_transactions', 'min_count_threshold'].
    """
    from mlxtend.frequent_patterns import fpgrowth

    N = len(df_binary)
    min_count = math.ceil(minsup * N)
    effective_minsup = min_count / N

    feature_cols = [c for c in df_binary.columns if c != "Date"]
    X = df_binary[feature_cols].astype(bool)

    frequent_itemsets = fpgrowth(X, min_support=effective_minsup, use_colnames=True)
    frequent_itemsets["support_count"] = (frequent_itemsets["support"] * N).round().astype(int)
    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(lambda s: tuple(sorted(list(s))))
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    frequent_itemsets["database"] = database_label
    frequent_itemsets["N_transactions"] = N
    frequent_itemsets["min_count_threshold"] = min_count

    return frequent_itemsets.sort_values(by=["support", "length"], ascending=[False, False]).reset_index(drop=True)


def compute_conviction(conf: float, supp_consequent: float) -> float:
    """Computes rule conviction: (1 - P(B)) / (1 - Conf(A => B))."""
    if conf >= 1.0:
        return float("inf")
    denom = 1.0 - conf
    if denom == 0.0:
        return float("inf")
    return float((1.0 - supp_consequent) / denom)


def generate_association_rules(
    frequent_itemsets: pd.DataFrame,
    N: int,
    min_confidence: float = 0.40,
    database_label: str = "Dataset",
) -> pd.DataFrame:
    """
    Generates association rules from frequent itemsets and calculates:
    Support, Confidence, Lift, Conviction, and rule type.
    """
    from mlxtend.frequent_patterns import association_rules

    if len(frequent_itemsets) == 0:
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        support_only=False,
    )

    if len(rules) == 0:
        return pd.DataFrame()

    rules["antecedents"] = rules["antecedents"].apply(lambda s: tuple(sorted(list(s))))
    rules["consequents"] = rules["consequents"].apply(lambda s: tuple(sorted(list(s))))

    convictions = []
    for _, row in rules.iterrows():
        c = compute_conviction(row["confidence"], row["consequent support"])
        convictions.append(c)
    rules["conviction"] = convictions

    # Classify rules as Predictive vs Contemporaneous
    rule_types = []
    for _, row in rules.iterrows():
        conseq = row["consequents"]
        if len(conseq) == 1 and conseq[0] in PREDICTIVE_TARGET_ITEMS:
            rule_types.append("Predictive")
        elif all(item in CONTEMPORANEOUS_ITEMS for item in conseq) and all(item in CONTEMPORANEOUS_ITEMS for item in row["antecedents"]):
            rule_types.append("Contemporaneous")
        else:
            rule_types.append("Mixed/Other")

    rules["rule_type"] = rule_types
    rules["database"] = database_label
    rules["N_transactions"] = N
    return rules


def validate_rules_significance(
    rules: pd.DataFrame,
    df_binary: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Performs two-sided Fisher's Exact Test on each rule's 2x2 contingency table,
    then applies Benjamini-Hochberg (BH) False Discovery Rate (FDR) control.
    """
    if len(rules) == 0:
        return rules

    p_values = []
    odds_ratios = []

    for _, row in rules.iterrows():
        ante = list(row["antecedents"])
        conseq = list(row["consequents"])

        mask_ante = df_binary[ante].all(axis=1) if len(ante) > 1 else (df_binary[ante[0]] == 1)
        mask_conseq = df_binary[conseq].all(axis=1) if len(conseq) > 1 else (df_binary[conseq[0]] == 1)

        # 2x2 Contingency Table
        n_11 = int((mask_ante & mask_conseq).sum())
        n_10 = int((mask_ante & ~mask_conseq).sum())
        n_01 = int((~mask_ante & mask_conseq).sum())
        n_00 = int((~mask_ante & ~mask_conseq).sum())

        table = [[n_11, n_10], [n_01, n_00]]
        odds, pval = stats.fisher_exact(table, alternative="two-sided")
        odds_ratios.append(odds)
        p_values.append(pval)

    rules_eval = rules.copy()
    rules_eval["fisher_odds_ratio"] = odds_ratios
    rules_eval["fisher_p_value"] = p_values

    # Benjamini-Hochberg FDR correction
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]
    ranks = np.arange(1, m + 1)
    adjusted_pvals = np.minimum.accumulate((sorted_pvals * m / ranks)[::-1])[::-1]
    adjusted_pvals = np.clip(adjusted_pvals, 0.0, 1.0)

    bh_pvals = np.zeros(m)
    bh_pvals[sorted_indices] = adjusted_pvals
    rules_eval["bh_adjusted_p_value"] = bh_pvals
    rules_eval["is_statistically_significant"] = rules_eval["bh_adjusted_p_value"] < alpha

    return rules_eval.sort_values(by=["lift", "confidence"], ascending=[False, False]).reset_index(drop=True)


def evaluate_predictive_rules_on_test(
    train_rules: pd.DataFrame,
    df_test_binary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluates discovered predictive rules out-of-sample on the test period (N_test = 357).
    Computes out-of-sample test support, confidence, lift, conviction, and test Fisher p-value.
    """
    if len(train_rules) == 0:
        return train_rules

    pred_rules = train_rules[train_rules["rule_type"] == "Predictive"].copy()
    if len(pred_rules) == 0:
        return train_rules

    N_test = len(df_test_binary)
    test_supps = []
    test_confs = []
    test_lifts = []
    test_convs = []
    test_fisher_pvals = []

    for _, row in pred_rules.iterrows():
        ante = list(row["antecedents"])
        conseq = list(row["consequents"])

        mask_ante = df_test_binary[ante].all(axis=1) if len(ante) > 1 else (df_test_binary[ante[0]] == 1)
        mask_conseq = df_test_binary[conseq].all(axis=1) if len(conseq) > 1 else (df_test_binary[conseq[0]] == 1)

        n_ante = int(mask_ante.sum())
        n_conseq = int(mask_conseq.sum())
        n_11 = int((mask_ante & mask_conseq).sum())
        n_10 = int((mask_ante & ~mask_conseq).sum())
        n_01 = int((~mask_ante & mask_conseq).sum())
        n_00 = int((~mask_ante & ~mask_conseq).sum())

        supp = n_11 / N_test
        conf = (n_11 / n_ante) if n_ante > 0 else 0.0
        supp_conseq = n_conseq / N_test
        lift = (conf / supp_conseq) if supp_conseq > 0 else 0.0
        conv = compute_conviction(conf, supp_conseq)

        table = [[n_11, n_10], [n_01, n_00]]
        _, pval = stats.fisher_exact(table, alternative="two-sided")

        test_supps.append(round(supp, 4))
        test_confs.append(round(conf, 4))
        test_lifts.append(round(lift, 4))
        test_convs.append(round(conv, 4) if not math.isinf(conv) else 999.0)
        test_fisher_pvals.append(round(pval, 6))

    pred_rules["test_support"] = test_supps
    pred_rules["test_confidence"] = test_confs
    pred_rules["test_lift"] = test_lifts
    pred_rules["test_conviction"] = test_convs
    pred_rules["test_fisher_p_value"] = test_fisher_pvals
    pred_rules["test_is_statistically_significant"] = [p < 0.05 for p in test_fisher_pvals]

    return pred_rules


def execute_association_mining_pipeline(
    df_binary_c1: pd.DataFrame,
    df_binary_c2: pd.DataFrame,
    target_index: str,
    minsup: float = 0.08,
    minconf: float = 0.40,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes end-to-end association rule mining with rigorous database separation:
    1. Contemporaneous intra-day mining:
       - Database: df_binary_c1 (N = 1,784, 22 items).
       - Denominator: 1,784.
       - Threshold: ceil(0.08 * 1,784) = 143 counts.
    2. Predictive rule discovery:
       - Database: df_binary_c2.iloc[:1425] (N_train = 1,425, 24 items).
       - Denominator: 1,425.
       - Threshold: ceil(0.08 * 1,425) = 114 counts.
       - Evaluated out-of-sample on test set (N_test = 357).
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Contemporaneous Mining (Descriptive Full-Sample, N = 1,784)
    itemsets_contemp = run_integer_fp_growth(
        df_binary_c1,
        minsup=minsup,
        database_label="Contemporaneous_Full_1784",
    )
    rules_contemp = generate_association_rules(
        itemsets_contemp,
        N=len(df_binary_c1),
        min_confidence=minconf,
        database_label="Contemporaneous_Full_1784",
    )
    val_rules_contemp = validate_rules_significance(rules_contemp, df_binary_c1)

    # 2. Predictive Mining (Training Sample Only, N = 1,425)
    df_train_c2 = df_binary_c2.iloc[:TRAIN_ROW_COUNT].copy()
    test_start = TRAIN_ROW_COUNT + EMBARGO_ROW_COUNT
    test_end = test_start + TEST_ROW_COUNT
    df_test_c2 = df_binary_c2.iloc[test_start:test_end].copy()

    itemsets_predictive = run_integer_fp_growth(
        df_train_c2,
        minsup=minsup,
        database_label="Predictive_Train_1425",
    )
    raw_rules_pred = generate_association_rules(
        itemsets_predictive,
        N=len(df_train_c2),
        min_confidence=minconf,
        database_label="Predictive_Train_1425",
    )
    # Filter strictly predictive rules
    pred_only_rules = raw_rules_pred[raw_rules_pred["rule_type"] == "Predictive"].copy()
    val_rules_pred = validate_rules_significance(pred_only_rules, df_train_c2)

    # Out-of-sample test evaluation for predictive rules
    val_rules_pred = evaluate_predictive_rules_on_test(val_rules_pred, df_test_c2)

    # Combine Itemsets
    all_itemsets = pd.concat([itemsets_contemp, itemsets_predictive], ignore_index=True)
    all_itemsets.to_csv(TABLES_DIR / f"table_08_fp_growth_itemsets_{target_index}.csv", index=False)

    # Combine Rules
    all_raw_rules = pd.concat([rules_contemp, pred_only_rules], ignore_index=True)
    all_raw_rules.to_csv(TABLES_DIR / f"table_09_association_rules_{target_index}.csv", index=False)

    # Combine Validated Rules
    all_val_rules = pd.concat([val_rules_contemp, val_rules_pred], ignore_index=True)
    all_val_rules.to_csv(TABLES_DIR / f"table_10_validated_rules_{target_index}.csv", index=False)

    return all_itemsets, all_raw_rules, all_val_rules
