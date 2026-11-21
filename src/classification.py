"""
Naive Bayes Classification Module for CNNpred Big Data Pipeline.
Implements:
1. Gaussian Naive Bayes on standardized continuous stationary features.
2. Bernoulli Naive Bayes on binary contemporaneous transaction features.
3. Benchmark evaluation against the honest 55.46% test majority baseline.
4. Statistical significance testing via McNemar's Test.
5. Class likelihood and feature parameter extraction.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from src.config import (
    TRAIN_ROW_COUNT,
    EMBARGO_ROW_COUNT,
    TEST_ROW_COUNT,
    TEST_MAJORITY_CLASS_RATIO,
    TABLES_DIR,
)


def mcnemar_test(y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_base: np.ndarray) -> Tuple[float, float]:
    """
    Computes McNemar's test with continuity correction:
    b: model correct, baseline wrong
    c: model wrong, baseline correct
    chi2 = (|b - c| - 1)^2 / (b + c)
    """
    corr_model = (y_pred_model == y_true)
    corr_base = (y_pred_base == y_true)

    b = int((corr_model & ~corr_base).sum())
    c = int((~corr_model & corr_base).sum())

    if (b + c) == 0:
        return 0.0, 1.0

    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_val = 1.0 - stats.chi2.cdf(stat, df=1)
    return float(stat), float(p_val)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba)
    except Exception:
        auc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # Exact one-sided binomial test: is accuracy above the 55.46% rate a constant
    # "always Up" rule achieves on the test period?
    n_correct = int((np.asarray(y_true) == np.asarray(y_pred)).sum())
    p_base = float(np.mean(np.asarray(y_true) == 1))
    binom_p = float(stats.binomtest(n_correct, len(y_true), p_base, alternative="greater").pvalue)

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(auc, 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
        "Binomial_p_vs_Majority": round(binom_p, 4),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def execute_classification_pipeline(
    df_features: pd.DataFrame,
    df_binary_c1: pd.DataFrame,
    target_index: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Gaussian & Bernoulli Naive Bayes on chronological train/embargo/test split:
    - Train: First 1,425 rows.
    - Embargo: Row 1,425 (skipped).
    - Test: Rows 1,426 to 1,782 (357 rows).
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare Target vector y for labeled rows (1,783 rows)
    close_vals = df_features["Close"].values
    n_labeled = min(len(close_vals) - 1, 1783)
    y_all = (close_vals[1 : n_labeled + 1] > close_vals[:n_labeled]).astype(int)

    # Partition indices
    train_idx = slice(0, TRAIN_ROW_COUNT)
    test_idx = slice(TRAIN_ROW_COUNT + EMBARGO_ROW_COUNT, TRAIN_ROW_COUNT + EMBARGO_ROW_COUNT + TEST_ROW_COUNT)

    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    # Majority class baseline predictions: strictly derived from 1,425 training labels
    train_c1 = int((y_train == 1).sum())
    train_c0 = int((y_train == 0).sum())
    train_majority_class = 1 if train_c1 >= train_c0 else 0
    train_majority_ratio = train_c1 / len(y_train) if train_majority_class == 1 else train_c0 / len(y_train)

    # Apply the fixed training-derived majority class to the 357 test observations
    y_pred_baseline = np.full(len(y_test), train_majority_class)
    y_proba_baseline = np.full(len(y_test), 0.5)

    test_c1 = int((y_test == 1).sum())
    test_c0 = int((y_test == 0).sum())
    test_majority_ratio = test_c1 / len(y_test)
    baseline_test_acc = accuracy_score(y_test, y_pred_baseline)

    # 1. Gaussian Naive Bayes on Continuous Stationary Features
    cont_cols = [
        "return_idx", "dist_EMA50", "volatility_20", "term_spread",
        "delta_term_spread", "credit_spread", "delta_credit_spread", "Volume"
    ]
    cont_cols = [c for c in cont_cols if c in df_features.columns]

    X_cont_all = df_features[cont_cols].iloc[:n_labeled].values
    scaler = StandardScaler()
    X_cont_train = scaler.fit_transform(X_cont_all[train_idx])
    X_cont_test = scaler.transform(X_cont_all[test_idx])

    gnb = GaussianNB()
    gnb.fit(X_cont_train, y_train)
    y_pred_gnb = gnb.predict(X_cont_test)
    y_proba_gnb = gnb.predict_proba(X_cont_test)

    # 2. Bernoulli Naive Bayes on Binary Contemporaneous Features
    bin_cols = [c for c in df_binary_c1.columns if c != "Date"]
    X_bin_all = df_binary_c1[bin_cols].iloc[:n_labeled].values
    X_bin_train = X_bin_all[train_idx]
    X_bin_test = X_bin_all[test_idx]

    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(X_bin_train, y_train)
    y_pred_bnb = bnb.predict(X_bin_test)
    y_proba_bnb = bnb.predict_proba(X_bin_test)

    # Statistical significance: McNemar test
    mc_stat_gnb, mc_pval_gnb = mcnemar_test(y_test, y_pred_gnb, y_pred_baseline)
    mc_stat_bnb, mc_pval_bnb = mcnemar_test(y_test, y_pred_bnb, y_pred_baseline)

    # Compile performance metrics table
    res_base = evaluate_classifier(y_test, y_pred_baseline, y_proba_baseline, "Majority_Baseline")
    res_base["McNemar_Stat_vs_Base"] = 0.0
    res_base["McNemar_pVal_vs_Base"] = 1.0

    res_gnb = evaluate_classifier(y_test, y_pred_gnb, y_proba_gnb, "GaussianNB_Continuous")
    res_gnb["McNemar_Stat_vs_Base"] = round(mc_stat_gnb, 4)
    res_gnb["McNemar_pVal_vs_Base"] = round(mc_pval_gnb, 4)

    res_bnb = evaluate_classifier(y_test, y_pred_bnb, y_proba_bnb, "BernoulliNB_Binary")
    res_bnb["McNemar_Stat_vs_Base"] = round(mc_stat_bnb, 4)
    res_bnb["McNemar_pVal_vs_Base"] = round(mc_pval_bnb, 4)

    df_metrics = pd.DataFrame([res_base, res_gnb, res_bnb])
    df_metrics["Index"] = target_index
    df_metrics["Train_Majority_Class"] = train_majority_class
    df_metrics["Train_Majority_Ratio"] = round(train_majority_ratio, 4)
    df_metrics["Test_Majority_Ratio"] = round(test_majority_ratio, 4)
    df_metrics["Baseline_Test_Accuracy"] = round(baseline_test_acc, 4)
    df_metrics.to_csv(TABLES_DIR / f"table_14_classification_metrics_{target_index}.csv", index=False)

    # Extract feature likelihoods and parameters
    likelihood_records = []
    # Gaussian parameters
    for idx, col in enumerate(cont_cols):
        likelihood_records.append({
            "Model": "GaussianNB",
            "Feature": col,
            "Class_0_Mean": round(float(gnb.theta_[0, idx]), 4),
            "Class_0_Var": round(float(gnb.var_[0, idx]), 4),
            "Class_1_Mean": round(float(gnb.theta_[1, idx]), 4),
            "Class_1_Var": round(float(gnb.var_[1, idx]), 4),
        })

    # Bernoulli log probabilities
    for idx, col in enumerate(bin_cols):
        p_c0 = np.exp(bnb.feature_log_prob_[0, idx])
        p_c1 = np.exp(bnb.feature_log_prob_[1, idx])
        likelihood_records.append({
            "Model": "BernoulliNB",
            "Feature": col,
            "Class_0_Prob": round(float(p_c0), 4),
            "Class_1_Prob": round(float(p_c1), 4),
            "Ratio_C1_to_C0": round(float(p_c1 / (p_c0 + 1e-6)), 4),
        })

    df_likelihoods = pd.DataFrame(likelihood_records)
    df_likelihoods.to_csv(TABLES_DIR / f"table_15_feature_likelihoods_{target_index}.csv", index=False)

    conf_mats = {
        "Majority baseline": confusion_matrix(y_test, y_pred_baseline, labels=[0, 1]),
        "Gaussian NB": confusion_matrix(y_test, y_pred_gnb, labels=[0, 1]),
        "Bernoulli NB": confusion_matrix(y_test, y_pred_bnb, labels=[0, 1]),
    }
    execute_classification_pipeline.conf_mats = conf_mats  # exposed for plotting
    execute_classification_pipeline.y_pred_gnb = y_pred_gnb
    return df_metrics, df_likelihoods
