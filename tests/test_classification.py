"""Unit tests for Naive Bayes Classification (REQ-18, REQ-19)."""
import pytest
import pandas as pd
from src.config import TABLES_DIR


def test_naive_bayes_performance():
    """REQ-18: Verifies Gaussian and Bernoulli NB evaluated vs 55.46% test baseline."""
    df_metrics = pd.read_csv(TABLES_DIR / "table_14_classification_metrics_S&P.csv")
    assert len(df_metrics) == 3
    models = df_metrics["Model"].tolist()
    assert "Majority_Baseline" in models
    assert "GaussianNB_Continuous" in models
    assert "BernoulliNB_Binary" in models

    base_acc = df_metrics[df_metrics["Model"] == "Majority_Baseline"]["Accuracy"].iloc[0]
    assert abs(base_acc - 0.5546) < 1e-3


def test_mcnemar_significance():
    """REQ-19: Verifies McNemar test results and feature likelihood extraction."""
    df_metrics = pd.read_csv(TABLES_DIR / "table_14_classification_metrics_S&P.csv")
    assert "McNemar_pVal_vs_Base" in df_metrics.columns

    df_lik = pd.read_csv(TABLES_DIR / "table_15_feature_likelihoods_S&P.csv")
    assert len(df_lik) > 0
    assert "GaussianNB" in df_lik["Model"].values
    assert "BernoulliNB" in df_lik["Model"].values
