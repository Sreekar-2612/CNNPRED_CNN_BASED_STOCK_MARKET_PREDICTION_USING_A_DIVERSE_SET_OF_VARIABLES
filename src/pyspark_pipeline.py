"""
Distributed PySpark MLlib Pipeline Module for CNNpred Big Data Pipeline.
Implements:
1. SparkSession initialization with optimized JVM driver and executor settings.
2. Distributed Gaussian Naive Bayes (modelType="gaussian") for continuous financial features.
3. Prediction parity verification against Scikit-learn (target >= 99.5% agreement).
4. Distributed FP-Growth on transaction itemsets.
5. Systematic result export via .toPandas() to CSV and Parquet.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

# Enforce Python executable for PySpark worker processes
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.fpm import FPGrowth

from src.config import (
    TRAIN_ROW_COUNT,
    EMBARGO_ROW_COUNT,
    TEST_ROW_COUNT,
    TABLES_DIR,
)


def get_spark_session(
    app_name: str = "CNNpred_BigData_Pipeline",
    master: str = "local[*]",
) -> SparkSession:
    """
    Creates or retrieves active SparkSession with optimized configuration:
    - 4GB driver memory
    - Disabled UI for low overhead
    - 4 default shuffle partitions for local parallel processing
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", "4g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_spark_gaussian_naive_bayes(
    df_features: pd.DataFrame,
    y_sklearn_pred: np.ndarray,
    target_index: str,
    spark: Optional[SparkSession] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Trains PySpark Gaussian Naive Bayes on standardized continuous features.
    Verifies parity against Scikit-learn predictions.
    Saves results to outputs/tables/table_17_spark_parity.csv.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if spark is None:
        spark = get_spark_session("Spark_Gaussian_NB")

    # Features and Target construction
    cont_cols = [
        "return_idx", "dist_EMA50", "volatility_20", "term_spread",
        "delta_term_spread", "credit_spread", "delta_credit_spread", "Volume"
    ]
    cont_cols = [c for c in cont_cols if c in df_features.columns]

    close_vals = df_features["Close"].values
    n_labeled = min(len(close_vals) - 1, 1783)
    y_all = (close_vals[1 : n_labeled + 1] > close_vals[:n_labeled]).astype(int)

    # Standardize continuous features using training rows
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_raw = df_features[cont_cols].iloc[:TRAIN_ROW_COUNT].values
    scaler.fit(X_train_raw)
    X_scaled = scaler.transform(df_features[cont_cols].iloc[:n_labeled].values)

    df_data = pd.DataFrame(X_scaled, columns=cont_cols)
    df_data["label"] = y_all
    df_data["row_id"] = np.arange(len(df_data))

    # Convert to PySpark DataFrame
    spark_df = spark.createDataFrame(df_data)

    assembler = VectorAssembler(inputCols=cont_cols, outputCol="features")
    spark_assembled = assembler.transform(spark_df)

    # Chronological partition
    train_df = spark_assembled.filter(F.col("row_id") < TRAIN_ROW_COUNT)
    test_start = TRAIN_ROW_COUNT + EMBARGO_ROW_COUNT
    test_end = test_start + TEST_ROW_COUNT
    test_df = spark_assembled.filter((F.col("row_id") >= test_start) & (F.col("row_id") < test_end))

    # Gaussian Naive Bayes (modelType="gaussian" handles continuous positive and negative values)
    nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="gaussian")
    model = nb.fit(train_df)

    # Predict on test set
    predictions = model.transform(test_df)
    pred_df = predictions.select("row_id", "label", "prediction", "probability").toPandas()
    pred_df = pred_df.sort_values(by="row_id").reset_index(drop=True)

    y_spark_pred = pred_df["prediction"].values.astype(int)
    y_test_true = pred_df["label"].values.astype(int)

    # Compute Parity with Scikit-learn
    agreement_count = int(np.sum(y_spark_pred == y_sklearn_pred))
    parity_agreement_pct = (agreement_count / len(y_sklearn_pred)) * 100.0

    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")
    spark_acc = float(evaluator_acc.evaluate(predictions))
    spark_f1 = float(evaluator_f1.evaluate(predictions))

    parity_summary = pd.DataFrame([{
        "Index": target_index,
        "Test_Observations": len(y_sklearn_pred),
        "Identical_Predictions": agreement_count,
        "Parity_Agreement_Pct": round(parity_agreement_pct, 2),
        "Parity_Requirement_Met": parity_agreement_pct >= 99.5,
        "Spark_Accuracy": round(spark_acc, 4),
        "Spark_F1": round(spark_f1, 4),
    }])

    parity_summary.to_csv(TABLES_DIR / f"table_17_spark_parity_{target_index}.csv", index=False)
    return parity_summary, parity_agreement_pct


def run_spark_fp_growth(
    transactions: List[List[str]],
    target_index: str,
    minsup: float = 0.08,
    minconf: float = 0.40,
    spark: Optional[SparkSession] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs distributed PySpark FP-Growth on transactions.
    Extracts frequent itemsets and association rules, then exports via .toPandas().
    Saves to outputs/tables/table_18_spark_rules.csv.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if spark is None:
        spark = get_spark_session("Spark_FPGrowth")

    # Build Spark DataFrame with schema: ArrayType(StringType())
    records = [(i, tx) for i, tx in enumerate(transactions)]
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("items", ArrayType(StringType()), False),
    ])

    spark_tx_df = spark.createDataFrame(records, schema=schema)

    # Distributed FP-Growth
    fp = FPGrowth(itemsCol="items", minSupport=minsup, minConfidence=minconf)
    model = fp.fit(spark_tx_df)

    # Itemsets
    itemsets_spark = model.freqItemsets.toPandas()
    itemsets_spark["items"] = itemsets_spark["items"].apply(lambda s: tuple(sorted(list(s))))
    itemsets_spark["length"] = itemsets_spark["items"].apply(len)

    # Rules
    rules_spark = model.associationRules.toPandas()
    if len(rules_spark) > 0:
        rules_spark["antecedent"] = rules_spark["antecedent"].apply(lambda s: tuple(sorted(list(s))))
        rules_spark["consequent"] = rules_spark["consequent"].apply(lambda s: tuple(sorted(list(s))))
        rules_spark = rules_spark.sort_values(by=["confidence", "lift"], ascending=[False, False]).reset_index(drop=True)

    rules_spark.to_csv(TABLES_DIR / f"table_18_spark_rules_{target_index}.csv", index=False)
    return itemsets_spark, rules_spark


def compare_spark_with_mlxtend(itemsets_spark: pd.DataFrame, rules_spark: pd.DataFrame,
                               df_binary: pd.DataFrame, minsup: float, minconf: float,
                               target_index: str) -> pd.DataFrame:
    """
    Exact equivalence check: Spark FPGrowth vs mlxtend fpgrowth on the SAME transaction
    database. Spark keeps itemsets with count >= ceil(minsup*N); mlxtend is called with
    the identical integer threshold. Spark only emits single-item consequents, so rules
    are compared on |consequent| = 1.
    """
    import math
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    N = len(df_binary)
    eff = math.ceil(minsup * N) / N
    X = df_binary[[c for c in df_binary.columns if c != "Date"]].astype(bool)
    fi = fpgrowth(X, min_support=eff, use_colnames=True)
    ml_sets = {frozenset(s): int(round(v * N)) for s, v in zip(fi["itemsets"], fi["support"])}
    sp_sets = {frozenset(s): int(c) for s, c in zip(itemsets_spark["items"], itemsets_spark["freq"])}
    ar = association_rules(fi, metric="confidence", min_threshold=minconf)
    ar = ar[ar["consequents"].apply(len) == 1]
    ml_rules = {(frozenset(a), frozenset(c)): float(v) for a, c, v in zip(ar["antecedents"], ar["consequents"], ar["confidence"])}
    sp_rules = {(frozenset(a), frozenset(c)): float(v) for a, c, v in zip(rules_spark["antecedent"], rules_spark["consequent"], rules_spark["confidence"])}
    common = set(ml_rules) & set(sp_rules)
    max_conf_diff = max((abs(ml_rules[k] - sp_rules[k]) for k in common), default=0.0)
    df = pd.DataFrame([{
        "Index": target_index, "N_Transactions": N, "Min_Count": math.ceil(minsup * N),
        "mlxtend_Itemsets": len(ml_sets), "Spark_Itemsets": len(sp_sets),
        "Itemsets_Identical": ml_sets == sp_sets,
        "mlxtend_Rules_SingleConsequent": len(ml_rules), "Spark_Rules": len(sp_rules),
        "Rules_Identical": set(ml_rules) == set(sp_rules),
        "Max_Confidence_Abs_Diff": max_conf_diff,
    }])
    df.to_csv(TABLES_DIR / f"table_19_spark_mlxtend_fpgrowth_parity_{target_index}.csv", index=False)
    return df
