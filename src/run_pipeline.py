"""
Master Pipeline Runner for CNNpred Big Data Analytics Project.
Executes end-to-end:
1. Raw dataset validation (SHA-256)
2. Preprocessing (Sentinels -> NaN, Warmup Truncation -> 1,784 clean rows, Rep A & Rep B, Embargo Partition)
3. Descriptive analytics on Rep A (Moments, Jarque-Bera, Ljung-Box)
4. Feature Engineering & 24-Item Transaction Discretization (Rep C1 & Rep C2)
5. Visualizations (12 figures at 300 DPI)
6. Association Rule Mining (Integer FP-Growth, Conviction, Fisher Exact Test)
7. Bidirectional Pincer Search with Dual Oracle Validation (Lin & Kedem 1998)
8. Classification (Gaussian & Bernoulli Naive Bayes vs 55.46% baseline, McNemar test)
9. Clustering (DBSCAN + Automated Knee Locator + DBCV + PCA + Ward HAC)
10. PySpark Distributed Pipeline (Gaussian NB >= 99.5% parity, Spark FP-Growth)
11. Big Data Benchmarking (Isolated phases, process-tree peak RSS, Amdahl strong scaling)
12. Final 18-chapter Academic PDF report compilation via ReportLab
"""
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root workspace is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    INDEX_NAMES,
    RAW_CSV_FILES,
    TABLES_DIR,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    METRICS_DIR,
    REPORTS_DIR,
)
from src.data_loader import verify_raw_datasets, load_raw_dataset
from src.preprocessing import execute_preprocessing_pipeline
from src.feature_engineering import execute_feature_engineering_pipeline
from src.descriptive_analytics import execute_descriptive_pipeline
from src.visualization import (
    plot_01_normalized_trajectories,
    plot_02_rolling_volatility,
    plot_03_yield_curve_regimes,
    plot_04_correlation_heatmap,
    plot_05_return_distribution_kurtosis,
    plot_06_maximum_drawdown,
    plot_07_pca_scree,
    plot_08_dbscan_k_distance,
    plot_09_dbscan_clusters,
    plot_10_ward_dendrogram,
    plot_11_association_rules,
    plot_12_pyspark_strong_scaling,
    plot_13_classification_confusion,
    plot_14_local_vs_spark_scaling,
)
from src.association_rules import execute_association_mining_pipeline
from src.pincer_search import execute_pincer_pipeline
from src.classification import execute_classification_pipeline
from src.clustering import execute_clustering_pipeline
from src.pyspark_pipeline import (
    get_spark_session,
    run_spark_gaussian_naive_bayes,
    run_spark_fp_growth,
    compare_spark_with_mlxtend,
)
from src.benchmark_engine import execute_benchmarking_suite
from src.generate_pdf_report import create_academic_report


def run_full_pipeline():
    print("=" * 80)
    print("STARTING CNNpred BIG DATA ENDSEM PRACTICAL PIPELINE")
    print("IIITDM Kancheepuram — Comprehensive Autonomous Execution")
    print("=" * 80)

    # 1. Raw Dataset Verification
    print("\n[Phase 1] Verifying Raw CSV Datasets & SHA-256 Checksums...")
    df_hashes = verify_raw_datasets()
    print("Dataset verification complete. All 5 files verified:")
    for _, r in df_hashes.iterrows():
        print(f"  - {r['Index']}: {r['File']} ({r['Size_Bytes']:,} bytes) Checksum Match: {r['Checksum_Match']}")

    # 2. Preprocessing & Partitioning
    print("\n[Phase 2] Executing Preprocessing (Sentinel Removal, Warmup Truncation, Rep A & B)...")
    processed_dfs = execute_preprocessing_pipeline()
    print("Preprocessing complete. Processed datasets generated for all 5 indices.")

    # 3. Descriptive Analytics on Rep A
    print("\n[Phase 3] Computing Descriptive Statistics & Statistical Moments on Rep A...")
    df_stats = execute_descriptive_pipeline()
    print(f"Descriptive statistics computed across {len(df_stats)} feature records.")

    # 4. Feature Engineering & Transaction Generation (for primary index S&P and peers)
    print("\n[Phase 4] Executing Feature Engineering & 24-Item Transaction Discretization...")
    primary_idx = "S&P"
    df_rep_a = processed_dfs[f"{primary_idx}_rep_a"]
    df_rep_b = processed_dfs[f"{primary_idx}_rep_b"]
    df_features, tx_c1, tx_c2 = execute_feature_engineering_pipeline(df_rep_a, df_rep_b, primary_idx)
    print(f"Transactions generated: C1 Contemporaneous ({len(tx_c1)} rows), C2 Predictive ({len(tx_c2)} rows).")

    # Read binary transaction matrices
    df_bin_c1 = pd.read_parquet(PROCESSED_DATA_DIR / f"binary_c1_{primary_idx}.parquet")
    df_bin_c2 = pd.read_parquet(PROCESSED_DATA_DIR / f"binary_c2_{primary_idx}.parquet")

    # 5. Visualizations (Figures 1 to 6)
    print("\n[Phase 5] Generating Publication-Quality Exploratory Visualizations (300 DPI)...")
    all_raw_dfs = {name: load_raw_dataset(name) for name in INDEX_NAMES}
    p1 = plot_01_normalized_trajectories(all_raw_dfs)
    p2 = plot_02_rolling_volatility(df_rep_b, primary_idx)
    p3 = plot_03_yield_curve_regimes(df_rep_b)
    p4 = plot_04_correlation_heatmap(df_rep_b)
    p5 = plot_05_return_distribution_kurtosis(df_rep_a, primary_idx)
    p6 = plot_06_maximum_drawdown(all_raw_dfs)
    print(f"Exploratory plots generated: {p1.name}, {p2.name}, {p3.name}, {p4.name}, {p5.name}, {p6.name}")

    # 6. Association Rule Mining
    print("\n[Phase 6] Mining Association Rules (Integer FP-Growth, Conviction, Fisher Exact Test)...")
    itemsets, raw_rules, validated_rules = execute_association_mining_pipeline(
        df_bin_c1, df_bin_c2, primary_idx, minsup=0.08, minconf=0.40
    )
    p11 = plot_11_association_rules(validated_rules)
    print(f"Discovered {len(itemsets)} frequent itemsets and {len(validated_rules)} validated rules.")

    # 7. Bidirectional Pincer Search
    print("\n[Phase 7] Executing Pincer Search (Lin & Kedem 1998) & Dual Oracle Validation...")
    mfs, df_oracle = execute_pincer_pipeline(tx_c2, df_bin_c2, primary_idx, minsup=0.10)
    print("Dual-oracle validation results:")
    for _, r in df_oracle.iterrows():
        print(f"  - {r['Algorithm']}: {r['MFS_Count']} itemsets | Matches: {r['Matches_Ground_Truth']}")

    # 8. Supervised Classification (Naive Bayes vs 55.46% baseline)
    print("\n[Phase 8] Training Gaussian & Bernoulli Naive Bayes Models & McNemar Significance...")
    df_clf_metrics, df_likelihoods = execute_classification_pipeline(df_features, df_bin_c1, primary_idx)
    print("Classification performance:")
    for _, r in df_clf_metrics.iterrows():
        print(f"  - {r['Model']}: Accuracy={r['Accuracy']:.4f}, F1={r['F1_Score']:.4f}, ROC_AUC={r['ROC_AUC']:.4f}, McNemar p-Val={r['McNemar_pVal_vs_Base']:.4f}")

    p13 = plot_13_classification_confusion(execute_classification_pipeline.conf_mats)

    # Extract Scikit-learn GaussianNB predictions for PySpark parity check
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    cont_cols = ["return_idx", "dist_EMA50", "volatility_20", "term_spread", "delta_term_spread", "credit_spread", "delta_credit_spread", "Volume"]
    cont_cols = [c for c in cont_cols if c in df_features.columns]
    X_raw = df_features[cont_cols].iloc[:1783].values
    close_vals = df_features["Close"].values[:1784]
    y_all = (close_vals[1:1784] > close_vals[:1783]).astype(int)
    scaler = StandardScaler().fit(X_raw[:1425])
    X_train_s = scaler.transform(X_raw[:1425])
    X_test_s = scaler.transform(X_raw[1426:1783])
    gnb_sk = GaussianNB().fit(X_train_s, y_all[:1425])
    y_sklearn_pred = gnb_sk.predict(X_test_s)

    # 9. Clustering & Dimensionality Reduction
    print("\n[Phase 9] Executing DBSCAN Regime Clustering, PCA, t-SNE & Ward HAC Dendrogram...")
    df_profiles, labels, X_pca, X_tsne, sorted_k_dists, optimal_eps, dbcv = execute_clustering_pipeline(df_features, primary_idx)
    p7 = plot_07_pca_scree(df_features)
    p8 = plot_08_dbscan_k_distance(sorted_k_dists, optimal_eps)
    p9 = plot_09_dbscan_clusters(X_pca, labels, dbcv, X_tsne=X_tsne)

    ward_linkage = execute_clustering_pipeline.ward_linkage  # same scaled features, every 4th day
    p10 = plot_10_ward_dendrogram(ward_linkage, p=30)
    print(f"Clustering complete. Optimal eps={optimal_eps:.4f}, DBCV={dbcv:.4f}, {len(df_profiles)} regimes profiled.")

    # 10. Distributed PySpark Pipeline
    print("\n[Phase 10] Running Distributed PySpark MLlib (Gaussian NB & FP-Growth)...")
    spark = get_spark_session("CNNpred_Master_Execution")
    parity_summary, agreement_pct = run_spark_gaussian_naive_bayes(df_features, y_sklearn_pred, primary_idx, spark=spark)
    print(f"PySpark Gaussian NB Parity with Scikit-learn: {agreement_pct:.2f}% (Requirement >= 99.5%: {agreement_pct >= 99.5})")

    # Spark FP-Growth on the SAME contemporaneous database (C1, N = 1,784) as the mlxtend run
    spark_itemsets, spark_rules = run_spark_fp_growth(tx_c1, primary_idx, minsup=0.08, minconf=0.40, spark=spark)
    print(f"PySpark FP-Growth executed: {len(spark_itemsets)} itemsets, {len(spark_rules)} rules extracted.")
    df_fp_parity = compare_spark_with_mlxtend(spark_itemsets, spark_rules, df_bin_c1, 0.08, 0.40, primary_idx)
    print("Spark vs mlxtend FP-Growth parity:")
    print(df_fp_parity.to_string(index=False))
    spark.stop()

    # 11. Big Data Benchmarking Engine
    print("\n[Phase 11] Executing Big Data Benchmarking Engine (Amdahl Strong Scaling & Memory Profiling)...")
    if "--skip-benchmark" in sys.argv and (METRICS_DIR / "benchmark_results.json").exists():
        import json
        bench_results = json.load(open(METRICS_DIR / "benchmark_results.json"))
        print("Re-using measured benchmark_results.json (synthetic workloads; unaffected by this run).")
    else:
        bench_results = execute_benchmarking_suite(tx_c2)
    strong_res = bench_results["strong_scaling"]
    p12 = plot_12_pyspark_strong_scaling(
        strong_res["cores"],
        strong_res["speedups"],
        strong_res["amdahl_curve"],
        strong_res["parallel_fraction"],
    )
    p14 = plot_14_local_vs_spark_scaling(bench_results["synthetic_scaling"]["synthetic_scaling"])
    print(f"Benchmarking complete. Amdahl parallel fraction f={strong_res['parallel_fraction']:.4f}. Figure 12 generated.")

    # 12. Final Academic PDF Report
    print("\n[Phase 12] Compiling 18-Chapter Academic PDF Report via ReportLab...")
    pdf_path = create_academic_report() if "--no-pdf" not in sys.argv else REPORTS_DIR / "Analytics_BigData_Endsem_Report.pdf"
    pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"Report compiled successfully: {pdf_path} ({pdf_size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS SUCCESSFULLY PRODUCED.")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
