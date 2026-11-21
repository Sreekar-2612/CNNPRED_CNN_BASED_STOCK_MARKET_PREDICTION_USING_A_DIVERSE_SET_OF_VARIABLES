"""
Script to build and execute all 11 Jupyter Notebooks for the CNNpred Hackathon.
Ensures every notebook contains rich markdown documentation, executable code importing from src/,
and executed outputs preserved in the notebook JSON.
"""
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_and_execute_notebook(filename: str, title: str, cells_spec: list):
    """Creates a notebook with specified cells, executes it, and saves it."""
    nb = nbf.v4.new_notebook()
    nb["metadata"]["language_info"] = {"name": "python"}

    # Add title and header markdown
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n**Analytics and Systems of Big Data Practice — Endsem Hackathon**\n*IIITDM Kancheepuram*"))

    for cell_type, content in cells_spec:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content))

    out_path = NOTEBOOKS_DIR / filename
    print(f"Executing and saving notebook: {filename}...")

    # Execute notebook using NotebookClient
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated: {out_path} ({out_path.stat().st_size:,} bytes)")


def build_all_notebooks():
    # Setup path setup snippet
    path_setup = (
        "import sys, os\n"
        "from pathlib import Path\n"
        "ROOT = Path(os.getcwd()).resolve()\n"
        "if ROOT.name == 'notebooks':\n"
        "    ROOT = ROOT.parent\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n"
        "import pandas as pd, numpy as np\n"
        "from src.config import *\n"
        "from IPython.display import Image, display\n"
    )

    # -------------------------------------------------------------
    # Notebook 01: Dataset Exploration
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "01_dataset_exploration.ipynb",
        "01 — Dataset Exploration and Provenance Validation",
        [
            ("markdown", "## 1. Environment Setup & Data Provenance\nVerify the cryptographic integrity of all five raw index files using SHA-256 hashes."),
            ("code", path_setup + "\nfrom src.data_loader import verify_raw_datasets, load_raw_dataset, get_column_classification\ndf_hashes = verify_raw_datasets()\ndf_hashes"),
            ("markdown", "## 2. Dimensionality and Schema Inspection\nInspect the dimensions of each index file ($1,984 \\times 84$) and verify the column classification."),
            ("code", "df_sp = load_raw_dataset('S&P')\nprint('S&P Shape:', df_sp.shape)\nprint('Date Range:', df_sp['Date'].iloc[0], 'to', df_sp['Date'].iloc[-1])\nclassification = get_column_classification(df_sp, 'S&P')\n{k: len(v) for k, v in classification.items()}"),
            ("markdown", "## 3. Sample Data Inspection\nDisplay the first 5 rows of key financial and macroeconomic columns."),
            ("code", "sample_cols = ['Date', 'Close', 'Volume', 'ROC_5', 'DGS10', 'DTB3', 'DBAA', 'Gold', 'Oil', 'Dollar index']\ndf_sp[[c for c in sample_cols if c in df_sp.columns]].head()"),
        ]
    )

    # -------------------------------------------------------------
    # Notebook 02: Data Preprocessing
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "02_data_preprocessing.ipynb",
        "02 — Preprocessing, Sentinel Elimination & Chronological Embargo Partition",
        [
            ("markdown", "## 1. Sentinel Detection and Elimination\nNon-NaN sentinels: 0.0 in US rate columns (15 dates) and -1.0 in foreign returns (40 dates)."),
            ("code", path_setup + "\nfrom src.preprocessing import clean_sentinels, truncate_warmup, execute_preprocessing_pipeline\nprocessed_dfs = execute_preprocessing_pipeline()\ndf_sentinel_audit = pd.read_csv(TABLES_DIR / 'table_02_sentinel_audit.csv')\nprint('Total Sentinel Instances Found:', len(df_sentinel_audit))\ndf_sentinel_audit.head(10)"),
            ("markdown", "## 2. Warm-Up Truncation & Representations A and B\nRemove the first 200 warm-up rows (leaving 1,784 clean rows)."),
            ("code", "df_trunc = pd.read_csv(TABLES_DIR / 'table_03_truncation_summary.csv')\ndf_trunc"),
            ("markdown", "## 3. Chronological Train-Embargo-Test Partition\nPartition: 1,425 Train + 1 Embargo + 357 Test = 1,783 Labeled observations."),
            ("code", "df_splits = pd.read_csv(TABLES_DIR / 'table_04_split_partitions.csv')\ndf_splits"),
            ("markdown", "## 4. Test Target Distribution and Honest Baseline\nVerify the exact test majority class ratio (55.46%)."),
            ("code", "df_target_dist = pd.read_csv(TABLES_DIR / 'table_05_target_distribution_S&P.csv')\ndf_target_dist"),
        ]
    )

    # -------------------------------------------------------------
    # Notebook 03: Exploratory Data Analysis
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "03_exploratory_data_analysis.ipynb",
        "03 — Exploratory Data Analysis & Descriptive Analytics",
        [
            ("markdown", "## 1. Descriptive Statistics on Representation A (skipna=True)\nCompute moments, skewness, excess kurtosis, Jarque-Bera normality tests, and Ljung-Box autocorrelation."),
            ("code", path_setup + "\nfrom src.descriptive_analytics import execute_descriptive_pipeline\ndf_stats = execute_descriptive_pipeline()\ndf_stats[df_stats['Index'] == 'S&P'].head(10)"),
            ("markdown", "## 2. Multi-Index Normalized Trajectories & Volatility Clustering\nDisplay Fig 1 (Normalized Trajectories) and Fig 2 (Rolling Volatility Clustering)."),
            ("code", "from IPython.display import Image, display\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_01_multi_index_normalized_trajectories.png')))\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_02_rolling_volatility_clustering.png')))"),
            ("markdown", "## 3. Yield Curve & Credit Spread Dynamics\nDisplay Fig 3 (Yield Curve & Credit Spread Regimes) and Fig 4 (Spearman Rank Correlation Heatmap)."),
            ("code", "display(Image(filename=str(FIGURES_DIR / 'fig_03_yield_curve_regimes.png')))\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_04_correlation_heatmap.png')))"),
            ("markdown", "## 4. Heavy Tails and Drawdown Analysis\nDisplay Fig 5 (Return Distribution & Kurtosis) and Fig 6 (Maximum Drawdown)."),
            ("code", "display(Image(filename=str(FIGURES_DIR / 'fig_05_return_distribution_kurtosis.png')))\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_06_maximum_drawdown_comparison.png')))"),
        ]
    )

    # -------------------------------------------------------------
    # Notebook 04: Association Rule Mining
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "04_association_rule_mining.ipynb",
        "04 — Association Rule Mining & Statistical Significance",
        [
            ("markdown", "## 1. Transaction Representation and Discretization\nInspect Representation C1 (22 contemporaneous tokens) and Representation C2 (24 predictive tokens).\n\n**Database & Denominator Separation:**\n- **Contemporaneous Mining:** Database C1 (N = 1,784, 22 tokens), minsup count = ceil(0.08 * 1784) = 143.\n- **Predictive Rule Discovery:** Database C2 Train Only (N = 1,425, 24 tokens), minsup count = ceil(0.08 * 1425) = 114.\n- **Predictive Test Evaluation:** Evaluated out-of-sample on Test period (N = 357)."),
            ("code", path_setup + "\nimport json\nwith open(PROCESSED_DATA_DIR / 'transactions_predictive_S&P.json') as f:\n    tx = json.load(f)\nprint('Total Predictive Transactions:', len(tx))\nprint('Sample Transaction:', tx[0])"),
            ("markdown", "## 2. Integer Support FP-Growth & Rule Generation\nLoad mined frequent itemsets across both databases (143 for contemporaneous, 114 for predictive)."),
            ("code", "df_itemsets = pd.read_csv(TABLES_DIR / 'table_08_fp_growth_itemsets_S&P.csv')\nprint('Total Frequent Itemsets:', len(df_itemsets))\nprint('Breakdown by database:')\nprint(df_itemsets['database'].value_counts())\ndf_itemsets.head(10)"),
            ("markdown", "## 3. Statistically Validated Association Rules & Out-of-Sample Test Evaluation\nInspect rules filtered by Conviction, Lift, and two-sided Fisher's Exact Test with Benjamini-Hochberg FDR correction.\nFor predictive rules, out-of-sample test support, confidence, lift, and conviction are evaluated on the 357 test rows."),
            ("code", "df_rules = pd.read_csv(TABLES_DIR / 'table_10_validated_rules_S&P.csv')\nprint('Total Validated Rules:', len(df_rules))\nprint('Predictive Rules evaluated out-of-sample on test set:')\npred_val = df_rules[df_rules['rule_type'] == 'Predictive']\nif len(pred_val) > 0:\n    display(pred_val[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'test_support', 'test_confidence', 'test_lift', 'test_is_statistically_significant']].head(10))\nelse:\n    display(df_rules.head(10))"),
            ("markdown", "## 4. Association Rule Bubble Plot\nDisplay Fig 11 (Support vs Confidence vs Lift with Conviction colormap)."),
            ("code", "from IPython.display import Image, display\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_11_association_rules_support_confidence_lift.png')))")
        ]
    )

    # -------------------------------------------------------------
    # Notebook 05: Pincer Search Algorithm
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "05_pincer_search.ipynb",
        "05 — Bidirectional Pincer Search & Dual Oracle Validation",
        [
            ("markdown", "## 1. Theoretical Architecture of Pincer Search (Lin & Kedem 1998)\nBidirectional search: Bottom-up candidate generation ($C_k, L_k$) and top-down candidate reduction ($MFCS$)."),
            ("code", path_setup + "\nfrom src.pincer_search import PincerSearch, validate_pincer_against_oracles\nimport json\nwith open(PROCESSED_DATA_DIR / 'transactions_predictive_S&P.json') as f:\n    tx = json.load(f)\ndf_bin = pd.read_parquet(PROCESSED_DATA_DIR / 'binary_c2_S&P.parquet')\npincer = PincerSearch(tx, min_support=0.10)\nmfs = pincer.fit()\nprint('Discovered MFS Count:', len(mfs))"),
            ("markdown", "## 2. Level-Wise Iteration Logs & MFCS Antichain Pruning\nInspect the progress of bottom-up $L_k$ and top-down $MFCS$ at each pass."),
            ("code", "df_iter = pd.read_csv(TABLES_DIR / 'table_11_pincer_iterations_S&P.csv')\ndf_iter"),
            ("markdown", "## 3. Dual-Oracle Mathematical Validation\nVerify 100% exact set identity between Pincer Search, mlxtend.fpmax, and maximal FP-Growth."),
            ("code", "df_oracle = pd.read_csv(TABLES_DIR / 'table_13_oracle_comparison_S&P.csv')\ndf_oracle"),
        ]
    )

    # -------------------------------------------------------------
    # Notebook 06: Naive Bayes Classification
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "06_naive_bayes_classification.ipynb",
        "06 — Supervised Classification & McNemar Significance Testing",
        [
            ("markdown", "## 1. Classification Setup & The Honest 55.46% Baseline\nEvaluate Gaussian Naive Bayes (continuous) and Bernoulli Naive Bayes (binary) against the test majority baseline."),
            ("code", path_setup + "\nfrom src.classification import execute_classification_pipeline\ndf_clf = pd.read_csv(TABLES_DIR / 'table_14_classification_metrics_S&P.csv')\ndf_clf"),
            ("markdown", "## 2. Confusion Matrices & Performance Metrics\nCompare accuracy, precision, recall, F1, and ROC-AUC."),
            ("code", "df_clf[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'TP', 'FP', 'TN', 'FN']]"),
            ("markdown", "## 3. Feature Likelihood Inspection\nInspect class conditional means/variances for Gaussian NB and probability ratios for Bernoulli NB."),
            ("code", "df_lik = pd.read_csv(TABLES_DIR / 'table_15_feature_likelihoods_S&P.csv')\nprint('Gaussian Feature Parameters:')\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_13_naive_bayes_confusion_matrices.png')))\ndisplay(df_lik[df_lik['Model'] == 'GaussianNB'].head(8))\nprint('Bernoulli Feature Probabilities:')\ndisplay(df_lik[df_lik['Model'] == 'BernoulliNB'].sort_values(by='Ratio_C1_to_C0', ascending=False).head(8))"),
        ]
    )

    # -------------------------------------------------------------
    # Notebook 07: Clustering Analysis
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "07_clustering_analysis.ipynb",
        "07 — Unsupervised Market Regime Clustering (DBSCAN)",
        [
            ("markdown", "## 1. Market Regime Feature Space\nClustering on 5 domain regime features: volatility_20, term_spread, credit_spread, return_idx, Volume."),
            ("code", path_setup + "\nfrom src.clustering import execute_clustering_pipeline\ndf_profiles = pd.read_csv(TABLES_DIR / 'table_16_cluster_profiles_S&P.csv')\ndf_profiles"),
            ("markdown", "## 2. Automated Knee Detection & 9-Distance Graph\nDisplay Fig 8: Automated detection of optimal epsilon from k-distance elbow."),
            ("code", "from IPython.display import Image, display\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_08_dbscan_k_distance_knee.png')))"),
            ("markdown", "## 3. Market Regime Clusters in Latent Space\nDisplay Fig 9: DBSCAN regime clusters projected on 2D principal component space."),
            ("code", "display(Image(filename=str(FIGURES_DIR / 'fig_09_dbscan_regime_clusters_pca.png')))")
        ]
    )

    # -------------------------------------------------------------
    # Notebook 08: Dimensionality Reduction
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "08_dimensionality_reduction.ipynb",
        "08 — Dimensionality Reduction & Hierarchical Agglomerative Clustering",
        [
            ("markdown", "## 1. Principal Component Analysis (PCA) & Scree Plot\nDisplay Fig 7: Individual and cumulative explained variance ratio across principal components.\nThe scree plot is fitted on standardised stationary features (training rows only); its title states how many components reach 80% of the variance."),
            ("code", path_setup + "\nfrom IPython.display import Image, display\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_07_pca_scree_explained_variance.png')))"),
            ("markdown", "## 2. t-SNE Non-Linear Manifold Projection (REQ-21)\nDisplay Fig 9b: 2D non-linear manifold projection via t-Distributed Stochastic Neighbor Embedding.\n\n**Interpretation:** Unlike PCA, which is constrained to orthogonal linear projections, t-SNE preserves local neighborhood topologies and clearly separates the high-volatility crisis market regimes from normal expansionary regimes."),
            ("code", "display(Image(filename=str(FIGURES_DIR / 'fig_09b_tsne_manifold.png')))"),
            ("markdown", "## 3. Ward Hierarchical Agglomerative Clustering (HAC)\nDisplay Fig 10: Truncated Ward HAC dendrogram (p=30) uncovering hierarchical regime taxonomy."),
            ("code", "display(Image(filename=str(FIGURES_DIR / 'fig_10_ward_hac_dendrogram.png')))")
        ]
    )

    # -------------------------------------------------------------
    # Notebook 09: PySpark Implementation
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "09_pyspark_implementation.ipynb",
        "09 — Distributed PySpark MLlib Pipeline & Parity Verification",
        [
            ("markdown", "## 1. PySpark Gaussian Naive Bayes Configuration\nUsing modelType='gaussian' to correctly model continuous positive and negative financial returns."),
            ("code", path_setup + "\nfrom src.pyspark_pipeline import get_spark_session\nspark = get_spark_session('Notebook_Spark')\nprint('Spark Version:', spark.version)\ndf_parity = pd.read_csv(TABLES_DIR / 'table_17_spark_parity_S&P.csv')\ndf_parity"),
            ("markdown", "## 2. Distributed FP-Growth on PySpark\nInspect frequent itemsets and association rules mined with distributed Spark MLlib."),
            ("code", "df_spark_rules = pd.read_csv(TABLES_DIR / 'table_18_spark_rules_S&P.csv')\nprint('Total Spark Rules:', len(df_spark_rules))\ndf_spark_rules.head(10)\nspark.stop()"),
        ]
    )

    # -------------------------------------------------------------
    # Notebook 10: Big Data Benchmarking
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "10_big_data_benchmarking.ipynb",
        "10 — Big Data Benchmarking, Amdahl's Law & Memory Profiling",
        [
            ("markdown", "## 1. Phase-Isolated Latency & Process-Tree Peak RSS\nInspect isolated JVM startup, ingestion, warmup, and steady fit times."),
            ("code", path_setup + "\nimport json\nwith open(METRICS_DIR / 'benchmark_results.json') as f:\n    bench = json.load(f)\nprint('Amdahl Parallel Fraction f:', bench['strong_scaling']['parallel_fraction'])\nprint('Strong Scaling Summary:')\nfor p, s in zip(bench['strong_scaling']['cores'], bench['strong_scaling']['speedups']):\n    print(f'  - Cores: {p}, Speedup S(p): {s}')"),
            ("markdown", "## 2. PySpark Strong Scaling Speedup Curve\nDisplay Fig 12: Observed speedup vs Amdahl's Law asymptote."),
            ("code", "from IPython.display import Image, display\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_12_pyspark_strong_scaling_speedup.png')))\ndisplay(Image(filename=str(FIGURES_DIR / 'fig_14_local_vs_spark_data_scaling.png')))"),
            ("markdown", "## 3. Synthetic Scaling Across Increasing Dataset Sizes\nLatency and Peak Physical RSS (psutil) for local mlxtend vs Spark FPGrowth, N = 1,784 to 250,000 rows."),
            ("code", "pd.DataFrame(bench['synthetic_scaling']['synthetic_scaling'])")
        ]
    )

    # -------------------------------------------------------------
    # Notebook 11: Final Results and Conclusions
    # -------------------------------------------------------------
    create_and_execute_notebook(
        "11_final_results_and_conclusions.ipynb",
        "11 — Final Results Synthesis & Acceptance Verification",
        [
            ("markdown", "## 1. Master Acceptance Criteria Traceability\nReview the requirement traceability matrix (statuses reflect the final verification pass)."),
            ("code", path_setup + "\nwith open(ROOT / 'IMPLEMENTATION_TRACEABILITY.md') as f:\n    matrix = f.read()\nprint(matrix[:1500] + '...')"),
            ("markdown", "## 2. Summary of Produced Artifacts\nVerify generated tables, figures, models, metrics, and academic report."),
            ("code", "print('Total Figures in outputs/figures/:', len(list(FIGURES_DIR.glob('*.png'))))\nprint('Total Tables in outputs/tables/:', len(list(TABLES_DIR.glob('*.csv'))))\nprint('Generated Academic Report:', (REPORTS_DIR / 'Analytics_BigData_Endsem_Report.pdf').exists())"),
        ]
    )


if __name__ == "__main__":
    build_all_notebooks()
