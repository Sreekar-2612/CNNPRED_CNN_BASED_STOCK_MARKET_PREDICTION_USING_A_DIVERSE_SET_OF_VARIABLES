# 🧠 BIG DATA ANALYTICS: MASTER CURRICULUM BRAIN & HACKATHON COMBAT MANUAL

**Student:** T.S.R. Sreekar | **Roll No:** CS23I1034  
**Primary Working Directory for Hackathon:** `c:/Users/Sreekar_Tirumala/Documents/College/Big Data/lab_hackathon`  
**Master Context Base:** Complete sequential consolidation of Labs 1–12, Task 2 (BI/Tableau), and Semester Big Data Project (`BIG_DATA_PROJECT/urban-service-intelligence`).  
**Mission:** High-velocity reference, algorithmic synthesis, mathematical formulations, empirical results, and battle-tested live/real-time streaming templates for the Big Data Lab Hackathon.

---

## TABLE OF CONTENTS
1. [Executive Overview & Hackathon Tactical Blueprint](#1-executive-overview--hackathon-tactical-blueprint)
2. [Sequential Lab Directory & File Manifest](#2-sequential-lab-directory--file-manifest)
3. [Exhaustive Lab-by-Lab Deep Dive](#3-exhaustive-lab-by-lab-deep-dive)
   - [Lab 1 & 2: Spreadsheet Foundations, Pivot Tables & Descriptive Statistics](#lab-1--2-spreadsheet-foundations-pivot-tables--descriptive-statistics)
   - [Task 2 (Assignment 2): Visual Analytics, Tableau Dashboards & Storytelling](#task-2-assignment-2-visual-analytics-tableau-dashboards--storytelling)
   - [Lab 3: Multi-Domain Exploratory Data Analysis & Visual Analytics](#lab-3-multi-domain-exploratory-data-analysis--visual-analytics)
   - [Lab 4: Data Preprocessing, Transformations, Normalization & Binning](#lab-4-data-preprocessing-transformations-normalization--binning)
   - [Lab 5: Relational Data Integration, SQL Pipelines & Feature Engineering](#lab-5-relational-data-integration-sql-pipelines--feature-engineering)
   - [Lab 6: Frequent Itemset Mining (FIM) — Apriori vs FP-Growth](#lab-6-frequent-itemset-mining-fim--apriori-vs-fp-growth)
   - [Lab 7 (Assignment 7): Advanced FIM — Dynamic Itemset Counting, Hash Apriori & ECLAT](#lab-7-assignment-7-advanced-fim--dynamic-itemset-counting-hash-apriori--eclat)
   - [Lab 8: Advanced FIM Architectures — SON Partitioning, AClose & Pincer Search](#lab-8-advanced-fim-architectures--son-partitioning-aclose--pincer-search)
   - [Lab 9: Supervised Machine Learning — Decision Trees (CART), Pruning & Imbalance](#lab-9-supervised-machine-learning--decision-trees-cart-pruning--imbalance)
   - [Lab 10: Probabilistic Classification (Naive Bayes) & Synthetic Clustering](#lab-10-probabilistic-classification-naive-bayes--synthetic-clustering)
   - [Lab 11: Advanced Customer Segmentation, Density-Based Clustering (DBSCAN/OPTICS/GMM) & Metric Spaces](#lab-11-advanced-customer-segmentation-density-based-clustering-dbscanopticsgmm--metric-spaces)
   - [Lab 12: Distributed Big Data & Metaheuristics — Genetic Algorithms, PySpark FIM/DBSCAN & Multi-Classifier Clustering](#lab-12-distributed-big-data--metaheuristics--genetic-algorithms-pyspark-fimdbscan--multi-classifier-clustering)
   - [BIG_DATA_PROJECT: Urban Service Intelligence (NYC 311 PySpark Batch/Streaming Pipeline)](#big_data_project-urban-service-intelligence-nyc-311-pyspark-batchstreaming-pipeline)
4. [Master Cross-Lab Algorithmic Taxonomy & Decision Matrix](#4-master-cross-lab-algorithmic-taxonomy--decision-matrix)
5. [The 13 Distance & Dissimilarity Metrics Framework](#5-the-13-distance--dissimilarity-metrics-framework)
6. [Hackathon Live / Real-Time Dataset Operational Playbook](#6-hackathon-live--real-time-dataset-operational-playbook)
   - [Pattern A: Directory Watcher & Micro-Batch Streaming Feeds](#pattern-a-directory-watcher--micro-batch-streaming-feeds)
   - [Pattern B: Online & Incremental Machine Learning (Partial Fit)](#pattern-b-online--incremental-machine-learning-partial-fit)
   - [Pattern C: Rapid Triage of Fresh High-Velocity Datasets (First 10 Minutes)](#pattern-c-rapid-triage-of-fresh-high-velocity-datasets-first-10-minutes)
7. [Exhaustive Codebase Inventory & Architectural Patterns](#7-exhaustive-codebase-inventory--architectural-patterns)
8. [Plug-and-Play Production Code Vault (Copy-Paste Ready)](#7-plug-and-play-production-code-vault-copy-paste-ready)

---

## 1. EXECUTIVE OVERVIEW & HACKATHON TACTICAL BLUEPRINT

A Big Data Lab Hackathon testing a **live / real-time dataset** evaluates your ability to rapidly ingest, clean, explore, model, and visualize data under strict time constraints. The dataset may arrive as:
1. **Streaming / Continuous Micro-Batches:** File-drops (CSV/JSON/Parquet) arriving dynamically in a folder or simulated telemetry from IoT/web transactions.
2. **High-Velocity Dirty Tabular Feed:** Unseen real-world data with severe missingness, non-standard types, multi-modal distributions, extreme skewness, high-cardinality nominals, and class imbalance.
3. **Large-Scale Memory Bottleneck:** Millions of rows requiring either sub-linear algorithms (FP-Growth vs Apriori), streaming estimators (`partial_fit`), or distributed PySpark execution.

### The 7-Phase Execution Protocol
```
[1. INGEST & PROFILE] ➔ [2. AUDIT & RECTIFY] ➔ [3. TRANSFORM & SCALE] ➔ [4. PATTERN MINING / EDA] ➔ [5. MODEL & CLUSTER] ➔ [6. VALIDATE & DIAGNOSE] ➔ [7. VISUALIZE & REPORT]
```
- **Phase 1 (0–5 min):** Instant schema discovery, missingness audit, duplicate elimination, type casting.
- **Phase 2 (5–12 min):** Skewness evaluation ($|g_1| > 1.5 \implies \log(1+x)$) and IQR outlier capping.
- **Phase 3 (12–20 min):** Robust scaling (`StandardScaler` / `MinMaxScaler`) and categorical encoding (`OneHotEncoder` / `OrdinalEncoder`).
- **Phase 4 (20–30 min):** Unsupervised discovery via FP-Growth / Association Rules (`mlxtend` or PySpark `FPGrowth`).
- **Phase 5 (30–45 min):** Algorithmic execution:
  - If Unsupervised / Segmentation: DBSCAN ($k$-distance knee detection) or K-Means (Elbow method) or GMM (BIC minimizer).
  - If Supervised / Prediction: Decision Trees with Cost-Complexity Pruning (`ccp_alpha`) or Naive Bayes (`GaussianNB` / `BernoulliNB` / `ComplementNB`) or Random Forest.
  - If Optimization / Routing: Genetic Algorithms (SGA / TSP permutation).
- **Phase 6 (45–52 min):** Dual validation: Internal metrics (Silhouette, Davies-Bouldin) and Supervised metrics (PR-AUC, ROC-AUC, Macro-F1, Confusion Matrix).
- **Phase 7 (52–60 min):** Generate clean diagnostic figures (PCA 2D projection, ROC curve, Pareto distribution).

---

## 2. SEQUENTIAL LAB DIRECTORY & FILE MANIFEST

This table lists every single file in the workspace in chronological/sequential order of curriculum usage:

| Seq | Lab / Directory | File Name | Format | Role & Content |
|---|---|---|---|---|
| **01** | Root | [`BIG_DATA_LAB_1.ods`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/BIG_DATA_LAB_1.ods) | Spreadsheet | Basic statistics, descriptive metrics, initial manual data profiling |
| **02** | Root | [`BIG_DATA_LAB_2.ods`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/BIG_DATA_LAB_2.ods) | Spreadsheet | Advanced pivot tables, multi-attribute cross-tabulation, aggregations |
| **03** | Root | [`Supply_Chain_Data_Analysis.xlsx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Supply_Chain_Data_Analysis.xlsx) | Excel | Base supply chain logistics dataset (100 rows, 24 attributes) |
| **04** | Root | [`Supply_Chain_Data_Analysis_Solution_v3.xlsx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Supply_Chain_Data_Analysis_Solution_v3.xlsx) | Excel | Solved spreadsheet model with formulas, KPI summaries, and chart models |
| **05** | Root | [`Book1.xlsx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Book1.xlsx), [`nithin.xlsx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/nithin.xlsx), [`SRIPADH.ods`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/SRIPADH.ods) | Spreadsheets | Peer/supplementary benchmark runs on course problems |
| **06** | Root | [`question 7.ods`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/question%207.ods), [`question-8.ods`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/question-8.ods) | Spreadsheets | Targeted spreadsheet exercises on probability & frequencies |
| **07** | `task2/` | [`BIG_DATA_ASSGN -2.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/BIG_DATA_ASSGN%20-2.pdf) | PDF Spec | Problem statement for visual storytelling, Tableau & spreadsheets |
| **08** | `task2/` | [`Task_1_Supply_Chain_Data_Visualization_ReporT.docx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/Task_1_Supply_Chain_Data_Visualization_ReporT.docx) | Word Doc | 5 core chart analyses: Scatter, Histogram, Pie, Bar, Line |
| **09** | `task2/` | [`Tableau_Dashboard_Analysis.docx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/Tableau_Dashboard_Analysis.docx) | Word Doc | Comprehensive Titanic & Supply Chain Tableau dashboard report |
| **10** | `task2/` | [`dataset1.twb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/dataset1.twb), [`dataset2.twb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/dataset2.twb), [`dataset3.twb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/dataset3.twb) | Tableau | Interactive Tableau dashboards (Survival rates, carrier costs, logistics) |
| **11** | `task2/` | [`supply_chain_analysis.xlsx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/supply_chain_analysis.xlsx) | Excel | Integrated supply chain data formatted for Tableau consumption |
| **12** | `LAB3/` | [`Data_analysis.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB3/Data_analysis.ipynb) | Jupyter | 55-cell master visual analytics suite across 4 domains |
| **13** | `LAB3/` | [`supply_chain_data_analysis.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB3/supply_chain_data_analysis.ipynb) | Jupyter | Logistics-specific visual profiling (Doughnut, Pareto, Radar, Area) |
| **14** | `LAB3/` | [`supply_chain_data.csv`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB3/supply_chain_data.csv) | CSV | 100 rows, 24 columns supply chain benchmark dataset |
| **15** | `LAB3/` | [`titanic passenger list.csv`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB3/titanic%20passenger%20list.csv) | CSV | 1,309 rows Titanic demographics dataset |
| **16** | `LAB3/` | [`Summer_Olympic_medallists_1896-2008.xlsx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB3/Summer_Olympic_medallists_1896-2008.xlsx) | Excel | Historical sports analytics multi-level dataset |
| **17** | `LAB3/` | [`Research Excellence Framework 2014 Results_Pivoted.xlsx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB3/Research%20Excellence%20Framework%202014%20Results_Pivoted.xlsx) | Excel | Higher education academic research assessment dataset |
| **18** | `lab4/` | [`lab4_solution.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.py) | Python | 321 lines: Min-Max, Z-Score, Decimal Scaling, Binning, Reduction |
| **19** | `lab4/` | [`lab4_solution.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.ipynb) | Jupyter | Interactive preprocessing notebook with stepwise outputs |
| **20** | `lab4/` | [`transformed_supply_chain_data.csv`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/transformed_supply_chain_data.csv) | CSV | Transformed dataset including normalized columns and smoothed bins |
| **21** | `lab4/` | [`data_preprocessing_visualizations.png`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/data_preprocessing_visualizations.png) | Image | Multi-panel visual verification of distribution scaling and smoothing |
| **22** | `lab4/task2/`| [`lab4_task2_complete_analysis.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/task2/lab4_task2_complete_analysis.ipynb) | Jupyter | Financial ecosystem statistics analysis (IMPS bank performance & NFS chargebacks) |
| **23** | `LAB5/` | [`Northwind_Preprocessing_Final.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Final.ipynb) | Jupyter | 38-step relational integration, SQL joins, regex parsing, OHE, IQR |
| **24** | `LAB5/` | [`Northwind_Preprocessing_Professional_Report.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Professional_Report.pdf) | PDF Report | Exhaustive formal report on 7-table Northwind relational engineering |
| **25** | `LAB5/` | [`northwind.db`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/northwind.db) | SQLite DB | Full relational database file containing Northwind Traders tables |
| **26** | `LAB5/` | 8 Relational CSVs (`orders.csv`, `order_details.csv`, `products.csv`, etc.) | CSV Tables | Source relational tables producing 2,155 analytical order-line rows |
| **27** | `lab6/` | [`fim_test.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/fim_test.py) | Python | Baseline FIM pipeline: mlxtend Apriori vs FP-Growth on FoodMart |
| **28** | `lab6/` | [`FIM_Apriori_vs_FPGrowth_Report.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/FIM_Apriori_vs_FPGrowth_Report.pdf) | PDF Report | Formal evaluation of support grids, execution times, and sparsity |
| **29** | `lab6/` | [`foodmartFIM.txt`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/foodmartFIM.txt) | Text Data | 4,141 retail transactions, 1,559 items, average basket size 4.42 |
| **30** | `lab6/` | [`performance_comparison.png`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/performance_comparison.png) | Image | Benchmark plot: Apriori exponential runtime vs FP-Growth linearity |
| **31** | `Assignment_7_FIM_All_Files/` | [`assignment7_fim.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py) | Python | Advanced FIM: Dynamic Itemset Counting (DIC), Hash Apriori (DHP), ECLAT |
| **32** | `Assignment_7_FIM_All_Files/` | [`Assignment_7_FIM_Report_revised.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_Report_revised.pdf) | PDF Report | In-depth technical report on DIC checkpointing and vertical TID-sets |
| **33** | `Assignment_7_FIM_All_Files/` | [`assignment7_results.csv`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_results.csv) & `scalability.csv` | CSV | Empirical runtimes and candidate set sizes across support grids |
| **34** | `lab8/` | [`fim_algorithms.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py) | Python | 412 lines: Partitioning Apriori (SON), AClose (Closed), Pincer (MFI) |
| **35** | `lab8/` | [`FIM_Algorithms_Test.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/FIM_Algorithms_Test.ipynb) | Jupyter | 27 cells comparing 2-pass SON, closure operators, and bidirectional search |
| **36** | `lab8/` | [`FIM_Lab_Report.docx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/FIM_Lab_Report.docx) | Word Doc | Formal report on SON guarantees, minimal generators, and MFCS |
| **37** | `lab 9/` | [`online_shoppers_decision_tree.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/online_shoppers_decision_tree.ipynb) | Jupyter | Decision Tree (CART): Gini vs Entropy, CCP pruning, ROC-AUC |
| **38** | `lab 9/` | [`online_shoppers_decision_tree.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/online_shoppers_decision_tree.py) | Python | Production script for decision tree training and cost-complexity pruning |
| **39** | `lab 9/` | [`online_shoppers_intention.csv`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/online+shoppers+purchasing+intention+dataset/online_shoppers_intention.csv) | CSV | 12,330 sessions, 18 features, imbalanced target (15.5% purchase) |
| **40** | `lab10/` | [`01_naive_bayes_online_shoppers.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/01_naive_bayes_online_shoppers.ipynb) | Jupyter | Probabilistic classifiers: Gaussian, Multinomial, Bernoulli, Complement |
| **41** | `lab10/` | [`02_naive_bayes_supply_chain.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/02_naive_bayes_supply_chain.ipynb) | Jupyter | Multiclass logistics status prediction using Naive Bayes |
| **42** | `lab10/` | [`03_clustering_synthetic_dataset.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/03_clustering_synthetic_dataset.ipynb) | Jupyter | K-Means vs Hierarchical (Ward/Complete/Single), Dendrograms |
| **43** | `lab10/` | [`Lab_Report_Version_2.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/Lab_Report_Version_2.pdf) | PDF Report | Comparative report on Bayes formulations, synthetic clusters, DBI/CH |
| **44** | `lab11/` | [`online_shoppers_clustering.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb) | Jupyter | Advanced clustering: DBSCAN knee tuning, OPTICS, GMM, PCA/t-SNE |
| **45** | `lab11/` | [`Lab_11_Report_Version_2.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/Lab_11_Report_Version_2.pdf) | PDF Report | Formal e-commerce behavioral segmentation report |
| **46** | `lab11/` | [`report_agent1.md`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/report_agent1.md) | Markdown | DBSCAN specialization: Skewness rectification, 96% skew reduction |
| **47** | `lab11/` | [`report_agent2.md`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/report_agent2.md) | Markdown | Distance metrics specialization: 13 mathematical metrics analyzed |
| **48** | `lab11/` | `clustering_output/` & `advanced_clustering_outputs/` | Assets | 8 figures, grid searches, behavioral profile CSVs, t-SNE coordinate maps |
| **49** | `lab12/` | [`Big_Data_Lab12.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/Big_Data_Lab12.ipynb) | Jupyter | 17 master cells: SGA, TSP-GA, PySpark FIM/DBSCAN, Metric Clustering |
| **50** | `lab12/` | [`Lab12_Report_CS23I1034.md`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/Lab12_Report_CS23I1034.md) & `.pdf` | MD & PDF | 4-Task comprehensive lab report with complete empirical benchmarks |
| **51** | `lab12/` | [`task1_genetic_algorithm.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task1_genetic_algorithm.py) | Python | Simple Genetic Algorithm (SGA) continuous 5D Sphere optimizer |
| **52** | `lab12/` | [`task2_tsp_ga.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task2_tsp_ga.py) | Python | Permutation GA for TSP (OX1 crossover, 2-opt inversion mutation) |
| **53** | `lab12/` | [`task3_fpgrowth_dbscan.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task3_fpgrowth_dbscan.py) | Python | Parallel vs serial benchmarking for FP-Growth and spatial DBSCAN |
| **54** | `lab12/` | [`pyspark_fpgrowth_dbscan_cluster.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/pyspark_fpgrowth_dbscan_cluster.py) | Python | Production Apache Spark MLlib implementation for cluster execution |
| **55** | `lab12/` | [`task4_classification_clustering.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task4_classification_clustering.py) | Python | 6-Classifier benchmark across 4 datasets; K-Means/HAC metric clustering |
| **56** | `BIG_DATA_PROJECT` | `urban-service-intelligence/` | Full Repo | Enterprise NYC 311 PySpark platform: Snappy Parquet, ETL, FPGrowth, GBT |

---

## 3. EXHAUSTIVE LAB-BY-LAB DEEP DIVE

---

### LAB 1 & 2: SPREADSHEET FOUNDATIONS, PIVOT TABLES & DESCRIPTIVE STATISTICS
- **Workspace Assets:** [`BIG_DATA_LAB_1.ods`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/BIG_DATA_LAB_1.ods), [`BIG_DATA_LAB_2.ods`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/BIG_DATA_LAB_2.ods), [`Supply_Chain_Data_Analysis.xlsx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Supply_Chain_Data_Analysis.xlsx), [`Supply_Chain_Data_Analysis_Solution_v3.xlsx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Supply_Chain_Data_Analysis_Solution_v3.xlsx).
- **Core Concepts & Objectives:**
  - Establish ground-level data profiling using spreadsheet software.
  - Compute descriptive metrics: Mean ($\mu$), Median, Mode, Variance ($\sigma^2$), Standard Deviation ($\sigma$), Interquartile Range ($\text{IQR} = Q_3 - Q_1$).
  - Detect outliers via Tukey's fences: $[Q_1 - 1.5 \cdot \text{IQR}, Q_3 + 1.5 \cdot \text{IQR}]$.
  - Build multi-dimensional Pivot Tables aggregating revenue by `Product type`, `Location`, and `Shipping carrier`.
- **Inferences:**
  - Manual spreadsheet manipulation reveals data entry anomalies, non-standard text encodings, and missing values before automated scripting begins.
  - Pivot tables form the conceptual foundation for SQL aggregation (`GROUP BY`) and distributed Spark aggregation (`groupBy().agg()`).

---

### TASK 2 (ASSIGNMENT 2): VISUAL ANALYTICS, TABLEAU DASHBOARDS & STORYTELLING
- **Workspace Assets:** [`BIG_DATA_ASSGN -2.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/BIG_DATA_ASSGN%20-2.pdf), [`Task_1_Supply_Chain_Data_Visualization_ReporT.docx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/Task_1_Supply_Chain_Data_Visualization_ReporT.docx), [`Tableau_Dashboard_Analysis.docx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/Tableau_Dashboard_Analysis.docx), [`dataset1.twb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/dataset1.twb), [`dataset2.twb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/dataset2.twb), [`dataset3.twb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/task2/dataset3.twb).
- **5 Core Visualizations Analyzed:**
  1. **Scatter Plot (Price vs Number of Products Sold):** Examined price elasticity. Displayed wide scatter ($R^2 \approx 0$), proving sales volume was decoupled from price.
  2. **Histogram (Product Prices):** Segmented into 5 equal-width bins; identified bi-modal pricing tiers (\$10–\$30 and \$70–\$95).
  3. **Pie / Doughnut (Defect Rates by Inspection):** Segmented quality outcomes across Pass, Fail, and Pending audits.
  4. **Horizontal Bar (Titanic Passenger Class Distribution):** Total 1,309 passengers: Class 3 = 709 (54.2%), Class 1 = 323 (24.7%), Class 2 = 277 (21.2%).
  5. **Cross-Tab Survival Heatmap:** Sex vs Class survival: Female Class 1 achieved 96.8% survival, whereas Male Class 3 had 13.5% survival.
- **Key Inferences:**
  - Visual storytelling must always pair absolute metrics (raw counts) with normalized rates (percentages).
  - Effective dashboards require strict visual hierarchy: Top KPI cards $\to$ distribution charts $\to$ granular drill-down tables.

---

### LAB 3: MULTI-DOMAIN EXPLORATORY DATA ANALYSIS & VISUAL ANALYTICS
- **Workspace Assets:** [`Data_analysis.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB3/Data_analysis.ipynb), [`supply_chain_data_analysis.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB3/supply_chain_data_analysis.ipynb), datasets (`supply_chain_data.csv`, `titanic passenger list.csv`, `Summer_Olympic_medallists_1896-2008.xlsx`, `Research Excellence Framework 2014 Results_Pivoted.xlsx`).
- **Advanced Visual Framework:**
  1. **Doughnut Charts:** Clean categorical composition with high data-to-ink ratio.
  2. **Pareto Charts (80/20 Rule):** Combined bar chart of descending frequencies with cumulative percentage line. Identified that top 20% of product SKUs generated 78.4% of total revenue.
  3. **Radar / Spider Charts:** Multi-axial comparison across normalized operational vectors (e.g., comparing Shipping Carriers across Cost, Lead Time, Defect Rate, and Volume).
  4. **Stacked Area Charts:** Longitudinal volume trends across categories over time.
- **Key Inferences:**
  - Pareto analysis enables rapid prioritization in resource-constrained environments (e.g. hackathons).
  - Radar charts instantly expose trade-offs (e.g. Carrier B offers fastest transit but highest defect variance).

---

### LAB 4: DATA PREPROCESSING, TRANSFORMATIONS, NORMALIZATION & BINNING
- **Workspace Assets:** [`lab4_solution.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.py), [`lab4_solution.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.ipynb), [`transformed_supply_chain_data.csv`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/transformed_supply_chain_data.csv), [`data_preprocessing_visualizations.png`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/data_preprocessing_visualizations.png).
- **Mathematical Formulations & Implementation:**
  1. **Min-Max Normalization to $[0, 1]$:**
     $$v' = \frac{v - \min(A)}{\max(A) - \min(A)}$$
     - Applied to `Manufacturing costs` ($[10.03, 99.41] \to [0.0, 1.0]$).
  2. **Z-Score Normalization (Standardization):**
     $$v' = \frac{v - \mu}{\sigma}$$
     - Applied to `Revenue generated` ($\mu = 5776.05, \sigma = 2732.84 \to \mu' = 0.0, \sigma' = 1.0$).
  3. **Decimal Scaling Normalization:**
     $$v' = \frac{v}{10^j} \quad \text{where } j = \lceil \log_{10}(\max(|A|)) \rceil \text{ such that } \max(|v'|) < 1$$
     - Applied to `Price`: $\max(Price) = 99.20 \implies j = 2 \implies \text{divisor} = 100 \implies v' \in [0.016, 0.992]$.
  4. **Equal-Frequency Binning & Smoothing:**
     - Divides $N$ sorted records into $k$ equal bins each having $N/k$ records.
     - *Mean Smoothing:* $x_{i} \leftarrow \bar{x}_{bin}$.
     - *Median Smoothing:* $x_{i} \leftarrow \text{median}(x_{bin})$.
     - *Boundary Smoothing:* Replace with whichever boundary ($\min(bin)$ or $\max(bin)$) is closer:
       $$x_i \leftarrow \begin{cases} \min(bin) & \text{if } |x_i - \min| \le |x_i - \max| \\ \max(bin) & \text{otherwise} \end{cases}$$
  5. **Data Reduction via Aggregation:**
     - Aggregated order-level records by `Lead time` computing mean `Order quantities`, reducing rows by 68% while preserving trend fidelity.

---

### LAB 5: RELATIONAL DATA INTEGRATION, SQL PIPELINES & FEATURE ENGINEERING
- **Workspace Assets:** [`Northwind_Preprocessing_Final.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Final.ipynb), [`Northwind_Preprocessing_Professional_Report.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Professional_Report.pdf), [`northwind.db`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/northwind.db), 8 relational CSV tables.
- **The 12-Step Relational Engineering Architecture:**
  1. *Schema Audit:* Identified missing values in `employees.reportsTo` (11.11%) and `orders.shippedDate` (2.53% — 21 of 830 orders).
  2. *Type Casting:* String timestamps cast to `datetime64[ns]`.
  3. *High-Nullity Pruning:* Dropped unrecoverable sparse attributes.
  4. *Multi-Table Join:* Left outer joins from `order_details` (primary fact) to `orders`, `products`, `customers`, `employees`, and `shippers` $\implies$ **2,155 analytical records**.
  5. *Selective Projection:* Retained 23 relevant analytical features, discarding redundant surrogate keys.
  6. *Domain Imputation:* Imputed missing `shippedDate` using order-specific median transit duration ($\text{orderDate} + 7 \text{ days}$).
  7. *Regex Extraction:* Extracted numerical volume from messy packaging strings in `quantityPerUnit` (e.g. `"24 - 12 oz bottles"` $\to 288.0$).
  8. *IQR Outlier Capping:* Clamped `freight` and `line_total` at $Q_3 + 1.5 \cdot \text{IQR}$.
  9. *Feature Engineering:*
     $$\text{gross\_sales} = \text{unitPrice} \times \text{quantity}$$
     $$\text{net\_sales} = \text{gross\_sales} \times (1 - \text{discount})$$
     $$\text{shipping\_delay\_days} = \max(0, \text{shippedDate} - \text{requiredDate})$$
  10. *Skewness Correction:* `gross_sales` had Fisher-Pearson skewness $g_1 = 4.82$. Applied $x \mapsto \log(1 + x)$, collapsing skewness to $0.41$.
  11. *Ordinal Encoding:* Encoded ordered rank columns.
  12. *One-Hot Encoding:* Encoded nominal attributes (`shipVia`, `categoryName`) with dummy drop.

---

### LAB 6: FREQUENT ITEMSET MINING (FIM) — APRIORI VS FP-GROWTH
- **Workspace Assets:** [`fim_test.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/fim_test.py), [`FIM_Apriori_vs_FPGrowth_Report.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/FIM_Apriori_vs_FPGrowth_Report.pdf), [`foodmartFIM.txt`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/foodmartFIM.txt), [`performance_comparison.png`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/performance_comparison.png).
- **Core Formulations:**
  - Support: $\text{Supp}(X) = \frac{|\{t \in T \mid X \subseteq t\}|}{N}$
  - Confidence: $\text{Conf}(X \Rightarrow Y) = \frac{\text{Supp}(X \cup Y)}{\text{Supp}(X)}$
  - Lift: $\text{Lift}(X \Rightarrow Y) = \frac{\text{Supp}(X \cup Y)}{\text{Supp}(X) \times \text{Supp}(Y)}$
- **Mechanics Compared:**
  - **Apriori:** Level-wise breadth-first search. Generates candidates $C_k = L_{k-1} \Join L_{k-1}$, prunes using subset frequency principle, and performs a full database scan per pass. Incurs exponential candidate explosion.
  - **FP-Growth:** Builds a compact **FP-Tree** via 2 scans: Pass 1 discovers frequent items and orders them by frequency; Pass 2 inserts transactions into prefix paths with linked header lists. Mines recursively via conditional pattern bases without candidate generation.
- **Empirical Findings:**
  - FoodMart dataset is ultra-sparse: 4,141 transactions, 1,559 items, maximum single-item support is **0.60%** (25 transactions).
  - FP-Growth outperformed Apriori by **$5\times$ to $18\times$** in execution time. Apriori runtime grows exponentially as support threshold drops, while FP-Growth scales gracefully.

---

### LAB 7 (ASSIGNMENT 7): ADVANCED FIM — DYNAMIC ITEMSET COUNTING, HASH APRIORI & ECLAT
- **Workspace Assets:** [`assignment7_fim.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py), [`Assignment_7_FIM_Report_revised.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_Report_revised.pdf), CSV result benchmarks.
- **Three Breakthrough FIM Architectures:**
  1. **Dynamic Itemset Counting (DIC; Brin et al.):**
     - Relaxes strict pass boundaries by checking itemset frequency at intermediate checkpoints (every $M$ transactions).
     - Transition states: Dashed Circle (candidate) $\to$ Solid Circle (confirmed infrequent) $\to$ Dashed Box (suspected frequent) $\to$ Solid Box (confirmed frequent).
     - Allows new itemsets to be added dynamically before completing a full pass over the dataset, completing mining in roughly $1.5$ full passes.
  2. **Hash-Based Variant of Apriori (Direct Hashing & Pruning - DHP; Park et al.):**
     - During pass $k$, computes hash value for all $(k+1)$-subsets using hash function $h(x, y) = ((x \cdot 10) + y) \pmod B$ and increments hash bucket counters.
     - When constructing $C_{k+1}$, prunes any candidate whose hash bucket count is below $\text{min\_count}$.
     - Achieved **$72\%$ candidate pruning** in $C_2$, removing the main memory bottleneck of classic Apriori.
  3. **Vertical Transaction Variant (ECLAT; Zaki):**
     - Inverts horizontal transactions $\langle \text{TID}, \{i_1, i_2\}\rangle$ into vertical TID-sets $\langle \text{Item}, \{t_1, t_2\}\rangle$.
     - Computes support via set intersection:
       $$\text{TID-set}(X \cup Y) = \text{TID-set}(X) \cap \text{TID-set}(Y)$$
     - **Requires zero database scans** after the initial inverted index construction. Extremely fast in RAM using bitwise operations.

---

### LAB 8: ADVANCED FIM ARCHITECTURES — SON PARTITIONING, ACLOSE & PINCER SEARCH
- **Workspace Assets:** [`fim_algorithms.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py), [`FIM_Algorithms_Test.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/FIM_Algorithms_Test.ipynb), [`FIM_Lab_Report.docx`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/FIM_Lab_Report.docx).
- **Core Algorithms & Theorems:**
  1. **Partitioning Apriori (SON Algorithm; Savasere et al., 1995):**
     - **Fundamental Theorem:** An itemset $X$ can only be frequent globally in database $D$ (support $\ge s$) if it is locally frequent in *at least one* partition $P_i$ with local threshold $s \cdot |P_i|$.
     - **Pass 1:** Partition dataset into $p$ chunks in RAM. Mine local frequent itemsets $L^{(i)}$ independently. Aggregate candidate set $C = \bigcup_{i=1}^p L^{(i)}$.
     - **Pass 2:** Perform a single global pass across $D$ to count true supports for candidates in $C$.
     - **Guaranteed exactly 2 database scans**, regardless of transaction length! Ideal for distributed MapReduce and PySpark.
  2. **AClose (Pasquier et al., 1999) — Frequent Closed Itemsets (FCIs):**
     - An itemset $X$ is *closed* if no proper superset $Y \supset X$ has the same support ($\sigma(Y) = \sigma(X)$).
     - Closure Operator: $c(X) = \bigcap_{t \in \mathcal{T}(X)} t$.
     - Mines **minimal generators** and computes closures. Compresses frequent itemset volume by $80\% - 95\%$ with zero loss of association rule information.
  3. **Pincer Search (Lin & Dunham, 1998) — Maximal Frequent Itemsets (MFIs):**
     - An itemset is *maximal frequent* if no proper superset is frequent.
     - Bidirectional Search: Bottom-Up candidate generation combined with Top-Down search using the **Maximum Frequent Candidate Set (MFCS)**.
     - When a maximal itemset is confirmed from the top down, *all its subsets are automatically confirmed frequent* and pruned from bottom-up processing, preventing exponential subset evaluations.

---

### LAB 9: SUPERVISED MACHINE LEARNING — DECISION TREES (CART), PRUNING & IMBALANCE
- **Workspace Assets:** [`online_shoppers_decision_tree.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/online_shoppers_decision_tree.ipynb), [`online_shoppers_decision_tree.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/online_shoppers_decision_tree.py), [`online_shoppers_intention.csv`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/online+shoppers+purchasing+intention+dataset/online_shoppers_intention.csv).
- **Mathematical Formulations:**
  - Gini Impurity: $I_G(t) = 1 - \sum_{i=1}^C p(i|t)^2$
  - Entropy: $H(t) = -\sum_{i=1}^C p(i|t) \log_2 p(i|t)$
  - Split selection maximizes information gain: $\Delta I = I(t) - \frac{N_L}{N} I(t_L) - \frac{N_R}{N} I(t_R)$.
- **Handling Severe Class Imbalance & Overfitting:**
  - Target `Revenue`: 84.5% False, 15.5% True. Unconstrained trees scored 100% training accuracy but suffered on test data (positive-class precision dropped to 42.1%).
  - **Minimal Cost-Complexity Pruning (CCP):**
    $$R_\alpha(T) = R(T) + \alpha |T|$$
    - Extracted $\alpha$-path via `cost_complexity_pruning_path()`.
    - Tuned $\alpha = 0.0014$ via 5-fold Stratified Cross-Validation, preventing over-branching.
  - **Class Weight Balancing:** Applied `class_weight='balanced'` setting sample weights inversely proportional to class frequencies:
    $$w_j = \frac{N}{C \cdot N_j}$$
    - Minority purchase recall surged from $49.2\%$ to **$76.8\%$**.

---

### LAB 10: PROBABILISTIC CLASSIFICATION (NAIVE BAYES) & SYNTHETIC CLUSTERING
- **Workspace Assets:** [`01_naive_bayes_online_shoppers.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/01_naive_bayes_online_shoppers.ipynb), [`02_naive_bayes_supply_chain.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/02_naive_bayes_supply_chain.ipynb), [`03_clustering_synthetic_dataset.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/03_clustering_synthetic_dataset.ipynb), [`Lab_Report_Version_2.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/Lab_Report_Version_2.pdf).
- **Probabilistic Formulations & Comparison:**
  - Maximum A Posteriori (MAP) Rule: $\hat{y} = \arg\max_c \left[ \log P(Y = c) + \sum_{j=1}^d \log P(x_j \mid Y = c) \right]$
  - **GaussianNB:** $P(x_j \mid Y = c) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x_j - \mu)^2}{2\sigma^2} \right)$. Includes `var_smoothing` to avoid zero-variance breakdown.
  - **MultinomialNB:** Models discrete counts with Laplace smoothing $\alpha = 1$.
  - **BernoulliNB:** Models binary indicator features $p^{x}(1-p)^{1-x}$.
  - **ComplementNB:** Parameterized using data from all classes *except* class $c$, specifically countering class imbalance.
- **Empirical Results on E-Commerce Dataset:**
  - **BernoulliNB:** Highest accuracy (**85.04%**), precision (**51.73%**), and ROC-AUC (**0.8636**).
  - **Tuned GaussianNB:** Highest positive-class recall (**74.35%**) — best for capturing potential buyers.
  - **ComplementNB:** Balanced trade-off (64.56% accuracy, **73.04% recall**).
- **Synthetic Clustering Benchmarking:**
  - **K-Means:** Minimizes $\text{WCSS} = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$. Assumes isotropic, spherical clusters.
  - **Hierarchical Agglomerative Clustering (HAC):** Compared Single Linkage (chaining failure), Complete Linkage (tight clusters), Average Linkage, and Ward's Linkage (minimizes variance growth).
  - Evaluated via Silhouette Score, Davies-Bouldin Index (lower is better), and Calinski-Harabasz Index (higher is better).

---

### LAB 11: ADVANCED CUSTOMER SEGMENTATION, DENSITY-BASED CLUSTERING (DBSCAN/OPTICS/GMM) & METRIC SPACES
- **Workspace Assets:** [`online_shoppers_clustering.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb), [`Lab_11_Report_Version_2.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/Lab_11_Report_Version_2.pdf), [`report_agent1.md`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/report_agent1.md), [`report_agent2.md`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/report_agent2.md), outputs in `clustering_output/` and `advanced_clustering_outputs/`.
- **Breakthrough Discoveries & Theoretical Formulations:**
  1. **The Singularity Effect & Scale Swallowing:**
     - Raw features exhibited skewness up to **7.58** (`Informational_Duration`) and scale ranges from $[0, 0.2]$ (`BounceRates`) to $[0, 63,973.5]$ (`ProductRelated_Duration`).
     - In unscaled Euclidean distance, `ProductRelated_Duration` contributed **$99.9999\%$** of distance calculations, swallowing all other signals and collapsing $90\%$ of points into a high-density singularity.
     - Solution: $\log(1 + x)$ reduced skewness by up to **96%** (`ProductRelated_Duration` $7.26 \to -1.42$, `Administrative_Duration` $5.62 \to 0.26$), followed by `StandardScaler`.
  2. **DBSCAN Parameter Optimization:**
     - Parameter $k = 2 \cdot D - 1 = 19$. Plotted sorted 19-NN distance graph and detected knee at $\varepsilon = 1.6$, `min_samples = 15`.
     - Result: Identified 3 core behavioral clusters and flagged anomalous bot/crawler traffic as noise (-1).
  3. **Beyond-Classroom Algorithms:**
     - **OPTICS:** Generated reachability plots revealing hierarchical sub-clusters invisible to global $\varepsilon$ DBSCAN.
     - **Gaussian Mixture Models (GMM):** Soft probabilistic clustering using EM algorithm. Lowest BIC achieved at $K = 4$ full covariance components.
  4. **Behavioral Profile of Discovered Segments:**
     - *Cluster 0 (Window Shoppers, 68.2%):* Low duration, low page count, conversion rate $4.2\%$.
     - *Cluster 1 (High-Intent Buyers, 18.4%):* High page values (mean \$38.4), low bounce rate ($0.002$), conversion rate **$78.9\%$**!
     - *Cluster -1 (Noise / Bot Anomalies, 4.8%):* Extreme duration without interaction, conversion rate $0.0\%$.

---

### LAB 12: DISTRIBUTED BIG DATA & METAHEURISTICS — GENETIC ALGORITHMS, PYSPARK FIM/DBSCAN & MULTI-CLASSIFIER CLUSTERING
- **Workspace Assets:** [`Big_Data_Lab12.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/Big_Data_Lab12.ipynb), [`Lab12_Report_CS23I1034.md`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/Lab12_Report_CS23I1034.md), [`task1_genetic_algorithm.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task1_genetic_algorithm.py), [`task2_tsp_ga.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task2_tsp_ga.py), [`task3_fpgrowth_dbscan.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task3_fpgrowth_dbscan.py), [`pyspark_fpgrowth_dbscan_cluster.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/pyspark_fpgrowth_dbscan_cluster.py), [`task4_classification_clustering.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task4_classification_clustering.py).

#### Task 1: Simple Genetic Algorithm (5-Variable Continuous Optimization)
- **Objective Function:** Continuous 5D Sphere Function:
  $$\min_{\mathbf{x}} f(\mathbf{x}) = \sum_{i=1}^5 x_i^2, \quad x_i \in [-5.12, 5.12]$$
- **Chromosome Architecture:** 16 bits per variable $\implies$ 80-bit binary chromosome.
  $$x_i = x_{\min} + \frac{\text{int}(b_i)}{2^{16} - 1}(x_{\max} - x_{\min})$$
- **Fitness Formulation:** $F(\mathbf{x}) = \frac{1}{1 + f(\mathbf{x})}$.
- **Operators & Parameters:**
  - *Selection:* Tournament Selection ($k = 3$) outperformed Roulette Wheel by preserving selection pressure without premature convergence.
  - *Crossover:* Two-Point Crossover with $p_c = 0.85$.
  - *Mutation:* Bit-flip mutation with rate $p_m = \frac{1}{80} = 0.0125$.
  - *Elitism:* Preserved top 2 individuals into next generation.
- **Convergence:** Fitness surged from $0.038$ to **$0.9998$** at generation 150 ($f(\mathbf{x}) = 0.00018$).

#### Task 2: Traveling Salesperson Problem (Combinatorial Permutation GA)
- **Problem Formulation:** Hamiltonian cycle of minimal Euclidean tour distance across $N = 20$ cities:
  $$\min \sum_{i=1}^{N-1} d(c_i, c_{i+1}) + d(c_N, c_1)$$
- **Permutation-Preserving Operators:**
  - Standard crossover produces invalid tours (duplicate and missing cities).
  - *Order Crossover 1 (OX1):* Preserves relative ordering. Copies random slice from Parent 1, fills remaining positions using ordered cities from Parent 2.
  - *Partially Mapped Crossover (PMX):* Uses position mappings to preserve absolute city coordinates.
  - *Inversion Mutation (2-Opt Mutation):* Reverses a sub-sequence of cities, untangling crossing tour edges.
- **Result:** Tour distance reduced from random initial $1,184.2 \text{ km}$ to **$539.6 \text{ km}$** (a **$54.43\%$ reduction**).

#### Task 3: Distributed Big Data Scalability — Serial vs PySpark MLlib
- **Distributed FP-Growth (`pyspark.ml.fpm.FPGrowth`):**
  - Distributes item frequency counting across cluster executors and builds parallel projected trees.
  - At $10^6$ transactions, serial `mlxtend` crashed with `MemoryError`, while PySpark completed in 42 seconds with linear scaling.
- **Distributed Spatial Partitioning DBSCAN:**
  - Partitions spatial coordinates into grid bounding boxes with overlap margin $\varepsilon$. Clusters computed locally; cross-boundary clusters resolved via **Disjoint Set Union (DSU)**.

#### Task 4: Multi-Classifier Evaluation & Performance Metric Space Clustering
- **6 Classifiers:** Decision Tree, Gaussian Naive Bayes, KNN, SVM-RBF, Random Forest, Logistic Regression.
- **7 Evaluation Metrics:** Accuracy, Precision (Macro), Recall (Macro), F1 (Macro), ROC-AUC (OvR), Training Time (ms), Inference Time (ms).
- **4 Datasets:** Iris, Breast Cancer Wisconsin, Wine, Synthetic Complex Non-Linear.
- **Metric Clustering:**
  - Standardized the $6 \times 7$ metric matrix and clustered algorithms into performance tiers:
    - *Tier 1 (High Performance Ensembles):* Random Forest & SVM-RBF (highest ROC-AUC $\ge 0.98$, balanced F1, higher training time).
    - *Tier 2 (Ultra-Fast Linear Baselines):* Logistic Regression & GaussianNB (lowest latency $\le 2 \text{ ms}$, strong baselines on linear features).
    - *Tier 3 (Localized / Tree Baselines):* KNN & Decision Tree (sensitive to noise and high dimensionality).

---

### BIG_DATA_PROJECT: URBAN SERVICE INTELLIGENCE (NYC 311 PYSPARK BATCH/STREAMING PIPELINE)
- **Workspace Location:** `BIG_DATA_PROJECT/urban-service-intelligence/`
- **System Architecture & Key Features:**
  - Scalable enterprise Big Data platform on NYC 311 Open Data (`erm2-nwe9`, millions of records).
  - Pinned Stack: Python 3.10, PySpark 3.5.3 (local cluster mode), Java 17, Parquet, Pytest, Streamlit.
  - Data Contracts: Formal JSON schema validation (`schema_erm2-nwe9.json`) and automated data quality audits (`etl_quality.json`).
  - Optimized Storage: Partitioned Snappy Parquet by `year=YYYY/month=MM` for partition pruning.
  - Distributed Pattern Mining: PySpark MLlib `FPGrowth` discovering co-occurring complaints across municipal agencies.
  - Predictive Service Resolution: PySpark MLlib Pipeline (`VectorAssembler`, `StringIndexer`, `StandardScaler`, `GBTRegressor`, `RandomForestRegressor`) predicting incident closure duration in hours.

---

## 4. MASTER CROSS-LAB ALGORITHMIC TAXONOMY & DECISION MATRIX

| Algorithm | Paradigm | Supervised? | Time Complexity | Memory | Key Parameters | When to Deploy in Hackathon | Failure Modes & Risks |
|---|---|---|---|---|---|---|---|
| **Apriori** | Association | No | $O(2^d \cdot N)$ | High | `min_support`, `min_confidence` | Small retail basket analysis | Candidate explosion if $\text{supp} < 1\%$ |
| **FP-Growth** | Association | No | $O(N \cdot \text{avg\_len})$ | Low / Med | `min_support`, `min_confidence` | Large transaction feeds, co-occurrence | High branching if items are disjoint |
| **SON (Partitioning)** | Association | No | $O(N)$ (2 scans) | Low | `partitions`, `min_support` | Distributed / chunked streaming FIM | Skewed partition distributions |
| **AClose / Pincer** | Association | No | $O(\text{MFI\_search})$ | Very Low | `min_support` | Dense transactions with long itemsets | High overhead on sparse data |
| **Decision Tree (CART)**| Classification | Yes | $O(d \cdot N \log N)$ | Low | `max_depth`, `min_samples_leaf`, `ccp_alpha` | Interpretable baseline, feature ranking | Severe overfitting without pruning |
| **Random Forest** | Classification | Yes | $O(M \cdot d \cdot N \log N)$ | Moderate | `n_estimators`, `max_depth` | High tabular accuracy, robust to noise | Higher latency on micro-batches |
| **Gaussian NB** | Classification | Yes | $O(d \cdot N)$ | Ultra Low | `var_smoothing` | Ultra-fast continuous baseline, streaming | Breaks if features are correlated |
| **Bernoulli / Mult NB**| Classification | Yes | $O(d \cdot N)$ | Ultra Low | `alpha` (Laplace) | Text, binary flags, sparse features | Zero probability on unobserved words |
| **Complement NB** | Classification | Yes | $O(d \cdot N)$ | Ultra Low | `alpha` | Heavily imbalanced discrete datasets | Requires non-negative counts |
| **K-Means** | Clustering | No | $O(k \cdot N \cdot d \cdot I)$ | Low | $k$, `init='k-means++'` | Spherical, equal-variance clusters | Fails on non-convex arbitrary densities |
| **DBSCAN** | Clustering | No | $O(N \log N)$ to $O(N^2)$ | Moderate | $\varepsilon$, `min_samples`, `metric` | Arbitrary shapes, automatic noise detection | Collapses if clusters have variable densities |
| **OPTICS** | Clustering | No | $O(N \log N)$ to $O(N^2)$ | Moderate | `min_samples`, `xi` | Varying cluster densities | Computationally slower than DBSCAN |
| **GMM (EM)** | Clustering | No | $O(k \cdot N \cdot d^3 \cdot I)$ | Moderate | $k$, `covariance_type` | Soft probabilistic cluster assignments | Singularities if component collapses |
| **HAC (Ward)** | Clustering | No | $O(N^2 \log N)$ | High $O(N^2)$ | `n_clusters`, `linkage` | Hierarchical dendrograms, small $N$ | Memory explosion if $N > 15,000$ |
| **Genetic Algorithm** | Optimization | N/A | $O(G \cdot P \cdot d)$ | Low | $P$, $p_c, p_m$, elitism | Non-convex, discrete, NP-hard routing | Premature convergence without diversity |

---

## 5. THE 13 DISTANCE & DISSIMILARITY METRICS FRAMEWORK

Extracted directly from Lab 11 research (`report_agent2.md`):

### Formal Metric Axioms
A function $d: X \times X \to [0, \infty)$ is a strict mathematical metric iff:
1. **Non-Negativity:** $d(\mathbf{u}, \mathbf{v}) \ge 0$
2. **Identity of Indiscernibles:** $d(\mathbf{u}, \mathbf{v}) = 0 \iff \mathbf{u} = \mathbf{v}$
3. **Symmetry:** $d(\mathbf{u}, \mathbf{v}) = d(\mathbf{v}, \mathbf{u})$
4. **Subadditivity (Triangle Inequality):** $d(\mathbf{u}, \mathbf{w}) \le d(\mathbf{u}, \mathbf{v}) + d(\mathbf{v}, \mathbf{w})$

### Complete Metric Taxonomy
| Metric Name | Mathematical Formula | Data Type Suitability | Metric Axiom Status | Key Properties & Vulnerabilities |
|---|---|---|---|---|
| **Euclidean ($L_2$)** | $\|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum (u_i - v_i)^2}$ | Continuous numerical | Strict Metric | Rotation invariant; highly sensitive to scale discrepancies |
| **Manhattan ($L_1$)** | $\|\mathbf{u} - \mathbf{v}\|_1 = \sum |u_i - v_i|$ | Continuous numerical | Strict Metric | Robust to outliers; represents grid movement |
| **Minkowski ($L_p$)** | $\|\mathbf{u} - \mathbf{v}\|_p = \left(\sum |u_i - v_i|^p\right)^{1/p}$ | Continuous numerical | Strict Metric ($p \ge 1$) | Generalizes $L_1$ and $L_2$; $p < 1$ violates triangle inequality |
| **Chebyshev ($L_\infty$)** | $\|\mathbf{u} - \mathbf{v}\|_\infty = \max_i |u_i - v_i|$ | Continuous numerical | Strict Metric | Governed by maximum coordinate; degenerates in high-d nominal data |
| **Cosine Distance** | $d_{\cos} = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ | High-dim sparse vectors | **Semi-Metric (Violates Triangle Ineq!)** | Scale-invariant; measures angle only; can fail triangle test ($0.58 < 1.0$) |
| **Angular Distance** | $d_{\text{ang}} = \frac{\arccos(\cos \theta)}{\pi}$ | Directional vectors | Strict Metric | Normalized $[0, 1]$; restores metric validity to cosine space |
| **Mahalanobis** | $\sqrt{(\mathbf{u} - \mathbf{v})^T \mathbf{\Sigma}^{-1} (\mathbf{u} - \mathbf{v})}$ | Multi-normal continuous | Strict Metric (if $\mathbf{\Sigma} \succ 0$) | Scale-invariant; accounts for cross-feature covariance |
| **Canberra** | $\sum_{i=1}^d \frac{|u_i - v_i|}{|u_i| + |v_i|}$ | Non-negative ratios | Strict Metric | Highly sensitive to small values near zero |
| **Bray-Curtis** | $\frac{\sum |u_i - v_i|}{\sum (u_i + v_i)}$ | Ecological count data | Semi-Metric | Bounded $[0, 1]$; sensitive to absolute abundance |
| **Hamming** | $\frac{1}{d} \sum \mathbb{I}(u_i \ne v_i)$ | Categorical / Binary | Strict Metric | Symmetric; heavily diluted by mutual absences in sparse one-hot data |
| **Jaccard Distance** | $d_J = 1 - \frac{|A \cap B|}{|A \cup B|}$ | Asymmetric Binary / Sets | Strict Metric | Ignores negative-negative matches ($0-0$); ideal for sparse baskets |
| **Dice Dissimilarity** | $d_D = 1 - \frac{2|A \cap B|}{|A| + |B|}$ | Binary sets | Semi-Metric | Weights joint presence twice as heavily as Jaccard |
| **Gower's Coefficient** | $d_G(\mathbf{u}, \mathbf{v}) = \frac{\sum w_k s_k(u_k, v_k)}{\sum w_k}$ | **Heterogeneous Mixed Data** | Metric (under square root) | Combines continuous ranges, ordinal ranks, and nominal matches |

---

## 6. HACKATHON LIVE / REAL-TIME DATASET OPERATIONAL PLAYBOOK

### PATTERN A: DIRECTORY WATCHER & MICRO-BATCH STREAMING FEEDS
When live data is delivered via a folder where files arrive dynamically:
```python
import os, time, glob
import pandas as pd

def stream_folder_watcher(watch_dir="stream_data/", poll_interval=2.0):
    processed_files = set()
    print(f"[*] Watching {watch_dir} for incoming streaming batches...")
    while True:
        current_files = set(glob.glob(os.path.join(watch_dir, "*.csv")))
        new_files = sorted(list(current_files - processed_files))
        for nf in new_files:
            print(f"[+] Processing live micro-batch: {nf}")
            df_batch = pd.read_csv(nf)
            process_live_batch(df_batch)
            processed_files.add(nf)
        time.sleep(poll_interval)

def process_live_batch(df):
    print(f"    --> Ingested {len(df)} records. Shape: {df.shape}")
```

### PATTERN B: ONLINE & INCREMENTAL MACHINE LEARNING (PARTIAL FIT)
When data arrives continuously, retraining models from scratch creates latency spikes. Use incremental models with `partial_fit`:
```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Initialize streaming models
scaler = StandardScaler()
model = SGDClassifier(loss='log_loss', random_state=42)
all_classes = [0, 1] # MUST specify all possible target classes upfront

def online_train_step(X_batch, y_batch):
    scaler.partial_fit(X_batch)
    X_scaled = scaler.transform(X_batch)
    model.partial_fit(X_scaled, y_batch, classes=all_classes)
    
def online_predict_step(X_new):
    X_scaled = scaler.transform(X_new)
    return model.predict_proba(X_scaled)[:, 1]
```

### PATTERN C: RAPID TRIAGE OF FRESH HIGH-VELOCITY DATASETS (FIRST 10 MINUTES)
1. **Minute 0–2 (Profiling):** Run `df.info()`, `df.isnull().sum()`, `df.describe()`, check for duplicate rows.
2. **Minute 2–4 (Cleaning & Rectification):** Impute continuous nulls with median, nominals with `"Unknown"`. If positive feature skewness $> 1.5$, apply $\log(1+x)$.
3. **Minute 4–7 (Baseline Pipeline Execution):**
   - If Classification: Random Forest / Logistic Regression with 5-fold Stratified CV.
   - If Clustering: Standardize, compute $k$-NN knee, execute DBSCAN / K-Means.
   - If FIM: One-hot encode via `TransactionEncoder`, run FP-Growth.
4. **Minute 7–10 (Diagnostics & Visualization):** Output Confusion Matrix, ROC-AUC curve, or 2D PCA cluster visualization.

---


---

## 7. EXHAUSTIVE CODEBASE INVENTORY & ARCHITECTURAL PATTERNS

Every single code file across the workspace has been directly analyzed (68 standalone Python .py scripts and 12 Jupyter .ipynb workbooks). Below is the exact architectural and implementation map of the primary code modules:

### 1. Data Preprocessing & Relational Engineering Codebases
- **lab4/lab4_solution.py (321 lines):**
  - load_data(filepath): Validates file presence, loads CSV, sets clean pandas display options.
  - min_max_normalize(series, new_min=0, new_max=1): Implements formula ' = \frac{v - \min}{\max - \min}(\Delta) + new\_min$. Includes divide-by-zero check if $\max == \min$.
  - z_score_normalize(series): Implements ' = \frac{v - \bar{x}}{\sigma}$. Falls back to zeros if $\sigma == 0$.
  - decimal_scaling_normalize(series): Computes  = \lceil \log_{10}(\max(|v|)) \rceil$ and divides by ^j$.
  - equal_frequency_binning(series, num_bins=4): Uses pd.qcut(series, q=num_bins, duplicates='drop').
  - smooth_by_mean(), smooth_by_median(), smooth_by_boundaries(): Replaces elements in each quantile bin with bin mean, bin median, or the nearest boundary value ($|v - \min| \le |v - \max|$).
  - 
educe_data(df, group_col='Lead time', agg_col='Order quantities'): Performs aggregation grouping, printing before/after dimension reduction ratios.
- **LAB5/Northwind_Preprocessing_Final.ipynb (38 cells):**
  - Connects to SQLite 
orthwind.db and loads 8 relational tables.
  - Executes left outer joins anchored at order_details $\to$ 2,155 merged rows.
  - Handles missing shippedDate using dynamic median shipping duration (orderDate + 7 days).
  - Regex extraction: 
'(\d+(?:\.\d+)?)\s*(?:oz|ml|g|kg|lbs)?' to parse continuous quantity from unstructured strings.
  - Computes Fisher-Pearson skewness; applies 
p.log1p on skewed revenue features.

### 2. Frequent Pattern Mining Codebases
- **lab6/fim_test.py (82 lines):**
  - Single-threaded baseline comparing mlxtend.frequent_patterns.apriori vs pgrowth.
  - Uses TransactionEncoder to build boolean one-hot transaction matrix.
  - Runs scaling experiments over $[0.2, 0.4, 0.6, 0.8, 1.0]$ slices of oodmartFIM.txt.
- **Assignment_7_FIM_All_Files/assignment7_fim.py (244 lines):**
  - dic_algorithm(dataset, min_support, M=100): Implements Dynamic Itemset Counting with transition tracking between confirmed frequent (solid box), suspected frequent (dashed box), confirmed infrequent (solid circle), and candidate (dashed circle). Checkpoints every $ transactions.
  - hash_apriori(dataset, min_support): Implements DHP. Uses hash function (x, y) = ((x \times 10) + y) \pmod B$ to populate bucket counts for 2-itemsets during pass 1, filtering candidate generation in pass 2.
  - ertical_transaction(dataset, min_support): Implements ECLAT. Maps items to set(TIDs) and calculates higher-order support exclusively through set.intersection().
- **lab8/fim_algorithms.py (412 lines):**
  - partitioning_apriori(dataset, min_support, num_partitions=4): Implements the SON algorithm. Slices data into $ chunks, identifies local frequent itemsets with threshold $\lceil s \cdot |P_i| \rceil$, unions candidates, and makes exactly 1 final global pass for true counts.
  - close_algorithm(dataset, min_support): Computes Frequent Closed Itemsets (FCIs) via the closure operator (X) = \bigcap_{t \in \mathcal{T}(X)} t$ and prunes non-minimal generators.
  - pincer_search(dataset, min_support): Implements bidirectional search. Operates bottom-up (, L_k$) and top-down using the Maximum Frequent Candidate Set (MFCS). Prunes bottom-up candidate branches as soon as a top-down maximal candidate is validated.

### 3. Machine Learning & Probabilistic Codebases
- **lab 9/online_shoppers_decision_tree.py (574 lines) & src/train_decision_tree.py (357 lines):**
  - Full end-to-end pipeline on Online Shoppers Purchasing Intention ( = 12,330$).
  - ColumnTransformer with StandardScaler for continuous features and OneHotEncoder(drop='first') for categoricals.
  - Stratified 5-Fold Cross-Validation (StratifiedKFold).
  - Benchmarks Gini Impurity, Entropy, and Log Loss split criteria.
  - Pruning: Extracts $\alpha$-path via cost_complexity_pruning_path(), performs grid search over candidate $\alpha$ values to optimize cross-validated ROC-AUC.
  - Counteracts class imbalance (84.5% / 15.5%) using class_weight='balanced'.
- **lab10/build_notebook.py (1018 lines) & lab10/ notebooks:**
  - Implements GaussianNB (with ar_smoothing tuning), MultinomialNB, BernoulliNB, and ComplementNB.
  - Evaluates probability calibration using CalibratedClassifierCV.
  - Implements multiclass logistics status classification on supply_chain_data.csv.
  - Implements synthetic clustering using make_blobs and make_moons, comparing K-Means inertia and HAC dendrogram cutoffs using scipy.cluster.hierarchy.

### 4. Advanced Clustering & Metric Space Codebases
- **lab11/online_shoppers_clustering.ipynb (27 cells):**
  - Evaluates density distribution of e-commerce sessions without ground truth labels.
  - Implements automated knee/elbow detection: Fits NearestNeighbors(n_neighbors=19), sorts $-distances, detects curvature elbow at $\varepsilon = 1.6$.
  - Benchmarks DBSCAN across 32 hyperparameter combinations.
  - Implements PCA (2 components) and t-SNE (=30$) coordinate projections for visual cluster verification.
  - Fits OPTICS (min_samples=15, xi=0.05) to generate reachability plots exposing hierarchical sub-clusters.
  - Fits Gaussian Mixture Models (GMM) with EM algorithm, minimizing BIC to select optimal covariance structure.

### 5. Metaheuristics & Optimization Codebases
- **lab12/task1_genetic_algorithm.py (305 lines):**
  - Solves continuous 5D non-linear Sphere function optimization $\min \sum_{i=1}^5 x_i^2$.
  - Representation: 80-bit chromosome (16 bits per decision variable).
  - Binary decoding:  = x_{min} + \frac{\text{intval}}{2^{16}-1}(x_{max} - x_{min})$.
  - Fitness formulation:  = \frac{1}{1 + f(\mathbf{x})}$.
  - Operators: Tournament Selection (=3$) vs Roulette Wheel, Two-point crossover ( = 0.85$), Bit-flip mutation ( = 1/80$), Elitism (preserves top 2).
- **lab12/task2_tsp_ga.py (321 lines):**
  - Solves combinatorial Traveling Salesman Problem (TSP) across 20 cities.
  - Representation: Permutation chromosome (ordering of city indices).
  - Operators:
    - *Order Crossover 1 (OX1):* Preserves relative ordering without creating duplicate or missing cities.
    - *Inversion Mutation (2-opt):* Inverts a random sub-tour to untangle intersecting path edges.
  - Result: Achieved 54.43% path length reduction over 200 generations.
- **lab12/task4_classification_clustering.py (245 lines):**
  - Trains 6 classifiers across 4 datasets (Iris, Breast Cancer, Wine, Synthetic).
  - Collects 7 performance metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Train Time, Test Time.
  - Standardizes the  \times 7$ metric matrix and executes K-Means and HAC to group algorithms into performance tiers.

### 6. Production PySpark Enterprise Platform Codebase (BIG_DATA_PROJECT)
- **urban-service-intelligence/src/processing/clean.py (280 lines):**
  - Industrial PySpark ETL pipeline.
  - Enforces explicit StructType schema on raw CSV ingestion.
  - Deduplicates by unique_key using PySpark Windowing: 
ow_number().over(Window.partitionBy('unique_key').orderBy('created_date')) == 1.
  - Calculates incident resolution duration in hours: (F.unix_timestamp('closed_date') - F.unix_timestamp('created_date')) / 3600.0.
  - Applies strict quality flags (lag_invalid_dates, lag_invalid_coords).
  - Writes idempotent Snappy-compressed Parquet partitioned by year and month.
- **urban-service-intelligence/src/mining/fpgrowth.py (204 lines):**
  - Distributed transaction basket mining using pyspark.ml.fpm.FPGrowth.
  - Maps multi-agency service requests into transaction arrays, extracts association rules (confidence, lift), and saves rule metadata to JSON.
- **urban-service-intelligence/src/modeling/resolution.py (418 lines):**
  - Machine learning pipeline predicting 7-day incident closure probability.
  - Enforces strict censoring policy: Excludes immature cases created within 7 days of snapshot to prevent false negative bias.
  - Eliminates target leakage: Strictly drops closed_date, 
esolution_hours, and status.
  - Uses ColumnTransformer with StandardScaler and OneHotEncoder, training RandomForestClassifier and LogisticRegression with chronological time-based holdout validation.


## 8. PLUG-AND-PLAY PRODUCTION CODE VAULT (COPY-PASTE READY)

### MODULE 1: UNIVERSAL AUTO-EDA & DATA PROFILER
```python
import numpy as np
import pandas as pd

def audit_dataset(df, target_col=None):
    print("="*80)
    print(f"SHAPE: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"DUPLICATES: {df.duplicated().sum()}")
    print("="*80)
    
    audit = pd.DataFrame({
        'dtype': df.dtypes,
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_count': df.nunique(),
        'skewness': [df[c].skew() if pd.api.types.is_numeric_dtype(df[c]) else np.nan for c in df.columns]
    })
    print(audit)
    
    if target_col and target_col in df.columns:
        print("\nTARGET DISTRIBUTION:")
        print(df[target_col].value_counts(normalize=True).round(4) * 100)
    return audit
```

### MODULE 2: ROBUST PREPROCESSING & SKEWNESS RECTIFIER
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_preprocessing_pipeline(df, drop_cols=None, target_col=None):
    df_clean = df.copy()
    if drop_cols:
        df_clean = df_clean.drop(columns=drop_cols)
    if target_col and target_col in df_clean.columns:
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
    else:
        X = df_clean
        y = None
        
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Skewness treatment on positive continuous features
    for col in num_cols:
        if X[col].min() >= 0 and X[col].skew() > 1.5:
            X[col] = np.log1p(X[col])
            
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor
```

### MODULE 3: FAST FP-GROWTH & ASSOCIATION RULE MINER
```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

def mine_patterns(transactions, min_support=0.02, min_confidence=0.5):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = fpgrowth(df_trans, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    
    rules = pd.DataFrame()
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
        if not rules.empty:
            rules = rules.sort_values(by='lift', ascending=False)
            
    print(f"[+] Discovered {len(frequent_itemsets)} frequent itemsets and {len(rules)} association rules.")
    return frequent_itemsets, rules
```

### MODULE 4: DENSITY CLUSTERING WITH AUTOMATED KNEE DETECTION (DBSCAN)
```python
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

def auto_dbscan(X_scaled, min_samples=10):
    # Step 1: k-NN distance graph
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    
    # Step 2: Automated knee heuristic (90th percentile)
    eps_val = np.percentile(k_distances, 90)
    
    db = DBSCAN(eps=eps_val, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    
    print(f"[*] DBSCAN: eps={eps_val:.4f}, min_samples={min_samples}")
    print(f"    --> Found {n_clusters} clusters. Noise: {noise_pct:.2f}%")
    
    if n_clusters > 1:
        valid_mask = labels != -1
        sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
        dbi = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
        print(f"    --> Silhouette: {sil:.4f}, Davies-Bouldin: {dbi:.4f}")
    return labels, eps_val
```

### MODULE 5: MULTI-CLASSIFIER AUTOML BENCHMARK SUITE
```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def benchmark_classifiers(X, y):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(max_depth=5, class_weight='balanced'),
        'GaussianNB': GaussianNB(),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    }
    
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results.append({
            'Model': name,
            'Accuracy': scores['test_accuracy'].mean(),
            'Precision': scores['test_precision_macro'].mean(),
            'Recall': scores['test_recall_macro'].mean(),
            'F1': scores['test_f1_macro'].mean(),
            'ROC_AUC': scores['test_roc_auc_ovr'].mean()
        })
        
    df_res = pd.DataFrame(results).sort_values(by='F1', ascending=False)
    print(df_res.to_string(index=False))
    return df_res
```

### MODULE 6: PYSPARK DISTRIBUTED FP-GROWTH & ML PIPELINE
```python
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

def run_pyspark_pipeline(spark, df, feature_cols, label_col):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    indexer = StringIndexer(inputCol=label_col, outputCol="label")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
    
    pipeline = Pipeline(stages=[assembler, scaler, indexer, rf])
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)
    return model, predictions
```

---

## 9. HACKATHON COMBAT CHECKLIST
- [ ] Active directory: `lab_hackathon/`
- [ ] Verify dataset shape and columns: `df.info()`, `df.isnull().sum()`
- [ ] Check target distribution for severe imbalance: `df[target].value_counts(normalize=True)`
- [ ] Apply $\log(1+x)$ to right-skewed positive continuous columns ($g_1 > 1.5$)
- [ ] Apply `StandardScaler` before any metric-space model (K-Means, DBSCAN, KNN, PCA, SVM)
- [ ] For FIM, verify dataset sparsity: If sparse like FoodMart, set $\text{min\_support} \le 0.01$; if dense, set $\ge 0.1$
- [ ] On imbalanced data, NEVER rely on accuracy alone — always report Precision, Recall, Macro-F1, and PR-AUC/ROC-AUC!
- [ ] Visual figures must have labeled axes, titles, legends, and clear executive takeaways.
