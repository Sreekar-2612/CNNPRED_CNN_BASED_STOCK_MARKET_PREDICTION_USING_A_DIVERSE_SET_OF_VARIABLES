# COMPREHENSIVE TEST PLAN & VERIFICATION SPECIFICATION
*(Fully Reconciled & Audited Version)*

**Project:** Analytics and Systems of Big Data Practice — Endsem Hackathon  
**Standard:** Automated Regression Testing & Mathematical Proof Verification (Zero Predetermined Outcomes)  

---

## 1. Test Architecture & Pytest Hierarchy

The testing framework uses `pytest` with parameterized test cases covering unit logic, mathematical axioms, algorithmic cross-validation, and integration workflows:

```
tests/
├── test_data_loader.py         # UCI ingest, SHA-256 integrity, schema harmonization, column registry
├── test_preprocessing.py      # Sentinel elimination, warm-up drop, 3 representations, embargo split (1425+1+357=1783)
├── test_feature_engineering.py# Next-day target formulation, stationary spreads, 24-item vocabulary
├── test_fp_growth.py          # Integer ceiling support, metric definitions, predictive rule out-of-sample
├── test_pincer_search.py      # Lin & Kedem bi-directional search, invariants I1–I3, dual oracle validation
├── test_classification.py     # Gaussian/Bernoulli NB out-of-sample metrics, 55.46% majority baseline, McNemar
├── test_clustering.py         # DBSCAN on 5 regime features, knee locator, DBCV score, HAC dendrogram
├── test_pyspark.py            # Spark Gaussian NB parity (>=99.5%), Spark FPGrowth exact rule parity
├── test_benchmark.py          # Overhead isolation, psutil process-tree RSS, bootstrap scaling
└── test_report.py             # 12 curriculum plots exist (>50 KB), PDF report compiles cleanly (>2 MB)
```

---

## 2. Unit & Integration Test Specifications

### 2.1 Data Ingestion Tests (`tests/test_data_loader.py`)

| Test ID | Test Name | Description / Assertions | Success Threshold |
|---|---|---|---|
| **TEST-DL-01** | `test_raw_csv_existence_and_size` | Verify all 5 raw CSV files exist in `data/raw/` and each file size is $> 1.4$ MB. | Exactly 5 files, all $> 1,400,000$ bytes. |
| **TEST-DL-02** | `test_raw_dimensions` | Verify every loaded raw DataFrame has exactly 1,984 rows and 84 columns. | `assert df.shape == (1984, 84)` for all 5 files. |
| **TEST-DL-03** | `test_temporal_monotonicity` | Verify that the `Date` column in all files is strictly monotonically increasing with no duplicate timestamps. | `assert df['Date'].is_monotonic_increasing and df['Date'].nunique() == 1984`. |
| **TEST-DL-04** | `test_schema_harmonization` | Verify that peer index columns (`GSPC`/`S&P`, `IXIC`, `RUT`) map to canonical tokens without data loss. | Canonical schema resolves target index across all 5 datasets. |

### 2.2 Preprocessing & Sentinel Cleaning Tests (`tests/test_preprocessing.py`)

| Test ID | Test Name | Description / Assertions | Success Threshold |
|---|---|---|---|
| **TEST-PR-01** | `test_sentinel_conversion` | Verify Step 0 converts rate/spread zeros on bond holidays ($DGS10 == 0$) and commodity drops $\le -0.99$ to `np.nan`. | Assert exactly 15 bond holiday rows converted in rate block; assert 40 gold holiday rows converted. |
| **TEST-PR-02** | `test_warmup_truncation_count` | Verify that the first 200 rows containing `EMA_200` lookback nulls are removed, leaving exactly 1,784 clean trading rows. | `assert len(clean_df) == 1784`. |
| **TEST-PR-03** | `test_three_representations_separation` | Verify that Representation A preserves NaNs in returns (`assert df_descriptive.isnull().sum().sum() > 0`), while Representation B has 0 NaNs (`assert df_modeling.isnull().sum().sum() == 0`). | Complete separation of descriptive vs modeling datasets. |
| **TEST-PR-04** | `test_exact_embargo_split_counts` | Verify exact partition counts on labeled rows ($N=1,783$): Train = 1,425, Embargo = 1, Test = 357. | `assert len(train) == 1425 and len(embargo) == 1 and len(test) == 357 and len(train)+len(embargo)+len(test) == 1783`. |
| **TEST-PR-05** | `test_strict_chronological_ordering` | Verify strict temporal order across splits: `train['Date'].max() < embargo['Date'] < test['Date'].min()`. | Zero lookahead boundary overlap. |

### 2.3 Feature Engineering & Target Tests (`tests/test_feature_engineering.py`)

| Test ID | Test Name | Description / Assertions | Success Threshold |
|---|---|---|---|
| **TEST-FE-01** | `test_target_formulation_correctness` | Verify target: $y_t = 1 \iff Close_{t+1} > Close_t$ (`NextDay_Up`); $y_t = 0 \iff Close_{t+1} \le Close_t$ (`NextDay_NotUp`). | 100% exact match on rows 0 to 1,782. |
| **TEST-FE-02** | `test_final_row_target_omission` | Verify the final trading day ($t = 1,983$, post-warmup index 1,783) has target set to NaN and is omitted from predictive sets. | Last row omitted from predictive sets ($N=1,783$ labeled rows). |
| **TEST-FE-03** | `test_stationary_spread_features` | Verify stationary moving average spreads ($Close / EMA_k - 1$) are computed without infs or NaNs. | All spread features stationary and finite. |
| **TEST-FE-04** | `test_complete_24_item_vocabulary` | Verify transaction generator uses the complete 24-item vocabulary: 22 contemporaneous tokens for $N=1,784$; 24 tokens for $N=1,783$. | Every contemporaneous basket has $\ge 5$ items; `TermSpread_Steep/Flat` both have $>0$ support. |

### 2.4 FP-Growth Association Rule Mining Tests (`tests/test_fp_growth.py`)

| Test ID | Test Name | Description / Assertions | Success Threshold |
|---|---|---|---|
| **TEST-FP-01** | `test_integer_support_monotonicity` | Verify that all returned itemsets have count $\ge \lceil minsup \times N \rceil$, and for any $A \subset B$, $\text{Support}(B) \le \text{Support}(A)$. | Support monotonicity strictly satisfied on integer counts. |
| **TEST-FP-02** | `test_rule_confidence_lift_and_conviction` | Verify mathematical bounds: $\text{Confidence} \in [0, 1]$, $\text{Lift} \ge 0$, and $\text{Conviction} = \frac{1 - \text{Supp}(Y)}{1 - \text{Conf}(X \Rightarrow Y)}$ (handling $\infty$). | Relative error $< 10^{-12}$ on all generated rules. |
| **TEST-FP-03** | `test_predictive_vs_contemporaneous_separation` | Verify predictive rules have consequent $\in \{NextDay\_Up, NextDay\_NotUp\}$ and antecedents free of future tokens. | 100% clean rule separation. |

### 2.5 Pincer Search MFI Discovery & Dual Oracle Tests (`tests/test_pincer_search.py`)

| Test ID | Test Name | Description / Assertions | Success Threshold |
|---|---|---|---|
| **TEST-PS-01** | **`test_dual_oracle_equivalence_market_data`** | **CRITICAL CROSS-VALIDATION:** Assert exact mathematical set equality between Pincer Search MFIs, `mlxtend.frequent_patterns.fpmax`, and maximal itemsets filtered from FP-Growth. | **`assert set(MFIs_Pincer) == set(MFIs_fpmax) == set(maximal(FIs_FPGrowth))`** |
| **TEST-PS-02** | `test_mfi_maximality_axiom` | Verify that NO returned MFI is a proper subset of another frequent itemset: $\forall X \in \text{MFS}, \not\exists Y \supset X \text{ s.t. } \text{count}(Y) \ge \sigma_{\min}$. | 100% compliance with maximality axiom. |
| **TEST-PS-03** | `test_mfcs_invariants_and_antichain` | Verify invariants I1, I2 (antichain), and I3 (no infrequent subsets) hold after every pass $k$. | Invariants I1–I3 verified across all passes. |
| **TEST-PS-04** | `test_randomized_property_tests` | Run 200 randomized small transaction databases (5–12 items, 10–60 transactions) comparing Pincer Search against brute force. | 100% agreement across all 200 random databases. |

### 2.6 Naive Bayes & DBSCAN Tests (`tests/test_classification.py`, `tests/test_clustering.py`)

| Test ID | Test Name | Description / Assertions | Success Threshold |
|---|---|---|---|
| **TEST-NB-01** | `test_naive_bayes_out_of_sample_metrics` | Fit GaussianNB on train split ($N=1,425$); evaluate on test split ($N=357$); assert Accuracy, Balanced Acc, Precision, Recall, F1, MCC, ROC-AUC, Brier score are computed. | All metrics computed and bounded in $[0, 1]$. |
| **TEST-NB-02** | `test_naive_bayes_statistical_tests` | Run Binomial test of test accuracy against 55.46% majority baseline; run McNemar test against Persistence baseline. | Statistical tests execute and return valid p-values (no outcome asserted). |
| **TEST-DB-01** | `test_dbscan_knee_detection_and_dbcv` | Verify automated k-distance knee locator determines a finite $\varepsilon > 0$; compute DBCV score and non-noise silhouette. | $\varepsilon > 0$, DBCV score computed, no hardcoded cluster counts. |
| **TEST-DB-02** | `test_hac_dendrogram_cophenetic` | Verify Hierarchical Agglomerative Clustering companion computes cophenetic correlation coefficient $> 0.60$. | Cophenetic correlation $> 0.60$. |

### 2.7 PySpark Distributed Pipeline Tests (`tests/test_pyspark.py`)

| Test ID | Test Name | Description / Assertions | Success Threshold |
|---|---|---|---|
| **TEST-SPK-01** | `test_spark_gaussian_naive_bayes_parity` | Verify Spark MLlib `NaiveBayes(modelType='gaussian')` runs without error on negative returns and matches Scikit-learn predictions $\ge 99.5\%$. | Prediction agreement $\ge 99.5\%$. |
| **TEST-SPK-02** | `test_spark_fpgrowth_exact_rule_parity` | Verify Spark MLlib `FPGrowth` integer-count frequent itemsets and single-consequent rules match `mlxtend` exactly. | Exact itemset equality; confidence and lift match within $10^{-9}$. |

### 2.8 Benchmarking & Report Tests (`tests/test_benchmark.py`, `tests/test_report.py`)

| Test ID | Test Name | Description / Assertions | Success Threshold |
|---|---|---|---|
| **TEST-BM-01** | `test_benchmark_overhead_isolation` | Verify benchmark harness reports session initialization, conversion, warmup, steady fit, and collect times separately. | All 5 latency phases reported. |
| **TEST-BM-02** | `test_psutil_process_tree_rss_memory` | Verify memory measurement samples Resident Set Size (RSS) across Python driver and Java child processes via `psutil`. | Memory $> 0$, includes JVM footprint. |
| **TEST-BM-03** | `test_bootstrap_scaling_topology_expansion` | Verify synthetic scaling generates data via bootstrap perturbation, producing expanding distinct transaction counts. | Number of unique transactions expands with $N$. |
| **TEST-REP-01** | `test_curriculum_figure_count_and_size` | Verify that exactly 12 curriculum visualization files exist in `outputs/figures/` and each file size is $> 50$ KB. | Exactly 12 distinct plots, all valid PNGs $> 50$ KB. |
| **TEST-REP-02** | `test_pdf_report_compilation` | Verify `src/generate_pdf_report.py` compiles `reports/Analytics_BigData_Endsem_Report.pdf` without exceptions. | Exit code 0, file exists, size $> 2$ MB, all 18 chapters present. |

---

## 3. Master Acceptance Verification Checklist (25 Criteria)

Before declaring implementation complete, all 25 criteria must pass:

- [ ] Raw archive SHA-256 verified; all 5 CSVs loaded with shape $(1984, 84)$.
- [ ] Step 0 sentinel elimination sets US rate holidays ($DGS10 == 0$) and commodity drops ($\le -0.99$) to NaN.
- [ ] Rows 0–199 truncated ($N=1,784$ remaining).
- [ ] Descriptive dataset preserves missing returns as NaN; modeling dataset imputes levels with `ffill` and returns with 0.0.
- [ ] Strict chronological 80/20 train/test split with 1-row boundary embargo ($1,425 + 1 + 357 = 1,783$).
- [ ] Scalers, quantiles, and selectors fitted strictly on training observations.
- [ ] Next-day target constructed as `NextDay_Up` vs `NextDay_NotUp`; last row omitted from supervised sets.
- [ ] All 12 curriculum plots generated, saved at 300 DPI, each with post-rendering inferences.
- [ ] Rolling annualized volatility calculated on returns (`mom`), not price levels.
- [ ] Underwater drawdown calculated on cumulative return compounder.
- [ ] Complete 24-item vocabulary used: 22 contemporaneous tokens for $N=1,784$, 2 forward target tokens for $N=1,783$.
- [ ] FP-Growth rules separated into predictive ($NextDay\_*$) and contemporaneous co-movement.
- [ ] Conviction calculated with explicit handling of $\infty$.
- [ ] Pincer Search implements Lin & Kedem bi-directional search with MFCS antichain branching.
- [ ] **Dual cross-validation passes:** Pincer Search MFIs $\equiv$ `mlxtend.frequent_patterns.fpmax` $\equiv$ `maximal(FP-Growth)`.
- [ ] PySpark `NaiveBayes` configured with `modelType='gaussian'`; passes with $\ge 99.5\%$ Scikit-learn prediction agreement.
- [ ] PySpark `FPGrowth` integer support itemsets and single-consequent rules match `mlxtend` exactly.
- [ ] Synthetic scaling generates data via bootstrap perturbation, not pure replication.
- [ ] Memory profiled via `psutil` process-tree RSS.
- [ ] Benchmarking isolates SparkSession startup overhead from steady-state fit.
- [ ] DBSCAN evaluated via DBCV, non-noise silhouette, and knee locator (no hardcoded cluster counts).
- [ ] HAC Ward linkage dendrogram generated with Cophenetic Correlation reported.
- [ ] Code fully commented with docstrings and mathematical derivations (REQ-09).
- [ ] Automated test suite (`pytest tests/`) passes 100% with exit code 0.
- [ ] Final academic PDF report compiles cleanly ($> 2$ MB) with all 18 chapters and embedded figures.
