# MINIMUM VIABLE SUBMISSION (MVP) & BEST-VERSION SPECIFICATION
*(Fully Reconciled & Audited Version)*

**Project:** Analytics and Systems of Big Data Practice — Endsem Hackathon  
**Target:** Clear Boundary Between Core Exam Compliance and Best-in-Class Excellence  

---

## 1. MVP vs Best-Version Feature Matrix

| Functional Area | Minimum Viable Submission (MVP) | Best-Version Implementation (Excellence) |
|---|---|---|
| **Dataset Scope** | Focus on primary benchmark (`Processed_S&P.csv`, 1,984 rows, 84 features). | Cross-market generalization evaluated across **all 5 market indices** (S&P 500, DJI, NASDAQ, NYSE, Russell 2000), reported as correlated replications. |
| **Data Cleaning & Representations** | Step 0 sentinel elimination; truncate warm-up ($t < 200$, leaving $N=1,784$); separate into **Representation A** (descriptive, NaNs preserved) and **Representation B** (modeling, levels `ffill`, returns `0.0`). | Step 0 sentinel elimination; warm-up drop; 3 representations; rolling IQR outlier detection; calendar holiday audit. |
| **Data Partitioning** | Strict chronological 80/20 split with 1-row boundary embargo: **1,425 Train** + **1 Embargo** + **357 Test** = **1,783 labelled rows**. | Partitioning with 1-row embargo + 5-fold expanding window walk-forward validation on Train. |
| **Data Transformation** | Stationary moving average spreads ($Close/EMA_k - 1$); complete **24-item leak-free vocabulary** (22 contemporaneous, 2 forward target tokens). | Stationary spreads; 24-item vocabulary; rolling volatility and term spread slope features; train-fitted `StandardScaler`. |
| **Visualizations** | Exactly 12 curriculum-taught plots (300 DPI) on Representation A with post-rendering inferences. | 12 curriculum plots + additional diagnostic plots (rule scatter, PCA regime projection, calendar strip of labels, HAC dendrogram, reliability curve). |
| **FP-Growth ARM** | Integer ceiling support counts ($\sigma_{\min} = \lceil minsup \times N \rceil$); Conviction metric; predictive (`NextDay_Up/NotUp`, $N=1,783$) vs contemporaneous ($N=1,784$) rule separation. | Operating point ($\text{minsup}=0.10, \text{minconf}=0.60$); sensitivity grid heatmap; Fisher exact test + Benjamini-Hochberg FDR correction on predictive rules; out-of-sample test evaluation. |
| **Naive Bayes vs DBSCAN** | Gaussian Naive Bayes on stationary features evaluated out-of-sample vs **55.46% test majority baseline**; DBSCAN on 5 regime features with knee detection; HAC dendrogram. | Gaussian & Bernoulli NB with Binomial/McNemar tests and 5-fold walk-forward validation; class likelihood parameter tables ($\mu, \sigma^2$); DBSCAN with DBCV score and $\varepsilon \pm 10\%$ bootstrap stability. |
| **Pincer Search MFIs** | Lin & Kedem (1998) bi-directional search with sequential MFCS antichain branching and counting-pruning. | Dual oracle cross-validation asserting $\text{MFIs}_{\text{Pincer}} \equiv \text{MFIs}_{\text{fpmax}} \equiv \text{maximal}(\text{FIs}_{\text{FPGrowth}})$; 200 randomized property tests vs brute force; candidate instrumentation. |
| **PySpark MLlib** | PySpark MLlib `FPGrowth` and `NaiveBayes(modelType='gaussian')` on local SparkSession, exporting via `toPandas()`. | Full MLlib pipelines; exact rule equality on single consequents ($|consequent| == 1$); prediction agreement $\ge 99.5\%$ with Scikit-learn; dynamic `JAVA_HOME` detection. |
| **Benchmarking** | Phase-separated latency profiling (session, conversion, warmup, steady fit, collect); peak RSS process-tree memory via `psutil`. | Phase-separated latencies; `psutil` process-tree RSS; strong scaling with fitted Amdahl $f$; weak scaling across bootstrap-perturbed workloads ($N=1,784$ to $500,000$); empirical crossover analysis. |
| **Final PDF Report** | Comprehensive 20–30 page academic report compiled via ReportLab covering all 18 structured chapters and embedded figures. | Publication-grade ReportLab report with institutional header, numbered figures, styled tables, LaTeX math equations, formal academic citations, and negative results discussed. |
| **Source Packaging** | Modular repository with `src/`, `notebooks/`, `tests/` (25 pytest assertions), single-command `run_all.py`, pinned `requirements.txt`, and `README.md`. | Complete reproducible package with `requirements.txt`, SHA-256 data verification, verified one-click master execution, and clean git history. |

---

## 2. Minimum Viable Submission (MVP) Specification

The MVP satisfies 100% of the explicit examination requirements without taking shortcuts or pre-asserting outcomes:

### Mandatory Deliverables:
1. **Source Code (`src/`):**
   - `src/data_loader.py`: Ingests and harmonizes UCI dataset; builds column-type registry.
   - `src/preprocessing.py`: Implements Step 0 sentinel elimination, warm-up drop, 3 data representations, and 80/20 embargo split ($1,425 + 1 + 357 = 1,783$).
   - `src/feature_engineering.py`: Formulates target (`NextDay_Up` vs `NextDay_NotUp`), stationary spreads, and 24-item transaction baskets (22 contemporaneous for $N=1,784$; 24 tokens for $N=1,783$).
   - `src/descriptive_analytics.py`: Computes statistical moments, normality ($JB$), and autocorrelation ($Q$) on Representation A using `skipna=True`.
   - `src/visualization.py`: Renders 12 curriculum-aligned plots at 300 DPI on Representation A.
   - `src/association_rules.py`: Implements integer FP-Growth, Conviction, and predictive rule separation.
   - `src/classification.py`: Implements Gaussian Naive Bayes evaluated against 55.46% test majority baseline on Representation B.
   - `src/clustering.py`: Implements DBSCAN on 5 regime features with automated knee detection and HAC dendrogram.
   - `src/pincer_search.py`: Implements Lin & Kedem bi-directional Pincer Search.
   - `src/pyspark_pipeline.py`: Implements Spark MLlib FPGrowth and Gaussian Naive Bayes, saving via `toPandas()`.
   - `src/benchmark_engine.py`: Profiles phase-separated latency, `psutil` process-tree RSS, and bootstrap scaling.
   - `src/generate_pdf_report.py`: Automated ReportLab PDF generator compiling all 18 structured chapters.
   - `run_all.py`: Master runner executing the complete pipeline end-to-end.
2. **Notebooks (`notebooks/`):**
   - 8 structured notebooks corresponding to each analysis phase.
3. **Artifacts (`outputs/`):**
   - Exactly 12 curriculum figures in `outputs/figures/`.
   - Formatted CSV tables in `outputs/tables/`.
4. **PDF Report (`reports/`):**
   - Academic PDF report in `reports/Analytics_BigData_Endsem_Report.pdf` covering all required techniques, key inferences, plots, outputs, parameters, and interpretations.

---

## 3. Best-Version Enhancement Specification

Following MVP verification, these enhancements elevate the submission:
1. **Multi-Index Cross-Market Generalization:** Evaluate directional predictability across all 5 market indices (S&P 500, DJI, NASDAQ, NYSE, Russell 2000), reported as correlated replications.
2. **Dual Oracle Cross-Validation for Pincer Search:** Mathematically prove that $\text{MFIs}_{\text{Pincer}} \equiv \text{MFIs}_{\text{fpmax}} \equiv \text{maximal}(\text{FIs}_{\text{FPGrowth}})$ across real data and 200 random property test databases.
3. **Synthetic Scalability Experiment with Bootstrap Perturbation:** Benchmark across $N=1,784$ to $500,000$ transactions generated via continuous feature perturbation, expanding tree topology and demonstrating Amdahl's Law and the empirical crossover point.
4. **Statistical Rigor on Directional Forecasting:** Include Binomial and McNemar tests against the test majority baseline (55.46%) and persistence baseline, honestly reporting market efficiency limits.

---

## 4. Precise First Task for the Coding Agent

Upon explicit user approval to proceed to Phase 1 implementation, the coding agent's immediate first task will be:

```markdown
### Task 1: Environment Setup & Raw Data Acquisition (Phase 0)
1. Verify required Python packages are installed:
   pip install "numpy>=1.26.4,<2.0.0" scipy==1.15.2 pandas==2.2.3 scikit-learn==1.6.1 matplotlib==3.10.0 seaborn==0.13.2 mlxtend==0.23.4 kneed==0.8.5 pyspark==3.5.4 pyarrow==19.0.1 reportlab==4.3.1 psutil==6.1.1 pytest==8.3.4
2. Dynamically verify JAVA_HOME points to a valid Java 11 or 17 JDK.
3. Create project directory tree:
   mkdir -p data/raw data/interim data/processed notebooks src outputs/figures outputs/tables outputs/models outputs/metrics tests reports
4. Author `src/data_loader.py` to download `https://archive.ics.uci.edu/static/public/554/cnnpred%2Bcnn%2Bbased%2Bstock%2Bmarket%2Bprediction%2Busing%2Ba%2Bdiverse%2Bset%2Bof%2Bvariables.zip`, extract all 5 CSV files into `data/raw/`, verify SHA-256 checksums, assert exact dimensions (1,984 rows, 84 columns), and author the canonical column-type registry.
5. Run `tests/test_data_loader.py` to assert data acquisition integrity.
```
