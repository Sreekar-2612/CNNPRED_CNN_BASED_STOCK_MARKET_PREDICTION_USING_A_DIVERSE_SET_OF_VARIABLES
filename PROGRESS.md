# PROGRESS TRACKING — AUTONOMOUS EXECUTION
## Endsem Hackathon — Analytics and Systems of Big Data Practice

**Status:** Completed & Methodologically Reconciled (100% Execution Verified)  
**Timestamp:** 2026-11-21  
**Institution:** Indian Institute of Information Technology, Design and Manufacturing (IIITDM) Kancheepuram  

---

### Phase-by-Phase Review & Execution Status

| Phase | Description | Status | Completion % | Review Verification & Key Invariants |
|---|---|---|---|---|
| **Phase 0** | Workspace setup, directory tree, raw dataset copy & SHA-256 validation | Completed | 100% | Verified 5 raw CSVs copied to `data/raw/`. Checksums match 100% (`6c050...`, `fe5b0...`, `569a4...`, `c3c0e...`, `62f39...`). |
| **Phase 1** | Data loader module & column dictionary | Completed | 100% | `src/data_loader.py` implemented. All 5 indices verified: shape $(1984, 84)$. `outputs/tables/table_01_dataset_hashes.csv`. |
| **Phase 2** | Preprocessing, warm-up truncation, Rep A, Rep B, Rep C, Embargo partition | Completed | 100% | Sentinels eliminated: 15 US rate closure dates (0.0 -> NaN), 40 holiday negative sentinels (-1.0 -> NaN). Warmup truncated (200 rows). Rep A ($N=1,784$, NaNs preserved), Rep B ($N=1,784$, zero NaNs). Embargo partition: $1,425$ Train + $1$ Embargo + $357$ Test = $1,783$ labeled rows. |
| **Phase 3** | Descriptive analytics (moments, skew, kurt, JB, LB) on Rep A | Completed | 100% | `src/descriptive_analytics.py` executed with `skipna=True`. 60 feature series profiled. Severe non-normality verified (JB $p < 0.0001$), excess kurtosis observed. `table_07_descriptive_stats.csv`. |
| **Phase 4** | Feature engineering & 24-token transaction vocabulary | Completed | 100% | Stationary spreads derived ($(Close - EMA50)/EMA50$, $\Delta TE$, $\Delta DE$). Stationarity confirmed via ADF ($p < 0.05$). C1 Contemporaneous (22 tokens, $N=1,784$), C2 Predictive (24 tokens, $N=1,783$). Train majority class = **1 (Up)** (774/1,425 = 54.32%); applied to test = **55.46%** (198/357). |
| **Phase 5** | 12 Publication-Quality Visualizations at 300 DPI + t-SNE | Completed | 100% | `fig_01` through `fig_12` plus dedicated `fig_09b_tsne_manifold.png` generated and saved in `outputs/figures/`. Professional typography, seaborn whitegrid styling, DPI=300. |
| **Phase 6** | Integer FP-Growth, rule generation, conviction, Fisher exact test | Completed | 100% | Strict database separation: Contemporaneous ($N=1,784$, denom 1,784, $\sigma_{\min} = \lceil 0.08 \times 1,784 \rceil = 143$, 1,251 itemsets, 7,532 rules) vs Predictive Discovery ($N=1,425$, denom 1,425, $\sigma_{\min} = \lceil 0.08 \times 1,425 \rceil = 114$, 1,723 itemsets, 572 rules). Out-of-sample test evaluation performed on 357 test rows. |
| **Phase 7** | Pincer Search algorithm (Lin & Kedem 1998) & dual-oracle validation | Completed | 100% | Bidirectional Pincer Search implemented with intact $L_k$ join, counting-pruning, and sequential MFCS antichain branching. Discovered **279 Maximal Frequent Itemsets**. Verified 100% exact set identity against `mlxtend.fpmax` and maximal FP-Growth filter. |
| **Phase 8** | Naive Bayes classification (Gaussian & Bernoulli) & McNemar test | Completed | 100% | Gaussian NB (50.98% acc), Bernoulli NB (54.06% acc) evaluated against the training-derived **55.46%** test majority baseline (Train Maj: Class 1, 54.32%). Zero test leakage. McNemar significance tests and feature likelihood tables generated (`table_14`, `table_15`). |
| **Phase 9** | Clustering (DBSCAN + knee locator + DBCV) & Dimensionality Reduction (PCA, Ward HAC, t-SNE) | Completed | 100% | Automated knee locator on 9-NN graph derived optimal $\epsilon = 1.5018$. 2 distinct regimes profiled (normal vs stress/crisis). Truncated Ward HAC dendrogram ($p=30$), PCA scree plot, and t-SNE 2D non-linear manifold projection (`fig_09b_tsne_manifold.png`) generated. Scaler fit strictly on training set. |
| **Phase 10** | Distributed PySpark MLlib Pipeline (Gaussian NB, FPGrowth) | Completed | 100% | PySpark `NaiveBayes(modelType="gaussian")` achieved **100.00% exact prediction parity** (357/357) with Scikit-learn on the test set. Distributed FP-Growth mined 1,897 itemsets and 6,309 rules. |
| **Phase 11** | Big Data Benchmarking Engine (isolated timing, psutil RSS, Amdahl) | Completed | 100% | 5 isolated timing phases measured. Background `psutil` process-tree peak physical memory monitored. Strong scaling analyzed across $p \in [1, 2, 4]$ cores, fitting Amdahl parallel fraction $f = 0.6718$. Synthetic scaling tested up to $N = 30,000$. |
| **Phase 12** | Jupyter Notebooks (01 through 11) execution with stored outputs | Completed | 100% | All 11 Jupyter notebooks created, executed in prescribed order (01 to 11) via `jupyter nbconvert`, and saved in `notebooks/`. All cells contain evaluated outputs, figures, and rich markdown documentation. |
| **Phase 13** | Pytest test suite (10 modules, 26 unit tests) & `run_all.py` | Completed | 100% | 26 unit tests executed via `pytest tests/`. 100% pass rate achieved in 2.49s. Single-command master script `run_all.py` validated. |
| **Phase 14** | Academic PDF Report Compilation (ReportLab) & Documentation | Completed | 100% | 18-chapter academic monograph compiled: `reports/Analytics_BigData_Endsem_Report.pdf` (**4.93 MB**, exceeding the $> 2$ MB threshold). Master `README.md` and traceability matrix updated. |

---

### Targeted Methodological Audit Resolution (Final Turn)

1. **Majority Baseline Rigor:**
   - Evaluated strictly using training set labels $y_{\text{train}}$ ($N=1,425$).
   - Training Class 1 count: 774 ($54.32\%$), Class 0 count: 651 ($45.68\%$).
   - Fixed training prediction rule: $\hat{y}_{\text{baseline}} = 1$.
   - Out-of-sample application to test partition $y_{\text{test}}$ ($N=357$): Test Class 1 count: 198, Class 0 count: 159.
   - Resulting test accuracy: $198 / 357 = 55.46\%$.
   - Confirmed: 55.46% is not an assumed parameter or leaked test metric; it is the exact out-of-sample accuracy of the naive rule trained on $y_{\text{train}}$. Zero test label leakage. McNemar contingency tables recalculated.

2. **FP-Growth Database Separation & Denominator Precision:**
   - **Contemporaneous Mining (Representation C1):** Database size $N=1,784$, denominator 1,784, minimum support threshold $\sigma_{\min} = \lceil 0.08 \times 1,784 \rceil = 143$. Discovered 1,251 frequent itemsets and 7,532 association rules.
   - **Predictive Rule Discovery (Representation C2):** Database size $N=1,425$ (training transactions only), denominator 1,425, minimum support threshold $\sigma_{\min} = \lceil 0.08 \times 1,425 \rceil = 114$. Discovered 1,723 frequent itemsets and 572 predictive rules.
   - **Out-of-Sample Test Evaluation:** Mined predictive rules were applied out-of-sample to the 357 test period transactions ($N=357$). Confirmed that rules do not retain in-sample arbitrage under market efficiency (test confidence drops to ~50–55%, Fisher p-values lose significance).

3. **t-SNE Manifold Learning Verification (REQ-21):**
   - Implemented `sklearn.manifold.TSNE(n_components=2, perplexity=30.0, random_state=42)` in `src/clustering.py`.
   - Feature matrix standardized strictly with `StandardScaler` fit on the training partition ($N=1,425$) to prevent data snooping.
   - Visualized in `outputs/figures/fig_09_dbscan_regime_clusters_pca.png` (side-by-side Panel A: Linear PCA, Panel B: Non-linear t-SNE) and dedicated high-resolution figure `outputs/figures/fig_09b_tsne_manifold.png`.
   - Integrated into Notebook 08 and Chapters 13 and 14 of the Academic PDF Report.

4. **Master Validation Suite:**
   - `pytest tests/ -v`: 26 / 26 passed in 2.49s.
   - `python run_all.py`: Clean execution (all pipeline modules, notebook builds, and unit tests).
   - `jupyter nbconvert`: All 11 notebooks executed cleanly in sequential dependency order.

---

### Verification Summary
- **Acceptance Criteria:** 25 / 25 Formally Satisfied.
- **Pytest Suite:** 26 / 26 Unit Tests Passing.
- **PDF Report:** 4.93 MB, 18 Chapters, 12 Figures, 18 Tables.
- **Reproducibility:** 100% via `python run_all.py`.
