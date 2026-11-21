# PROJECT SPECIFICATION SUMMARY
## Analytics and Systems of Big Data Practice — Endsem Hackathon
**Institution:** Indian Institute of Information Technology, Design and Manufacturing (IIITDM) Kancheepuram  
**Dataset:** CNNpred: CNN-based stock market prediction using a diverse set of variables (UCI Machine Learning Repository, Dataset ID: 554)  
**Primary Research Reference:** Hoseinzade, E., & Haratizadeh, S. (2019). *Expert Systems with Applications*, 123, 363–376.  
**Algorithmic Foundation:** Lin, D.-I., & Kedem, Z. M. (1998). Pincer-Search: A New Algorithm for Discovering the Maximum Frequent Set. *EDBT 1998*, 105–119.

---

### 1. Executive Summary & Core Invariants
This project delivers a complete, reproducible, and mathematically rigorous implementation of big data analytics, machine learning, association rule mining, maximal frequent itemset discovery, clustering, distributed PySpark processing, benchmarking, and academic report generation on the five major index datasets of the CNNpred benchmark (DJI, NASDAQ, NYSE, RUSSELL, S&P 500).

Every analytical step satisfies the strict constraints established in the planning and audit reconciliation:
- **Raw Data Integrity:** Original CSV files in `cnnpred+cnn+based+stock+market+prediction+using+a+diverse+set+of+variables/` are preserved as read-only references; verified copies reside in `data/raw/`.
- **Chronological Embargo Partition:** Exactly $1,425$ training rows, $1$ boundary embargo row, and $357$ test rows ($1,425 + 1 + 357 = 1,783$ labeled target observations). Test majority baseline is exactly **55.46%** (198 Up / 159 NotUp).
- **Leakage Prevention:** All scalers, encoders, and quantile thresholds are fitted exclusively on the $1,425$ training observations.
- **Three Data Representations:**
  1. **Representation A (Descriptive Analytics, $N=1,784$):** Non-NaN sentinels converted to `np.nan`. Missing returns preserved as NaN. Statistical moments and distributional metrics computed with `skipna=True`.
  2. **Representation B (Continuous Modeling, $N=1,784$):** Zero NaNs. Level features forward-filled (`ffill`); stationary returns and relative change features imputed with `0.0`.
  3. **Representation C (Transaction Analytics):** Unavailable observations mapped to item absence.
     - **Representation C1 (Contemporaneous, $N=1,784$):** 22 item tokens reflecting same-day market dynamics.
     - **Representation C2 (Predictive, $N=1,783$):** 24 item tokens (22 contemporaneous + `NextDay_Up` / `NextDay_NotUp`). Final row `2017-11-15` (unobserved target) is strictly omitted.

---

### 2. Transaction Vocabulary (Exact 24-Item Specification)

| Index | Item Token | Mathematical Condition | Category |
|---|---|---|---|
| 1 | `SameDay_Up` | $Close_t > Close_{t-1}$ | Target/Direction |
| 2 | `SameDay_Down` | $Close_t \le Close_{t-1}$ | Target/Direction |
| 3 | `Volume_Jump` | $\Delta V_t / V_{t-1} > Q_{train}(0.75)$ | Volume Activity |
| 4 | `Mom5_Pos` | $mom5_t > 0$ | Short-Term Momentum |
| 5 | `Mom5_Neg` | $mom5_t \le 0$ | Short-Term Momentum |
| 6 | `Above_EMA50` | $Close_t > EMA50_t$ | Trend Filter |
| 7 | `Below_EMA50` | $Close_t \le EMA50_t$ | Trend Filter |
| 8 | `TermSpread_Steep` | $DGS10_t - DTB3_t > Q_{train}(0.66)$ | Yield Curve Regime |
| 9 | `TermSpread_Flat` | $DGS10_t - DTB3_t \le Q_{train}(0.33)$ | Yield Curve Regime |
| 10 | `TermSpread_Widening` | $\Delta(DGS10 - DTB3)_t > 0$ | Yield Curve Dynamic |
| 11 | `CreditSpread_High` | $DBAA_t - DGS10_t > Q_{train}(0.66)$ | Credit Risk Regime |
| 12 | `CreditSpread_Low` | $DBAA_t - DGS10_t \le Q_{train}(0.33)$ | Credit Risk Regime |
| 13 | `CreditSpread_Widening`| $\Delta(DBAA - DGS10)_t > 0$ | Credit Risk Dynamic |
| 14 | `USD_Up` | $r_{USD, t} > 0$ | FX Dynamics |
| 15 | `USD_Down` | $r_{USD, t} \le 0$ | FX Dynamics |
| 16 | `Gold_Up` | $r_{Gold, t} > 0$ | Commodity Dynamics |
| 17 | `Gold_Down` | $r_{Gold, t} \le 0$ | Commodity Dynamics |
| 18 | `Oil_Up` | $r_{Oil, t} > 0$ | Commodity Dynamics |
| 19 | `Oil_Down` | $r_{Oil, t} \le 0$ | Commodity Dynamics |
| 20 | `Tech_Leaders_Up` | Majority of $\{AAPL, AMZN, GOOGL\} > 0$ | Intermarket Equity |
| 21 | `Europe_Up` | Majority of $\{FTSE, GDAXI\} > 0$ | Intermarket Equity |
| 22 | `Asia_Up` | Majority of $\{N225, HSI\} > 0$ | Intermarket Equity |
| 23 | `NextDay_Up` | $Close_{t+1} > Close_t$ | Forward Target ($t < N-1$) |
| 24 | `NextDay_NotUp` | $Close_{t+1} \le Close_t$ | Forward Target ($t < N-1$) |

---

### 3. Core Algorithms and Methodological Rigor

1. **Association Rule Mining:**
   - Evaluated on integer ceiling counts ($\sigma_{\min} = \lceil minsup \times N \rceil$).
   - Strict separation between Contemporaneous Rules ($A \Rightarrow B$ where $A, B \subseteq \{1..22\}$) and Predictive Rules ($A \Rightarrow \text{NextDay\_Direction}$).
   - Rule ranking by Conviction, Lift, and Support; statistical validation via two-sided Fisher's Exact Test with Benjamini-Hochberg FDR correction ($\alpha = 0.05$).

2. **Bidirectional Pincer Search (Lin & Kedem 1998):**
   - Exact implementation: intact $L_k$ join candidate generation, MFCS-subsumption pruning during candidate counting, sequential MFCS antichain branching on infrequent sets $S \in C_k \setminus L_k$, immediate ancestor removal in MFCS, and early termination when $C_k = \emptyset$ or $MFCS \subseteq \bigcup L_i$.
   - Dual-oracle cross-validation against both `mlxtend.frequent_patterns.fpmax` and the maximal itemset subset of FP-Growth.

3. **Classification:**
   - Gaussian Naive Bayes on stationary standardized continuous features ($r_{idx}$, momentum, spreads).
   - Bernoulli Naive Bayes on the 22-item binary contemporaneous transaction vectors.
   - Honest evaluation against the $55.46\%$ test majority baseline; significance tested via McNemar's Test.

4. **Clustering & Dimensionality Reduction:**
   - DBSCAN applied to 5 domain-informed regime features ($\sigma_{20}$, $TE1$, $DE1$, $r_{idx}$, $Volume$).
   - Distance parameter $\epsilon$ selected automatically via $k$-distance graph elbow/knee detection ($k = 2 \times d - 1 = 9$).
   - Clustering quality evaluated via Density-Based Clustering Validation (DBCV) index and accompanied by truncated Ward HAC dendrogram ($p=30$).
   - PCA (orthogonal linear) vs t-SNE (non-linear manifold) dimensionality reduction with explained variance scree plot.

5. **Distributed PySpark MLlib Pipeline:**
   - PySpark `NaiveBayes(modelType="gaussian")` configured to handle negative financial returns without crash.
   - Prediction parity verification ensuring $\ge 99.5\%$ agreement with Scikit-learn.
   - Spark `FPGrowth` compared against local FP-Growth on single-consequent rules.
   - Explicit export of all Spark results to disk via `.toPandas()`.

6. **Benchmarking & Scalability:**
   - Five isolated timing phases: JVM Session Startup ($T_{\text{session}}$), Data Ingestion & Conversion, Warmup Fit, Steady-State Fit, and Result Collection.
   - Peak physical memory profiling via background `psutil` thread monitoring RSS across the entire process tree.
   - Strong scaling efficiency ($E(p) = S(p) / p$) and Amdahl's Law parallel fraction derivation ($f$).
   - Synthetic dataset scaling from $N=1,784$ to $500,000$ using Gaussian bootstrap perturbation ($\sigma = 0.25 \sigma_{\text{train}}$) followed by regime re-discretization.

---

### 4. Deliverables Checklist
- [x] Standard project directory layout (`data/`, `src/`, `notebooks/`, `outputs/`, `tests/`, `reports/`).
- [ ] Production Python modules in `src/` (12 modules).
- [ ] 11 fully executed Jupyter Notebooks with rich markdown and stored outputs.
- [ ] 12 publication-quality figures at 300 DPI in `outputs/figures/`.
- [ ] Statistical summary tables in `outputs/tables/` and metrics in `outputs/metrics/`.
- [ ] Comprehensive test suite in `tests/` verifying all 25 acceptance criteria.
- [ ] Master execution script `run_all.py` reproducing all results end-to-end.
- [ ] Comprehensive 18-chapter academic PDF report generated via ReportLab in `reports/`.
