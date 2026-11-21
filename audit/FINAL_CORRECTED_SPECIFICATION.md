# FINAL CORRECTED SPECIFICATION

This document supersedes the conflicting parts of the `planning/` documents. It is a specification only; no code is included.

## 0. Global rules
- Seed 42 everywhere.
- Every fitted quantity (scalers, quantiles, medians, MI ranks, PCA, var_smoothing) is fitted on **train rows only** and then frozen.
- No test may assert a modelling *outcome* (accuracy level, cluster count, eps range, crossover point). Tests assert only correctness properties and leakage properties.
- All numbers in the report are produced by code. None are written in advance.

## 1. Data loading
1. Download the UCI 554 zip. Record the SHA-256 of each CSV. Assert shape (1984, 84), monotonic dates and 0 duplicates.
2. Harmonise the peer columns: GSPC / S&P → SP500, IXIC → NASDAQ, RUT → RUSSELL.
3. Add a column-type registry with each column's **kind** (level_price, level_rate, return_frac, return_pct, rel_change) and its **units**. The registry drives every downstream decision.

## 2. Cleaning (in this order)
1. **Sentinels → NaN:**
   - In the rate/spread block: rows where DGS10 == 0 (all US rate and spread columns).
   - In return/relative-change columns: value ≤ −0.99.
   - Log the count per column and the list of dates.
2. Drop rows 0–199 (warm-up). Document the EMA seeding caveat.
3. **Impute by kind:**
   - Levels: forward-fill.
   - Returns and relative changes: 0 for NB/DBSCAN matrices; NaN means "item absent" for transactions.
   - Never use bfill.
4. Assert there are no NaNs in the modelling matrices. Assert that no sentinel values remain.

## 3. Target and split
- y_t = 1[Close_{t+1} > Close_t]. Drop the last row (supervised only).
- Chronological 80/20 on the 1,783 labelled rows, with a 1-row purge at the boundary.
- Expected and to be checked: train ≈ 2010-10-18 … 2016-06-15, test ≈ 2016-06-17 … 2017-11-14, test majority ≈ 55.5%.
- Primary market: S&P. Secondary: the other four, reported as correlated replications.

## 4. Features for NB/DBSCAN
- Exclude raw Close and EMA levels. Add `Close/EMA_k − 1` for k ∈ {10, 20, 50, 200}.
- Keep rate levels only for DBSCAN regime features. For NB, use ΔTE2 and ΔDE1 (in bp) plus the levels, provided the train/test overlap check passes.
- Collapse the redundant groups (Gold/XAU/gold-F, Oil/Brent/WIT, DTBs, Dollar index/-F, futures vs cash).
- Remove pairs with |r| > 0.90 (train).
- Select k by MI using TimeSeriesSplit inside train.

## 5. Descriptive analytics and ≥ 10 plots (all after cleaning)
- **Table:** count, mean, std, skew, kurtosis, JB p-value, Ljung–Box p-value, missing %, units — per column, per market.
- **Plots** (course-taught types first):
  1. line — normalised 5 indices;
  2. histogram + KDE + normal fit of `mom`;
  3. heatmap — Spearman, stationary features;
  4. box — `mom` by year;
  5. missingness matrix (NaN + sentinels highlighted);
  6. line — rolling 21-day annualised volatility;
  7. line — TE1/TE2/DE1 (sentinel-cleaned);
  8. scatter — ΔDE1 vs mom;
  9. bar — ACF of r and r² with Ljung–Box;
  10. scatter/hexbin — USD vs Gold returns (cleaned);
  11. violin — mom by weekday + Kruskal–Wallis;
  12. area — drawdown.
- Additional plots: rule scatter (supp × conf, colour = lift), k-distance, PCA scatter of DBSCAN labels, calendar strip of labels, dendrogram, confusion matrix, ROC, reliability curve, benchmark curves.
- Write each inference only after its figure has been rendered.

## 6. Transactions
Use the vocabulary in TRANSACTION_DESIGN_REVIEW §4: sign thresholds, plus train-fitted Q75, median and terciles. There are two rule families: predictive (consequent `NextDay_*`) and contemporaneous.

## 7. FP-Growth
- Run mlxtend `fpgrowth` with operating point minsup = 0.10 and minconf = 0.60. Use the integer count threshold ⌈minsup·N⌉, and choose minsup so that minsup·N is not an integer.
- Metrics: support, confidence, lift, conviction = (1−supp(Y))/(1−conf), with ∞ handled explicitly.
- Sensitivity grid: minsup {0.05, 0.075, 0.10, 0.15, 0.20} × minconf {0.5, 0.6, 0.7, 0.8}. Report counts as a heat map; select nothing from the grid.
- Pruning: lift > 1.05 → remove redundant rules (a sub-antecedent rule with conf ≥) → remove mirror rules → rank by conviction.
- Predictive rules: Fisher exact test + BH (q = 0.05). Report each rule's test-period confidence.

## 8. Pincer Search (from scratch)
- The algorithm exactly as in ALGORITHM_CORRECTNESS_AUDIT §B.1:
  - MFCS = {I};
  - count C_k ∪ MFCS per pass;
  - promote frequent MFCS elements to MFS;
  - MFCS-gen per infrequent set, sequentially, with subsumption;
  - Apriori join + subset prune + **recovery** + MFCS-subset prune;
  - termination when C_k = ∅;
  - final maximality filter as an assertion.
- Integer supports.
- **Tests:**
  - equality with mlxtend `fpmax` **and** with maximal(FP-Growth FI);
  - 200 random small databases vs brute force;
  - a hand-worked textbook example with per-pass MFCS states;
  - invariants I1–I3 checked after every pass.
- **Report:** per-pass |C_k|, |S_k|, |MFCS|, |MFS|, and candidates counted vs Apriori. State the exponential worst case honestly.

## 9. Naive Bayes
- GaussianNB on the continuous selected features. BernoulliNB on sign/median-binarised features.
- Baselines: majority class, and "persistence" (y_hat_t = 1[mom_t > 0]).
- Metrics: accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC, Brier score, confusion matrix, reliability curve.
- Binomial test vs the majority rate; McNemar NB vs baselines.
- Walk-forward 5-fold expanding window as a robustness check.
- "Classification rules" output: per-class μ/σ table and the top log-likelihood-ratio features.

## 10. DBSCAN (descriptive regime analysis) + HAC
- Regime feature set: 21-day realised vol, 21-day mean return, TE2, DE1, ΔDE1, 21-day USD vol. Optionally, the full stationary set with PCA (component count chosen by scree/parallel analysis and reported).
- StandardScaler, choosing either train-fit or full-sample fit and stating which.
- MinPts = 2·dim. k-distance at k = MinPts−1, with kneed to find the knee. Sweep eps ±30% around the knee and report the number of clusters and the noise fraction for each eps.
- Validation: DBCV, silhouette on non-noise points, bootstrap ARI stability, and a cluster profile table. Check temporal contiguity with the calendar strip.
- Ward HAC dendrogram (truncated) with the cophenetic correlation.
- No assertions about the cluster count.

## 11. PySpark (two native techniques)
- `FPGrowth(itemsCol='items', minSupport=s, minConfidence=c)`. Assert exact equality of the frequent itemsets with mlxtend, and of the single-consequent rules (conf and lift to 1e-9). Compute conviction in a post-processing step.
- `NaiveBayes(modelType='gaussian')` on VectorAssembler features. Assert prediction agreement ≥ 99.5% with sklearn (var_smoothing ≈ 0).
- Optionally, `modelType='bernoulli'` vs BernoulliNB.
- Write outputs through `toPandas()` so Windows does not need winutils.

## 12. Benchmarking
- Phase-separated timings (session, conversion, warm-up fit, steady fit, materialisation). Median and IQR over ≥ 5 runs.
- Strong scaling: `local[1]`, `local[2]`, `local[4]`, `local[P]` at a fixed N. Fit Amdahl's f.
- Weak/data scaling: N ∈ {1,983; 10k; 50k; 100k; 250k; 500k} using **bootstrap + noise + re-discretise** for FP-Growth and bootstrap + noise for NB. Report the number of distinct transactions and the number of frequent itemsets at each N.
- Memory: peak RSS of the Python process and of the java child processes (psutil sampler, 50 ms). Also report the configured driver memory.
- Report the crossover as measured, including "none observed".
- Record the hardware and software versions.

## 13. Environment
- Clean venv with a pinned Python version (3.10 or 3.11). Resolve the dependencies, then freeze them to a `requirements.lock`.
- Use NumPy < 2 if pyspark 3.5 needs it. Install mlxtend.
- Detect JAVA_HOME at runtime and assert Java 11 or 17.

## 14. Report
- One chapter per exam requirement, mapped 1:1 to the recovered REQ list (including REQ-09).
- Each chapter: method, parameters (support, confidence, eps, MinPts, k, splits), outputs, inferences, limitations, and negative results.
- Correct citations: Hoseinzade & Haratizadeh 2019; Lin & Kedem 1998; Han et al. 2000 (FP-Growth); Li et al. 2008 (PFP); Ester et al. 1996 (DBSCAN); Sander et al. 1998; Moulavi et al. 2014 (DBCV).
