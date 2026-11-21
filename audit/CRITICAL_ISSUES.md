# CRITICAL ISSUES — Executive summary of the independent audit

**Evidence base:** The official UCI archive was downloaded and checked independently (the scripts ran in a scratch area outside the project). No project files were modified. [V] = reproduced.

Detailed reasoning is in the companion files:
- DATASET_AND_PREPROCESSING_AUDIT
- TRANSACTION_DESIGN_REVIEW
- ALGORITHM_CORRECTNESS_AUDIT
- PYSPARK_BENCHMARK_AUDIT
- MISSING_REQUIREMENTS

## 1. Verified errors (must fix before implementation)

| ID | Severity | Issue | Evidence | Correction | Impact |
|---|---|---|---|---|---|
| C1 | **CRITICAL** | Sentinel-coded missing values are undetected. US rates/spreads are **0** on 15 US bond holidays; Gold is **−1.0** on 40 UK holidays; Oil −1.0 once; CTB* −1.0 on Canadian holidays | [V] dated lists in DATASET §3 | Convert to NaN before any statistics, then impute by type | Changes all descriptive statistics, plots 7 and 10, DBSCAN noise, NB variances, and ARM items |
| C2 | **CRITICAL** | Pincer Search spec has no **recovery procedure** for candidates lost when MFS-subsumed sets are removed from L_k | Lin & Kedem 1998 | Implement recovery, or keep L_k intact and skip *counting* only for subsumed candidates | Without it, MFIs can be missed and the equivalence test fails |
| C3 | **CRITICAL** | Spark `NaiveBayes` default (multinomial) rejects negative features, and returns are negative | Spark API contract | `modelType='gaussian'` (or bernoulli on 0/1) | The PySpark requirement fails as specified |
| C4 | **CRITICAL** | Synthetic scaling by replication leaves the FP-tree effectively unchanged | Supports are invariant under replication | Bootstrap + noise + re-discretise | The benchmark conclusion would be invalid |
| C5 | HIGH | Forward-filling **returns** (most columns) duplicates moves; SSEC has a 57-day NaN run | [V] | ffill levels only; returns → 0 / item absent | NB features, statistics, ACF plots |
| C6 | HIGH | `Yield_Curve_Inverted` has 0% support (TE2 never < 0 in 2010–17) | [V] | Replace with steep/flat terciles or ΔTE2 | Dead item; `Normal` item = constant |
| C7 | HIGH | `Market_Up` is same-day; rules like Tech⇒Market_Up are index arithmetic (conf 0.84 same-day vs 0.515 next-day) | [V] | Add `NextDay_*` consequent; label the rest contemporaneous | Misleading "predictive" claims |
| C8 | HIGH | TE5, TE6, DE4, DE5, DE6 formulas wrong; CTB* mislabelled as yields; Volume mislabelled as z-score | [V] exact reconstruction | Correct the dictionary | Report correctness |
| C9 | HIGH | NB "beats random chance" baseline. The test majority rate is 55.46% | [V] | Majority baseline + binomial/McNemar tests; no assertion of success | Prevents a false skill claim |
| C10 | HIGH | "PCA 80% ≈ 5 components" is false: 19 components are needed | [V] | Choose from the scree plot; report it | DBSCAN design |
| C11 | HIGH | DBSCAN tests hard-code outcomes (≥ 2 clusters, noise < 20%, eps ∈ [0.5, 3.5]) | Test plan | Remove; validate with DBCV/stability | Scientific validity |
| C12 | HIGH | Memory via `tracemalloc` misses the JVM and NumPy buffers | — | psutil RSS of the process tree | Benchmark validity |
| C13 | MEDIUM | Credit "Widening" item measures level, not change; train median splits mainly by date | — | High/Low + ΔDE1 item | Interpretation |
| C14 | MEDIUM | Spark rules are single-consequent only, and minSupport uses ⌈·⌉ counts | API | Compare exactly on \|Y\| = 1 using integer counts | Equivalence test |
| C15 | MEDIUM | Wrong Pincer citation (Lin & Dunham) in the knowledge base | — | Lin & Kedem, EDBT 1998 | Academic credibility |
| C16 | HIGH | REQ-09 missing from the traceability matrix | RTM numbering | Recover it from the exam paper | Compliance |

## 2. Unsupported assumptions (state as hypotheses or delete)
- Crossover at N ≈ 50k–100k.
- JVM 2–5 s vs 20 ms.
- "~5 PCA components".
- "Defect 3 mitigation" presented as novel. It is the original MFCS-gen subsumption step.
- Pincer "< 10 s" runtime bound and "prevents combinatorial explosion". The worst case remains exponential.
- "Economically intuitive" rules written before mining.
- "Volatility regimes" from DBSCAN.
- "ffill replicates trading reality".
- "1,984 not 1,985" UCI remark.
- Unverified dependency pins (matplotlib 3.11.2, pyarrow 25.0.1, NumPy 2 + pyspark 3.5).

## 3. Verified correct (keep)
- Shapes, dates, gaps, NaN counts, the 80/86 column structure, peer naming.
- The target definition and its 54.6% up-rate.
- The 200-row warm-up drop (the minimum is 199).
- The bfill prohibition.
- The FP-Growth support, confidence and lift formulas.
- The `prune_and_branch_mfcs` logic. It is correct for an antichain MFCS processed one infrequent set at a time; proof in ALGORITHM §B.3.
- The choice of FP-Growth + NB for PySpark.
- The HAC dendrogram companion.
- `USD_*`, `Gold_*` and `Oil_*` as **sign of return** are mathematically meaningful: the columns are returns, not levels [V]. They only need sentinel/zero handling.

## 4. Optional improvements
- Walk-forward NB evaluation.
- Out-of-sample rule-confidence check.
- BH-corrected rule significance.
- `fpmax` as an independent MFI oracle.
- `local[P]` strong-scaling sweep with a fitted Amdahl f.
- Calendar strip plot of DBSCAN labels.
- Five-market NB table reported as correlated replications (64 of 84 columns are identical across files [V]).
