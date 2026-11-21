# ALGORITHM CORRECTNESS AUDIT — FP-Growth, Pincer Search, Naive Bayes, DBSCAN

## A. FP-Growth and rule metrics

| Item | Gemini | Verdict |
|---|---|---|
| Transaction representation | `list[set[str]]` → one-hot | Correct. See TRANSACTION_DESIGN_REVIEW T-4 |
| Support σ(X)/\|D\| | stated | Correct |
| Confidence σ(X∪Y)/σ(X) | stated | Correct |
| Lift Conf/Supp(Y) | stated | Correct. Note that lift is symmetric in X and Y |
| Conviction | named but **no formula given** | **Add:** conv(X⇒Y) = (1 − supp(Y)) / (1 − conf(X⇒Y)), which is ∞ when conf = 1. Handle ∞ explicitly in tables and sorting |
| Validation "Support(X⇒Y) ≤ Support(X)" | stated | Correct, but trivial |
| Validation "Lift > 1 for positive rules" | stated | **Tautological** (lift > 1 is the definition of positive). Replace with "lift computed = conf/supp(Y) within 1e-12" |
| TEST-FP-02 "Lift ≥ 0" | stated | Correct. Also add the upper bound lift ≤ 1/supp(Y) |

### FP-1 · MEDIUM · Sensitivity grid lacks a multiple-testing view
- **Issue:** Sweeping minsup ∈ [0.05, 0.20] × minconf ∈ [0.5, 0.8] and then reporting the "best" rules is data dredging.
- **Correction:**
  - Report the number of itemsets and rules per grid cell as a heat map.
  - Choose one operating point *a priori* (minsup = 0.10, minconf = 0.60).
  - For predictive rules, add a test of independence: a χ² or Fisher exact test on the 2×2 contingency table, with Benjamini–Hochberg correction across rules.
  - Validate out of sample: compute each rule's confidence on the test period.

### FP-2 · LOW · Rule pruning
- "Prune by conviction" is under-specified.
- **Recommended pipeline:**
  1. minconf.
  2. Lift > 1 + δ (δ = 0.05).
  3. Remove redundant rules: drop X′⇒Y when there is an X ⊂ X′ with conf(X⇒Y) ≥ conf(X′⇒Y).
  4. Remove mirror duplicates (T-1).
  5. Rank by conviction, then by lift.
- mlxtend produces multi-item consequents, while Spark produces single-item consequents only (see PYSPARK audit). Restrict the cross-tool comparison to |Y| = 1.

---

## B. Pincer Search — critical audit

**Reference:** D.-I. Lin & Z. M. Kedem, "Pincer-Search: A New Algorithm for Discovering the Maximum Frequent Set," EDBT 1998 (extended version in IEEE TKDE 14(3), 2002).
- `knowledge-context/HACKATHON_MASTER_BRAIN.md` cites "Lin & Dunham, 1998". **That citation is wrong.** Lin & Dunham (1998) is a different paper (partition-based ARM). Use the Lin & Kedem citation everywhere.

### B.1 What the genuine algorithm requires

```
L0 = ∅; k = 1; C1 = {{i} | i ∈ I}
MFCS = { I }                       # one element: the whole item universe
MFS  = ∅
while Ck ≠ ∅:
    count supports of Ck ∪ MFCS                      (one DB pass)
    MFS  ← MFS ∪ { m ∈ MFCS | m frequent }
    MFCS ← MFCS \ { m frequent }
    Lk   ← { c ∈ Ck | c frequent and c ⊄ any m ∈ MFS }   # subsumed sets are not kept in Lk…
    Sk   ← { c ∈ Ck | c infrequent }
    if Sk ≠ ∅: MFCS ← MFCS-gen(MFCS, Sk)
    Ck+1 ← join(Lk) ; RECOVERY(Lk, MFS) ; prune(Ck+1)        # …which is why RECOVERY is needed
    Ck+1 ← { c ∈ Ck+1 | c ⊆ some m ∈ MFCS }                  # new MFCS-prune
    k ← k+1
return MFS                                            # every element already maximal
```

**MFCS-gen(MFCS, Sk):**

```
for s in Sk:
  for m in MFCS with s ⊆ m:
    MFCS.remove(m)
    for e in s:
      m' = m \ {e}
      if not ∃ m'' ∈ MFCS: m' ⊆ m'': MFCS.add(m')
```

**Invariants that give correctness:**
- (I1) Every frequent itemset is a subset of some element of MFS ∪ MFCS.
- (I2) MFCS is an antichain.
- (I3) No element of MFCS contains a known infrequent set.

### B.2 Verdict on Gemini's design

| Component | Gemini | Verdict | Severity |
|---|---|---|---|
| MFCS initialization | Not specified | Must be `{I}` (all items), or `{set of frequent 1-items}` after pass 1. Equivalent if pass 1 immediately runs MFCS-gen on infrequent singletons | MEDIUM (spec gap) |
| MFCS update (`prune_and_branch_mfcs`) | Code given | **Correct as written** (see B.3), but inefficient and mis-described | LOW |
| Bottom-up candidate generation | "Apriori join" implied | Under-specified. Needs the join + Apriori subset prune **plus** the MFCS-subset prune | MEDIUM |
| Top-down pruning of C_k | "prune any c ∈ C_k that is a subset of an MFI in MFS" | **Incomplete → can lose MFIs or produce wrong output.** If subsumed frequent sets are dropped from L_k *before the join*, some (k+1)-candidates that are not under MFS can no longer be generated. Lin & Kedem add a **recovery procedure** for exactly this case. The plan does not mention recovery. | **CRITICAL** |
| MFS maintenance | "verify no MFS element is subset of another" | Needed as an assertion only. Under the true algorithm, MFS elements are maximal by construction: any frequent superset would be under MFCS ∪ MFS, and MFCS elements are never subsets of frequent sets. With the shortcut variant below, a final maximality filter is mandatory | MEDIUM |
| Termination | Not stated | Holds: k ≤ \|I\|, and C_{k+1} ⊆ k+1-subsets of I. MFCS-gen strictly reduces cardinality per split, so it terminates | — |
| End-of-loop | Not stated | Some MFIs are discovered bottom-up only (frequent sets in L_k with no frequent superset). The genuine algorithm captures them because each of them eventually *is* an MFCS element. **Must test** that the final MFS ⊇ maximal(L_k) at the last k | HIGH (spec gap) |
| "Defect 3 mitigation" label | Presented as a novel hardening | The subsumption check *is* the original MFCS-gen. Presenting it as an extra fix is inaccurate | LOW (claims) |
| Worst-case complexity | "prevents combinatorial explosion"; TEST-PS-05 "< 10 s" | **Unsupported.** \|MFCS\| can grow exponentially in \|S_k\|: it is the minimal-transversal / hypergraph dualisation problem. Subsumption pruning does **not** change the worst case. Remove the "< 10 s" acceptance threshold or make it an observation | HIGH |

### B.3 Line-by-line check of `prune_and_branch_mfcs`

```python
for m in mfcs:
    if infrequent_set.issubset(m):
        for item in infrequent_set:
            child = m - {item}
            if not any(child.issubset(other) for other in mfcs if other != m):
                new_mfcs.add(child)
    else:
        new_mfcs.add(m)
return {m for m in new_mfcs if not any(m < other for other in new_mfcs)}
```

**Claim:** for an antichain MFCS, this equals MFCS-gen for one infrequent set s.

**Proof sketch:**
- Suppose child = m∖{e} ⊆ other, where `other` is in the old MFCS.
  - If `other` ⊇ s, then e ∈ other, so m = child ∪ {e} ⊆ other. That contradicts the antichain property. So `other` ⊉ s and `other` survives into new_mfcs, which means dropping `child` keeps I1.
  - If `other` ⊉ s, it also survives.
- The final cleanup removes children subsumed by sibling children. The result is the maximal elements of {m ∉ ↑s} ∪ {m∖{e}}, which is MFCS-gen. ∎

**Defects found:**
1. Applying the function **once per infrequent set** is correct. Batching several s together is not: a child must also avoid containing the *other* infrequent sets. The sequential loop guarantees this. Document it, and add a test that no MFCS element contains any element of S_k.
2. It is O(\|MFCS\|²·\|I\|) per infrequent set because it rebuilds the whole set. That is acceptable for about 22 items.
3. Empty children (m = s, \|s\| = 1) must be discarded. ∅ is not a useful MFCS element.

### B.4 Correctness argument that must appear in the report
- **Soundness:** every element placed in MFS was counted frequent. Maximality: a frequent proper superset Y ⊋ X would, by I1, lie under some m ∈ MFS ∪ MFCS. Elements of MFS are mutually incomparable by the final filter, and MFCS elements counted in the same pass would have been promoted.
- **Completeness:** Let X be maximal frequent. By I1, X ⊆ m for some m ∈ MFCS ∪ MFS at every pass. MFCS elements only shrink by removing an item of an infrequent subset. Because X contains no infrequent set, some descendant chain of m always contains X. Each split loses at most one item, so the chain reaches X itself, or a frequent m ⊇ X (which must equal X by maximality). That element is counted and promoted.
- **Termination:** At most \|I\| passes. Each MFCS split strictly decreases \|m\|.

### B.5 Mandatory verification tests (replace TEST-PS-*)
1. **Oracle:** `mlxtend.frequent_patterns.fpmax`. This gives MFIs directly and is independent of the Pincer implementation. Compare against it in addition to "FP-Growth itemsets filtered to maximal". The filter: X ∈ FI and no Y ∈ FI with X ⊂ Y.
2. Use **integer support counts** (count ≥ ⌈minsup·N⌉) in all three implementations. This avoids float boundary mismatches.
3. **Randomised property tests:** 200 random databases (5–12 items, 10–60 transactions, random minsup). Pincer == brute-force maximal sets.
4. **Textbook test:** Lin & Kedem's own example, or a hand-worked one, with the expected MFCS state after each pass, not just the final MFS.
5. **Instrumentation** (report this; do not assert thresholds): per pass, record \|C_k\|, \|S_k\|, \|MFCS\|, \|MFS\|, and the number of candidates pruned by MFCS/MFS. Compare the candidate count against Apriori. Pincer's benefit appears only when long MFIs exist, and on these ~22-item, ~3–5-length patterns it may be **no faster than Apriori**. Report what is observed.

---

## C. Naive Bayes

| Item | Gemini | Verdict | Severity |
|---|---|---|---|
| Target alignment | y_t = 1[C_{t+1} > C_t], features at t | Correct [V] | — |
| Chronological 80/20 split | yes | Correct. Add a 1-row purge at the boundary | LOW |
| Baseline | "accuracy exceeds random chance" (TEST plan / roadmap) | **Wrong baseline.** The test-set majority class is 55.46% [V]. "Better than 50%" is not evidence of skill | **HIGH** |
| TEST-NB-02 "beats dummy" | phrased as a contrast | Must **not** be an assertion. The result may legitimately fail, and the test must not force an outcome | HIGH |
| Metrics | Acc, P, R, F1, ROC-AUC | Add **balanced accuracy, MCC**, the Brier score / log-loss (NB probabilities are poorly calibrated), and a reliability curve | MEDIUM |
| Class imbalance (54/46) | "no SMOTE needed" | Agree. Do not use SMOTE on time series. Report per-class recall | — |
| Feature suitability | Raw 82 features incl. levels | **Reject level features** (DATASET P-3). GaussianNB assumes conditional independence and Gaussian marginals. Returns are fat-tailed (check the JB statistic), and the groups FX, futures and cash are strongly dependent, so duplicated evidence is counted multiple times. Collapse the redundant groups | HIGH |
| Hyper-parameters | none | `var_smoothing` via TimeSeriesSplit inside train only | LOW |
| Significance | none | **Binomial test** of test accuracy vs the majority rate (n = 357). The 95% CI half-width is ≈ ±5.2 pp, so differences under 5 pp are not significant. **McNemar test** NB vs Dummy | MEDIUM |
| Robustness | single split | **Walk-forward evaluation** (expanding window, e.g. 5 folds) gives a distribution rather than one number | MEDIUM |
| BernoulliNB | planned | Features = sign of each return (0 threshold) plus train-median binarisation for levels. Valid and natural here | — |

**Prior expectation (not a prediction to assert):** The CNNpred paper itself reports modest F-measures for daily direction. Daily index direction is close to a random walk. A sound plan writes acceptance as "metrics computed and reported with CI", **not** "NB beats baseline".

---

## D. DBSCAN

| Item | Gemini | Verdict | Severity |
|---|---|---|---|
| Scaling | Robust/Standard | Required because of mixed units (D-4). Remove sentinels first; otherwise the 15 zero-rate days and 40 gold −1.0 days become the "noise cluster" | **HIGH** |
| PCA 80% "~5 components" | stated | **False [V]:** on train-fitted standardised stationary features, 80% needs **19** components and 90% needs 29. The first 5 explain 53.6% | **HIGH** |
| k = 2·dim = 10 | stated | The heuristic is MinPts = 2·dim (Sander et al. 1998), with k = MinPts−1 nearest neighbour in the k-distance plot. The numbers must follow the *actual* dim chosen | MEDIUM |
| eps ∈ [0.5, 2.5]; TEST-DB-01 eps ∈ [0.5, 3.5] | asserted | **Predetermined outcome.** eps depends on dimension and scaling. Remove the range assertion; assert only eps > 0 and finite | HIGH |
| TEST-DB-02 "≥ 2 clusters, noise 0–20%" | asserted | **Predetermined labels. Reject.** A likely honest outcome on daily return vectors is one dense cluster plus noise (high-volatility days). That is itself a legitimate "calm vs stress" finding | **HIGH** |
| "volatility regimes" | asserted | Unsupported. DBSCAN on day-level vectors ignores time. Regimes require temporal coherence. **Better feature set for regime discovery:** rolling 20-day realised vol of mom, rolling mean return, TE2, DE1, ΔDE1, rolling FX vol. Then check whether clusters form contiguous calendar blocks | MEDIUM |
| Validation | knee only | Add: silhouette on non-noise points (with caveats for density clusters), **DBCV** (density-based validation), stability under bootstrap / eps ±10% (adjusted Rand index), the noise-fraction curve vs eps, cluster profile table (mean of original features), and a calendar strip plot of labels | MEDIUM |
| Dimensionality | 80 dims | PCA to the number of components chosen by the scree/parallel analysis. Alternatively, DBSCAN on 2–5 handcrafted regime features, which is interpretable. Say why | MEDIUM |
| Dendrogram | HAC Ward companion | Good. Satisfies "dendrograms or other clustering outputs". Use a truncated dendrogram (p = 30) and the cophenetic correlation | — |
| "whichever is relevant" | both | Fine. Justify **NB as the primary relevant technique** (a labelled target exists) and DBSCAN as a complementary descriptive technique. Do not call DBSCAN "predictive" | LOW |
