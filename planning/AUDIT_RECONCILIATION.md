# AUDIT RECONCILIATION & CONSISTENCY AUDIT REPORT
*(Fully Reconciled & Audited Version)*

**Project:** Analytics and Systems of Big Data Practice — Endsem Hackathon  
**Institution:** IIITDM Kancheepuram  
**Dataset:** CNNpred: CNN-based stock market prediction using a diverse set of variables (UCI Dataset ID: 554)  
**Date of Reconciliation:** 2026-11-21  

---

## 1. Executive Summary of Consistency Reconciliation

Following Claude's independent audit and the subsequent final consistency audit, all residual ambiguities, indexing discrepancies, vocabulary counts, and data representation policies have been resolved.

Every single finding has been verified empirically and mathematically against the official raw UCI dataset bitstream (`c:\Users\Sreekar_Tirumala\AppData\Local\Temp\...`). All eight planning deliverables in `planning/` and the master artifact now strictly align with 100% mathematical and architectural consistency.

---

## 2. Specific Consistency Resolutions

### 2.1 Train/Test Indexing Reconciliation
- **Total Raw Observations:** Exactly 1,984 rows (original 0-based indices 0 to 1,983; dates `2009-12-31` to `2017-11-15`).
- **Warm-Up Cutoff:** The first 200 rows (indices 0 to 199; `2009-12-31` to `2010-10-15`) are truncated to remove the 199 initial nulls of `EMA_200`.
- **Post-Warmup Observations:** Exactly **1,784 rows** (original rows 200 to 1,983; post-truncation indices 0 to 1,783; `2010-10-18` to `2017-11-15`).
- **Unlabelled Final Row:** Original row 1,983 (post-truncation index 1,783; `2017-11-15`) has unobserved $Close_{t+1}$ and is excluded from supervised training/testing sets.
- **Total Labeled Observations:** Exactly **1,783 rows** (original rows 200 to 1,982; post-truncation indices 0 to 1,782).
- **Exact Partition Breakdown:**
  - **Training Split ($N_{\text{train}}$):** Exactly **1,425** rows (original rows 200 to 1,624; post-truncation indices 0 to 1,424; dates `2010-10-18` to `2016-06-15`).
  - **Boundary Embargo ($N_{\text{embargo}}$):** Exactly **1** row (original row 1,625; post-truncation index 1,425; date `2016-06-16`). Purged because its target uses $Close_{1626}$, so its features are purged to prevent test boundary overlap.
  - **Test Split ($N_{\text{test}}$):** Exactly **357** rows (original rows 1,626 to 1,982; post-truncation indices 1,426 to 1,782; dates `2016-06-17` to `2017-11-14`).
  - **Mathematical Invariant:**
    $$N_{\text{train}} + N_{\text{embargo}} + N_{\text{test}} = 1,425 + 1 + 357 = \mathbf{1,783} \quad (\checkmark \text{ Exact})$$
  - Test set target distribution: 198 Up / 159 NotUp $\implies$ Test majority baseline = **55.46%**.

---

### 2.2 Transaction Vocabulary Count & Manual Enumeration
The specification previously declared 22 items while listing 24 tokens. The declared count has been corrected to **EXACTLY 24 DISTINCT ITEM TOKENS**:

- **Contemporaneous Item Tokens (22 tokens, evaluated on day $t$):**
  1. `SameDay_Up` ($mom_t > 0$)
  2. `SameDay_Down` ($mom_t < 0$)
  3. `Volume_Jump` ($Volume_t > Q75_{\text{train}}(Volume)$)
  4. `Mom5_Pos` ($ROC\_5_t > 0$)
  5. `Mom5_Neg` ($ROC\_5_t < 0$)
  6. `Above_EMA50` ($Close_t > EMA\_50_t$)
  7. `Below_EMA50` ($Close_t < EMA\_50_t$)
  8. `TermSpread_Steep` ($TE2_t > Q67_{\text{train}}(TE2)$)
  9. `TermSpread_Flat` ($TE2_t < Q33_{\text{train}}(TE2)$)
  10. `TermSpread_Widening` ($TE2_t - TE2_{t-1} > 0$)
  11. `CreditSpread_High` ($DE1_t > \text{Median}_{\text{train}}(DE1)$)
  12. `CreditSpread_Low` ($DE1_t \le \text{Median}_{\text{train}}(DE1)$)
  13. `CreditSpread_Widening` ($DE1_t - DE1_{t-1} > 0$)
  14. `USD_Up` ($DollarIndex_t > 0$)
  15. `USD_Down` ($DollarIndex_t < 0$)
  16. `Gold_Up` ($Gold_t > 0$)
  17. `Gold_Down` ($Gold_t < 0$)
  18. `Oil_Up` ($Oil_t > 0$)
  19. `Oil_Down` ($Oil_t < 0$)
  20. `Tech_Leaders_Up` ($AAPL_t > 0 \land MSFT_t > 0$)
  21. `Europe_Up` ($FTSE_t > 0 \land DAX\text{-}F_t > 0$)
  22. `Asia_Up` ($HSI_t > 0 \land Nikkei\text{-}F_t > 0$)
- **Forward Target Tokens (2 tokens, evaluating forward outcome $y_t$):**
  23. `NextDay_Up` ($Close_{t+1} > Close_t$)
  24. `NextDay_NotUp` ($Close_{t+1} \le Close_t$)
- **Total Universe:** $22 + 2 = \mathbf{24 \text{ distinct tokens}}$.

---

### 2.3 Separation of the Three Data Representations
1. **Representation A: Descriptive Analytics Dataset (`df_descriptive`):**
   - $N = 1,784$ observations. Sentinels converted to `np.nan`.
   - **Missing returns preserved as NaN.**
   - All moments, normality tests ($JB$), and autocorrelation tests ($Q$) are calculated pairwise via `skipna=True`. This prevents zero-imputation from artificially compressing variances or creating artificial spikes at zero.
2. **Representation B: Continuous Modeling Dataset (`df_modeling`):**
   - $N = 1,784$ observations. Zero NaNs.
   - Levels forward-filled (`ffill`); returns imputed with `0.0` (zero price movement on closed markets).
   - Used for dense vector operations in Naive Bayes, DBSCAN, PCA, and PySpark VectorAssembler.
3. **Representation C: Discrete Transaction Datasets:**
   - Unavailable/missing observations represent **item absence**.
   - C1 (Contemporaneous): $N = 1,784$ baskets ($t = 200 \dots 1,983$), using the 22 contemporaneous tokens.
   - C2 (Predictive): $N = 1,783$ baskets ($t = 200 \dots 1,982$), using the 22 contemporaneous tokens plus exactly one forward target token (`NextDay_Up` or `NextDay_NotUp`).

---

### 2.4 Predictive Transaction Alignment & Omission of Final Row
- Original row 1,983 (`2017-11-15`) has unobserved $Close_{t+1}$.
- It is strictly omitted from `transactions_predictive.json` ($N=1,783$).
- It is included in `transactions_contemp.json` ($N=1,784$) because all contemporaneous day-$t$ indicators are fully observed.

---

### 2.5 Clarified Target Terminology
- Directional target formulated as:
  $$y_t = \begin{cases} 1 & \text{if } Close_{t+1} > Close_t \quad (\textbf{NextDay\_Up}) \\ 0 & \text{if } Close_{t+1} \le Close_t \quad (\textbf{NextDay\_NotUp}) \end{cases}$$
- Replaced `NextDay_Down` with **`NextDay_NotUp`** to accurately capture non-positive movements ($Close_{t+1} \le Close_t$). In the S&P 500 post-warm-up dataset, this consists of 810 down days and 1 flat day (`2015-05-18`). In the test split of 357 rows, all 159 instances are strictly down days.

---

## 3. Master Document Synchronization Status

All planning deliverables have been synchronized with these resolutions:
1. `planning/FINAL_IMPLEMENTATION_SPECIFICATION.md` — Updated with exact partition counts, 24 tokens, 3 representations, target terminology.
2. `planning/DATASET_ANALYSIS_PLAN.md` — Updated with exact partition dates, 24 tokens, 3 representations.
3. `planning/ARCHITECTURE.md` — Updated with interim/processed Parquet and JSON schemas for Rep A, B, C1, and C2.
4. `planning/TEST_PLAN.md` — Updated TEST-PR-03, TEST-PR-04 ($1425+1+357=1783$), TEST-FE-01, TEST-FE-04 (24 tokens), 25 criteria.
5. `planning/REQUIREMENTS_TRACEABILITY.md` — Updated REQ-01 (Rep A), REQ-02 (Rep B & C), REQ-05 (predictive vs contemporaneous), partition counts.
6. `planning/IMPLEMENTATION_ROADMAP.md` — Updated Phases 2, 3, 4, 5, 6, 7, 8, 9 with exact counts and vocabulary.
7. `planning/MVP_SPECIFICATION.md` — Updated MVP boundaries to reflect exact counts and vocabulary.
8. `planning/AUDIT_RECONCILIATION.md` — Current consistency audit log.
9. Workspace artifact `master_implementation_plan.md` — Master plan updated and synced.
