# Final Verification Report — CNNpred Big Data Endsem

Date: 2026-11-21. The project was independently re-verified against the exam question (IIITDM 25CS5004/CS5010 Endsem, Nov 2025). The previous audit reports were not trusted as proof; every result below comes from a fresh run.

## 1. What was checked
- All `src/` modules, all 11 notebooks, the tests, the output tables and figures, the PDF, README/PROGRESS/traceability.
- The 8 requested methodology points:

| # | Point | Verdict |
|---|---|---|
| 1 | Majority baseline from the 1,425 training labels only, evaluated on the 357 test labels | **Correct.** Train majority = Up (774/1,425). Test accuracy 198/357 = 55.46%. Now also covered by an independent test. |
| 2 | FP-Growth thresholds 143 (N = 1,784) and 114 (N = 1,425) | **Correct.** Tested. |
| 3 | Predictive rules mined on train only, scored on the untouched test period | **Correct.** |
| 4 | Unchanged close → NextDay_NotUp | **Correct.** Tested on all 1,783 rows. |
| 5 | Pincer Search | **Correct.** Matches mlxtend `fpmax`, the maximal FP-Growth itemsets, and brute force on 60 random databases plus a textbook example (new tests). |
| 6 | t-SNE | Implemented and plotted. The earlier claim that it "prevents data snooping" was overstated: t-SNE is transductive (it embeds all days). The report now says so. |
| 7 | Leakage | Scalers, quantiles, NB parameters and the baseline are fitted on train only. One minor future-looking backward fill in `volatility_20` was removed. |
| 8 | Statistical claims / EMH | The earlier text overstated the results ("rules do not retain in-sample arbitrage under market efficiency"). The report now says the results are consistent with weak predictability and **do not prove** the EMH. |

## 2. What was corrected
1. **Transaction bugs (serious):**
   - `Mom5_*` read a non-existent column `mom5`, so every day became `Mom5_Neg`. It now reads `ROC_5`.
   - `USD_*` read the non-existent column `USD`, so the item never occurred. It now reads `Dollar index`.
   - Asia used the non-existent column `N225`, so it was HSI only. It is now HSI and Nikkei-F.
   - Tech used AAPL and AMZN; it is now AAPL and MSFT. Europe now uses FTSE and DAX-F.
   - Zero or missing returns produced a fake "Down" item. They now produce no item.
   - Transactions are now built from Rep A (NaN preserved) instead of the zero-filled Rep B.
2. **Sentinels:**
   - On bond-holiday days the spread columns TE*/DE*, DGS5 and DAAA stayed at 0.0. They are now NaN.
   - On 2017-11-10 DAAA and DBAA are 0 while Treasuries are present. That row is now handled.
   - Genuine 0.00% T-bill yields (2011–2015) were wrongly deleted. They are now kept.
3. **PCA scree plot:** it was computed on unscaled data, so the raw Close and EMA levels dominated it. It is now computed on standardised stationary features, fitted on train only.
4. **Correlation heatmap:** it silently dropped the non-existent columns `mom5` and `USD`, and it mixed trending price levels. It now uses stationary series only.
5. **Spark FP-Growth:** it was run on a different database from mlxtend. It now runs on the same C1 database. Exact parity is checked (table_19): 1,223 itemsets and 3,463 rules are identical.
6. **Benchmark:**
   - Added a real local-vs-Spark comparison at 1,784 to 250,000 transactions (fig_14).
   - Strong scaling now uses 200,000 transactions instead of 1,783.
   - An outcome-forcing test (`f > 0`) was replaced.
7. **Clustering:**
   - The "DBCV" metric was relabelled as a noise-penalised silhouette heuristic; it is not true DBCV.
   - The Ward dendrogram now samples days across the whole period instead of using the first 500 days.
8. **Classification outputs:** added balanced accuracy, MCC, the exact binomial test, and confusion matrices (fig_13).
9. **Windows Spark start-up failure** (Java 17 Unix-socket temp path): fixed in `config.py`.
10. **Tests:** added `tests/test_correctness.py` with 69 first-principles tests. Two old tests asserted wrong behaviour; both were corrected.
11. **PDF:** rebuilt as a 20-chapter, 36-page report with a table of contents, 20 numbered figures (including the architecture, workflow and algorithm-flowchart diagrams), 20 numbered tables and page numbers. All numbers are read from the output tables at build time.

## 3. Final results (S&P 500)
- FP-Growth, contemporaneous (N = 1,784): 1,223 itemsets, 5,222 rules, 4,041 significant after BH correction, max lift 2.67.
- FP-Growth, predictive (train N = 1,425): 481 rules. None is significant after BH correction. On the test period 21 of 481 reach p < 0.05, which is what chance would produce.
- Pincer Search: 444 MFIs. Identical to both oracles.
- Majority baseline: 55.46% accuracy.
- Bernoulli NB: 57.42% accuracy, AUC 0.557. The gain over the baseline is **not significant** (McNemar p = 0.56).
- Gaussian NB: 50.14% accuracy, **significantly worse** than the baseline (McNemar p = 0.023).
- DBSCAN: one dense regime of 1,739 days plus 45 stress/noise days.
- PySpark: NB parity is 357/357 predictions and FP-Growth is exactly identical. **No speed-up** over local Python on one 12-core machine at these sizes; the measured reasons are explained in report chapter 15.
- Tests: **95 passed, 0 failed**. All 11 notebooks executed without errors.

## 4. Remaining limitations (stated honestly)
- **Performance enhancement:** the exam asks to "establish the performance enhancement" from PySpark. On one laptop, with a 24-item vocabulary, Spark was slower than mlxtend, and adding cores did not help (fitted Amdahl f ≈ 0). This is reported as measured, not hidden. A real speed-up would need a multi-node cluster or much larger, wider data.
- **Predictive results:** next-day prediction is weak. That is a property of the data, not a bug, and it is not inflated.
- **Evaluation design:** there is a single test window. Contemporaneous mining is full-sample, because it is descriptive.
- **Benchmark timings** come from the run immediately before the final sentinel fix. That fix changed one row and does not affect the synthetic benchmark workloads.
