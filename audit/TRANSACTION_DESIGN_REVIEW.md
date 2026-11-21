# TRANSACTION DESIGN REVIEW — FP-Growth / Pincer Search item vocabulary

Scope: the 12-row table in `DATASET_ANALYSIS_PLAN.md` §6.
All supports below are on the **training rows** (S&P, rows 200 … split−1, n = 1,426) **[V]**.

## 1. Price levels vs returns — answering the core question

**Is `USD_Appreciating` (Dollar index > 0) mathematically meaningful?**
**Yes.** The column is a **daily percent change**, not the DXY level.
- Evidence [V]: mean 0.010, std 0.495, range −4.28 … +3.98, lag-1 autocorrelation −0.039. A level series would have autocorrelation ≈ 1 and would sit near 80–100.
- A sign threshold on a return is the correct definition of "appreciated today".
- Gemini reached the right construction **without verifying** the column type. Its dictionary also does not distinguish level columns from return columns, and that distinction is what makes the rule valid. If the column had been a level, "> 0" would be true on 100% of days and the item would be degenerate.

**`Gold_Rally` / `Oil_Rally` (return > 0)?**
The sign construction is meaningful (both columns are returns, autocorrelation ≈ −0.02/−0.03). But the plan as written is **corrupted**:
- Gold = −1.0 on 30 training days and 0 on 48. Oil has one −1.0. These are holiday sentinels (DATASET audit D-7/D-8). They are silently coded as `Gold_Decline`.
- "Rally" overstates a move of +0.01%. A sign split is "up day", not "rally".
- **Correct definition:** `Gold_Up` if r > 0, `Gold_Down` if r < 0, no item if r == 0 or the value is NaN or a sentinel. Optional: a magnitude item `Gold_BigUp` when r > Q80_train(r).

**Items that *are* level-based** — these need different treatment:
- `Close − EMA_50` → a level-vs-level comparison. Meaningful (a stationary sign).
- `TE2`, `DE1` → rate levels. Thresholding a *level* describes a **state**, not a **change**.

## 2. Item-by-item verdict

| # | Gemini item | Train support [V] | Verdict | Severity | Corrected definition |
|---|---|---|---|---|---|
| 1 | `Market_Up/Down` from `mom_t > 0` | 0.543 | **Mislabelled use.** `mom_t` is *today's* return. Rules "X_t ⇒ Market_Up_t" describe same-day co-movement, not prediction. | HIGH | Keep `SameDay_Up/Down` (mom_t). **Add `NextDay_Up/Down` = target y_t** for predictive rules. Only rules with consequent `NextDay_*` may be called "predictive". |
| 2 | `High_Volume_Surge` Volume > Q75 | 0.250 | OK, but Volume is ΔV/V (D-3). | LOW | `Volume_Jump` = Volume > Q75_train. Optional `Volume_Drop` < Q25_train. |
| 3 | `ShortTerm_Momentum_±` ROC_5 sign | 0.589 | OK. Overlaps item 1 (mom_t is part of ROC_5), so expect high-lift, trivial rules. | LOW | Keep. Flag rules that contain both as mechanical. |
| 4 | `Above/Below_50EMA` | 0.737 | OK (level vs level). | — | Keep. Note 74% support: almost every frequent itemset will contain it. |
| 5 | `Yield_Curve_Inverted/Normal` TE2 < 0 | **0.000** | **Degenerate.** TE2 is never negative between 2010 and 2017. The curve did not invert in this sample; the only values ≤ 0 are the holiday sentinels. `Yield_Curve_Normal` has support 1.0 and adds only noise. | **HIGH** | Drop the inversion item. Use the **slope regime**: `TermSpread_Steep` (TE2 > Q67_train) / `TermSpread_Flat` (TE2 < Q33_train), or the **change**: `TermSpread_Widening` (ΔTE2 > 0). Remove the sentinels first. |
| 6 | `Credit_Spread_Widening/Tight` DE1 > median | 0.488 | **Misnamed.** "Above median level" is not "widening". Over 2010–17, DE1 trends downward, so a train median splits the sample mainly by calendar time, and the test period is almost all "Tight". | MEDIUM | State item: `CreditSpread_High/Low` (DE1 vs median_train). Change item: `CreditSpread_Widening` = ΔDE1_t > 0 (ΔDE1 = 0 → no item). Report the test-period support so the drift is visible. |
| 7 | `USD_Appreciating/Depreciating` | 0.500 | Valid (return). The 1.8% zeros go to "Depreciating" by mistake. | MEDIUM | `USD_Up` r > 0, `USD_Down` r < 0, none if 0/NaN. |
| 8 | `Gold_Rally/Decline` | 0.478 | Corrupted by 40 −1.0 sentinels. | **HIGH** | See §1. |
| 9 | `Oil_Rally/Decline` | 0.499 | One sentinel. Valid otherwise. | LOW | `Oil_Up/Down` with the zero/sentinel rule. |
| 10 | `Tech_Leaders_Advancing` AAPL & MSFT > 0 | 0.318 | **Tautological with item 1.** AAPL and MSFT are large S&P constituents. Same-day P(Market_Up \| Tech) = **0.837** [V], but **next-day P(Up \| Tech) = 0.515**, which is *below* the 0.543 base rate. The "strong rule" is index arithmetic, not insight. | **HIGH** | Keep only as a contemporaneous item. Exclude it from rules whose consequent is `SameDay_*` **or** say explicitly that such rules are mechanical. |
| 11 | `European_Markets_Green` FTSE & DAX-F > 0 | 0.407 | Mostly valid. European markets close ~11:30 ET, *before* the US close, so this information is available on the same day. FTSE has 71 NaNs → no item on those days. | LOW | Keep. Put the NaN → absent rule in the docs. |
| 12 | `Asian_Markets_Green` HSI & Nikkei-F > 0 | 0.311 | Valid timing (Asia closes before the US opens). Nikkei-F settlement timing is not documented by the dataset. | LOW | Keep. Note the timing assumption. |

**Asymmetry note:** items 10–12 have no negative counterpart. Item absence ≠ "declining". That is acceptable if the report says so.

## 3. Structural issues with the transaction representation

### T-1 · HIGH · Complementary pairs generate mutually exclusive items
- Every day contains exactly one of {X_Up, X_Down} for every paired attribute. The pair itself therefore has support 0.
- Consequences:
  - About half the 2-item candidates are trivially infrequent.
  - Pincer's MFCS splits immediately on every pair. That helps convergence, but it doubles the item vocabulary.
  - Rules of the form X_Up ⇒ ¬X_Down do not exist, but rules of the form `{A_Down} ⇒ {B_Down}` and `{A_Up} ⇒ {B_Up}` are near-duplicates.
- **Recommendation:** Keep both polarities; they are needed to discover "down" rules. In the report, pair each rule with its mirror rule. Deduplicate by reporting the higher-lift member of each mirror pair.

### T-2 · MEDIUM · Transaction width is nearly constant (10–12 items)
- These are dense, attribute-value transactions, not sparse market baskets. Supports of single items are ≈ 0.3–0.74, so at minsup = 0.10 the expected frequent-itemset length is about 3–5.
- Gemini's test `len(tx) >= 5` is fine. Do not assert a specific number of itemsets in advance.

### T-3 · MEDIUM · Threshold leakage
- Sign thresholds (0) are data-independent and leak-free.
- Quantile/median thresholds (Volume Q75, DE1 median, TE2 terciles) must come from training rows only, and the **frozen values** must be applied to test rows.
- ARM is descriptive, so mining on the full post-warm-up sample is acceptable **if** the thresholds are still train-fitted and you report that choice. To show rule stability, mine train and test separately and compare support/confidence.

### T-4 · LOW · One-hot vs list representation
- mlxtend needs a boolean DataFrame built with `TransactionEncoder`.
- Spark needs `array<string>` with unique items per row. Spark raises on duplicates; sets guarantee uniqueness.
- The vocabulary and the ordering must be identical in both representations for the equivalence tests.

## 4. Final corrected item vocabulary (≈ 22 items, all leak-free)

```
SameDay_Up / SameDay_Down          : mom_t > 0 / < 0              (contemporaneous)
NextDay_Up / NextDay_Down          : y_t = 1 / 0                  (consequent for predictive rules; omit on last row)
Volume_Jump                        : Volume_t > Q75_train
Mom5_Pos / Mom5_Neg                : ROC_5_t > 0 / < 0
Above_EMA50 / Below_EMA50          : Close_t > EMA_50_t / <
TermSpread_Steep / TermSpread_Flat : TE2_t > Q67_train / < Q33_train   (sentinels → NaN, ffill)
CreditSpread_High / CreditSpread_Low: DE1_t > median_train / <=
CreditSpread_Widening              : DE1_t − DE1_{t−1} > 0
USD_Up / USD_Down                  : Dollar index_t > 0 / < 0 (0, NaN → none)
Gold_Up / Gold_Down                : Gold_t > 0 / < 0 (−1.0, 0, NaN → none)
Oil_Up / Oil_Down                  : Oil_t > 0 / < 0 (−1.0, 0, NaN → none)
Tech_Leaders_Up                    : AAPL_t > 0 AND MSFT_t > 0
Europe_Up                          : FTSE_t > 0 AND DAX-F_t > 0 (any NaN → none)
Asia_Up                            : HSI_t > 0 AND Nikkei-F_t > 0 (any NaN → none)
```

**Rule-reporting policy:**
- (a) **Predictive rules:** consequent ∈ {NextDay_Up, NextDay_Down}. The antecedent must not contain NextDay_*.
- (b) **Co-movement rules:** any other rule, labelled "contemporaneous".

Expect predictive rules to have lift close to 1. NB is expected to be near chance, and the market is close to efficient at a 1-day horizon. Report that result honestly; do not hunt for thresholds until lift > 1 appears.
