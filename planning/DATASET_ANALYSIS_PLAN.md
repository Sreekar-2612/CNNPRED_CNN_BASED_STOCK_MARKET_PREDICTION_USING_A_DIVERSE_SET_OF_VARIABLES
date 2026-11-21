# DATASET ANALYSIS PLAN & PROFILE: UCI DATASET 554 (CNNpred)
*(Reconciled & Consistent Version)*

**Project:** Analytics and Systems of Big Data Practice — Endsem Hackathon  
**Institution:** IIITDM Kancheepuram  
**Dataset Name:** CNNpred: CNN-based stock market prediction using a diverse set of variables  
**UCI Dataset ID:** 554  
**Direct Download URL:** `https://archive.ics.uci.edu/static/public/554/cnnpred%2Bcnn%2Bbased%2Bstock%2Bmarket%2Bprediction%2Busing%2Ba%2Bdiverse%2Bset%2Bof%2Bvariables.zip`  
**Reference Paper:** Hoseinzade, E., & Haratizadeh, S. (2019). *CNNpred: CNN-based stock market prediction using a diverse set of variables.* Expert Systems with Applications, 123, 363–376.

---

## 1. Verified Source Data Characteristics

Empirical audit of the official UCI ZIP archive (2.67 MB compressed, ~7.29 MB uncompressed) establishes the following ground-truth facts:

| Dataset File | Rows (Instances) | Columns (Features) | Size (Bytes) | Date Range | Total NaN | Duplicates | Pre-existing Target? |
|---|---|---|---|---|---|---|---|
| `Processed_S&P.csv` | 1,984 | 84 | 1,460,021 | 2009-12-31 to 2017-11-15 | 3,029 | 0 | **No** (Must be constructed) |
| `Processed_DJI.csv` | 1,984 | 84 | 1,462,710 | 2009-12-31 to 2017-11-15 | 3,031 | 0 | **No** (Must be constructed) |
| `Processed_NASDAQ.csv` | 1,984 | 84 | 1,465,716 | 2009-12-31 to 2017-11-15 | 3,029 | 0 | **No** (Must be constructed) |
| `Processed_NYSE.csv` | 1,984 | 84 | 1,461,210 | 2009-12-31 to 2017-11-15 | 3,029 | 0 | **No** (Must be constructed) |
| `Processed_RUSSELL.csv` | 1,984 | 84 | 1,459,384 | 2009-12-31 to 2017-11-15 | 3,029 | 0 | **No** (Must be constructed) |

*(Note on UCI Metadata: The UCI JSON API metadata states `"num_instances": 1985`, which counts the header row. The actual data row count is exactly 1,984).*

### Key Empirical Findings:
1. **Instances:** Exactly 1,984 daily rows per CSV file.
2. **Features:** Exactly 84 columns in every file (80 common, 4 cross-market peers varying per file).
3. **Temporal Ordering:** Strictly monotonically increasing daily trading timestamps from `2009-12-31` to `2017-11-15`.
   - 1-day step: 1,556 transitions (consecutive trading days)
   - 3-day step: 357 transitions (weekend gaps: Friday to Monday)
   - 4-day step: 53 transitions (3-day holiday weekends)
   - 2-day step: 16 transitions (mid-week market holidays)
   - 5-day step: 1 transition (Hurricane Sandy market closure, late October 2012)
4. **Data Types:** 82 `float64` columns, 1 string date column (`Date`), and 1 string identifier column (`Name`).

---

## 2. Separation of the Three Data Representations

To preserve statistical and algorithmic correctness, the project enforces three distinct data representations:

### Representation A: Descriptive Analytics Dataset (`df_descriptive`)
- **Rows:** Exactly 1,784 observations (rows 200 to 1,983).
- **Rule:** Sentinels converted to `np.nan`. **Missing values in return and relative change columns are preserved as NaN.**
- **Rationale:** Replacing missing returns with zero distorts variance, skewness, kurtosis, and Jarque-Bera normality tests. Moments and correlations are computed pairwise using `skipna=True`.

### Representation B: Continuous Modeling Dataset (`df_modeling`)
- **Rows:** Exactly 1,784 observations (rows 200 to 1,983).
- **Rule:** Uses approved feature-specific imputation:
  - Level features: Forward-fill (`ffill`).
  - Return and relative change features: Impute with **0.0** (zero change).
- **Rationale:** Estimators (Naive Bayes, DBSCAN, PCA, VectorAssembler) require dense, non-null numeric matrices.

### Representation C: Discrete Transaction Datasets (`transactions_contemp` & `transactions_pred`)
- **Rule:** Unavailable observations (NaN or sentinel) are represented as **item absence** (neither Up nor Down token is added).
- **Contemporaneous Transaction Database:** $N = 1,784$ daily baskets (rows 200 to 1,983), using the 22 contemporaneous tokens.
- **Predictive Transaction Database:** $N = 1,783$ daily baskets (rows 200 to 1,982), using the 22 contemporaneous tokens plus exactly one forward target token (`NextDay_Up` or `NextDay_NotUp`). Row 1,983 is strictly omitted.

---

## 3. Critical Hidden Missingness: Sentinel-Coded Values

Beyond the 3,029 explicit `NaN` values, empirical audit identified non-NaN sentinel missing value codes:

1. **U.S. Rate/Spread Block (= 0.0) on 15 U.S. Bond Market Holidays:**
   - On 15 dates (e.g. Columbus Day, Veterans Day: `2010-10-11`, `2010-11-11`, `2011-10-10`, `2011-11-11` ... `2017-10-09`), U.S. bond markets were closed while equity exchanges traded.
   - Sourced FRED rates (`DGS10`, `DTB3`, `DAAA`, `DBAA`, etc.) and all derived spreads (`TE*`, `DE*`) are recorded as **0.0**.
   - These 0.0 values represent missing holiday data, not genuine zero interest rates or zero spreads.
2. **London Bullion Market Holidays (`Gold` = -1.0) on 40 Dates:**
   - On 40 dates (Easter Monday, Early May, Spring Bank Holiday, Late August Bank Holiday, Christmas Eve, New Year's Eve), UK gold markets were closed.
   - `Gold` return is recorded as **-1.0** (a -100% crash). This corrupted the sample mean to -0.020/day.
3. **Single-Day Commodity & Canadian Holidays:**
   - `Oil` has one `-1.0` value on `2017-07-03` (day before U.S. Independence Day).
   - `CTB3M`, `CTB6M`, `CTB1Y` have `-1.0` on 15 dates (Canadian Thanksgiving / Remembrance Day).

### Mandatory Step 0 Cleaning Action:
- Rate/Spread Block: Set rows where $DGS10 == 0.0$ to `np.nan` across all 17 rate/spread columns.
- Returns & Relative Changes: Set any value $\le -0.99$ to `np.nan` across `Gold`, `Oil`, `CTB3M`, `CTB6M`, `CTB1Y`.

---

## 4. Corrected Column Dictionary & Exact Mathematical Formulas

### Structural Column-Type Classification:
- **`level_price`:** Raw dollar price levels: `Close`, `EMA_10`, `EMA_20`, `EMA_50`, `EMA_200`.
- **`level_rate`:** Yield levels & spreads (percent/bps): `DTB4WK`, `DTB3`, `DTB6`, `DGS5`, `DGS10`, `DAAA`, `DBAA`, `TE1`, `TE2`, `TE3`, `TE5`, `TE6`, `DE1`, `DE2`, `DE4`, `DE5`, `DE6`.
- **`return_frac`:** Daily simple return as decimal fraction ($\text{std} \approx 0.005 - 0.016$): `mom`, `mom1`, `mom2`, `mom3`, equities (`AAPL`…`XOM`), foreign indices (`FCHI`, `FTSE`, `GDAXI`, `HSI`, `SSEC`), currencies (`GBP`, `JPY`, `CAD`, `CNY`), peer index returns.
- **`return_pct`:** Daily simple return as percentage ($\text{std} \approx 0.5 - 2.7$): `Dollar index`, `Dollar index-F`, `EUR`, `AUD`, `NZD`, `CHF`, `Brent`, `WIT-oil`, all `*-F` futures, `XAU`, `XAG`.
- **`rel_change`:** Daily relative ratio change ($\Delta X / X$): `Volume` (relative volume change $\Delta V / V$, not a z-score), `CTB3M`, `CTB6M`, `CTB1Y` (daily relative change of Canadian T-bill rates, not yields), `ROC_5`…`ROC_20` (Rate of Change percentage).
- **`metadata`:** `Date` (ISO-8601 string), `Name` (constant index identifier string).

### Verified Spread Formulas (Empirically Validated Against Dataset):
- `TE1` = $DGS10 - DTB4WK$ (10Y Treasury minus 4-Week Treasury Bill)
- `TE2` = $DGS10 - DTB3$ (10Y Treasury minus 3-Month Treasury Bill)
- `TE3` = $DGS10 - DTB6$ (10Y Treasury minus 6-Month Treasury Bill)
- **`TE5` = $DTB3 - DTB4WK$** (3-Month Treasury minus 4-Week Treasury; short-end money market slope)
- **`TE6` = $DTB6 - DTB4WK$** (6-Month Treasury minus 4-Week Treasury)
- `DE1` = $DBAA - DAAA$ (Moody's Baa Corporate minus Aaa Corporate; credit default risk spread)
- `DE2` = $DBAA - DGS10$ (Moody's Baa Corporate minus 10Y Treasury)
- **`DE4` = $DBAA - DTB6$** (Moody's Baa Corporate minus 6-Month Treasury)
- **`DE5` = $DBAA - DTB3$** (Moody's Baa Corporate minus 3-Month Treasury)
- **`DE6` = $DBAA - DTB4WK$** (Moody's Baa Corporate minus 4-Week Treasury)

---

## 5. Supervised Target & Chronological Partition

- **Target Construction & Terminology:**
  $$y_t = \begin{cases} 1 & \text{if } Close_{t+1} > Close_t \quad (\textbf{NextDay\_Up}) \\ 0 & \text{if } Close_{t+1} \le Close_t \quad (\textbf{NextDay\_NotUp}) \end{cases}$$
  Row $t = 1,983$ (last day) is dropped from supervised sets because $Close_{t+1}$ is unobserved, leaving **1,783 labeled instances**.
- **Chronological 80/20 Partition with 1-Row Boundary Embargo:**
  - **Train Split (80%):** Rows 200 through 1,624 ($N_{\text{train}} = 1,425$ rows; dates `2010-10-18` to `2016-06-15`; up-fraction $\approx 54.25\%$).
  - **Boundary Embargo:** Row 1,625 ($N_{\text{embargo}} = 1$ row; date `2016-06-16`). Purged row: target uses $Close_{1626}$, so features are purged to prevent test boundary overlap.
  - **Test Split (20%):** Rows 1,626 through 1,982 ($N_{\text{test}} = 357$ rows; dates `2016-06-17` to `2017-11-14`; up-fraction = **55.46%**: 198 Up / 159 NotUp).
  - **Sum Labelled:** $1,425 + 1 + 357 = \mathbf{1,783}$.
  - The classification baseline is the **Test Majority Rate (55.46%)**, NOT 50.0%.

---

## 6. Complete 24-Item Transaction Vocabulary Enumeration

The discrete transaction universe comprises **EXACTLY 24 DISTINCT ITEM TOKENS**, split into 22 contemporaneous tokens and 2 forward target tokens:

```
Complete 24-Item Vocabulary Enumeration:
====================================================================================================================
Contemporaneous Tokens (22 tokens, evaluated on day t):
--------------------------------------------------------------------------------------------------------------------
1.  SameDay_Up             : mom_t > 0                                (Daily index return positive)
2.  SameDay_Down           : mom_t < 0                                (Daily index return negative)
3.  Volume_Jump            : Volume_t > Q75_train(Volume)             (Relative volume increase > 75th percentile)
4.  Mom5_Pos               : ROC_5_t > 0                              (5-day Rate of Change positive)
5.  Mom5_Neg               : ROC_5_t < 0                              (5-day Rate of Change negative)
6.  Above_EMA50            : Close_t > EMA_50_t                       (Price above 50-day moving average)
7.  Below_EMA50            : Close_t < EMA_50_t                       (Price below 50-day moving average)
8.  TermSpread_Steep       : TE2_t > Q67_train(TE2)                   (10Y-3M Treasury spread in upper tercile)
9.  TermSpread_Flat        : TE2_t < Q33_train(TE2)                   (10Y-3M Treasury spread in lower tercile)
10. TermSpread_Widening     : TE2_t - TE2_{t-1} > 0                    (Yield curve steepening day-over-day)
11. CreditSpread_High      : DE1_t > Median_train(DE1)                (Baa-Aaa corporate spread above median)
12. CreditSpread_Low       : DE1_t <= Median_train(DE1)               (Baa-Aaa corporate spread below median)
13. CreditSpread_Widening   : DE1_t - DE1_{t-1} > 0                    (Credit default risk increasing day-over-day)
14. USD_Up                 : Dollar index_t > 0                       (U.S. Dollar Index appreciating)
15. USD_Down               : Dollar index_t < 0                       (U.S. Dollar Index depreciating)
16. Gold_Up                : Gold_t > 0                               (Gold bullion return positive)
17. Gold_Down              : Gold_t < 0                               (Gold bullion return negative)
18. Oil_Up                 : Oil_t > 0                                (Crude oil return positive)
19. Oil_Down               : Oil_t < 0                                (Crude oil return negative)
20. Tech_Leaders_Up        : AAPL_t > 0 AND MSFT_t > 0                (Both Apple and Microsoft advancing)
21. Europe_Up              : FTSE_t > 0 AND DAX-F_t > 0               (Both UK and German markets advancing)
22. Asia_Up                : HSI_t > 0 AND Nikkei-F_t > 0             (Both Hong Kong and Japanese markets advancing)
--------------------------------------------------------------------------------------------------------------------
Forward Target Tokens (2 tokens, evaluating forward outcome y_t):
--------------------------------------------------------------------------------------------------------------------
23. NextDay_Up             : Close_{t+1} > Close_t                    (Forward next-day index price increases)
24. NextDay_NotUp          : Close_{t+1} <= Close_t                   (Forward next-day index price decreases/flat)
====================================================================================================================
```

### Transaction Dataset Alignment:
1. **Contemporaneous Database:** $N = 1,784$ baskets ($t = 200 \dots 1,983$). Drawn strictly from the 22 contemporaneous tokens (tokens 1–22). Rules generated here describe market co-movement.
2. **Predictive Database:** $N = 1,783$ baskets ($t = 200 \dots 1,982$). Each basket contains day-$t$ contemporaneous items plus exactly one forward target token (`NextDay_Up` or `NextDay_NotUp`). Row 1,983 is strictly omitted.
