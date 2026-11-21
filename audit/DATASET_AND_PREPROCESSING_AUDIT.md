# DATASET AND PREPROCESSING AUDIT — CNNpred (UCI 554)

**Auditor role:** Independent verification of `planning/DATASET_ANALYSIS_PLAN.md`, `ARCHITECTURE.md`, `IMPLEMENTATION_ROADMAP.md`.
**Method:** I downloaded the official UCI archive myself (`static/public/554/...zip`, 2.80 MB) into a scratch area outside the project. I ran independent pandas checks on all five CSVs. Every number marked **[V]** below was reproduced by me. **[G]** marks a claim from Gemini that I could not reproduce or that does not apply.

Severity scale: **CRITICAL** (invalidates results) · **HIGH** (materially biases results / examinable error) · **MEDIUM** (defensibility issue) · **LOW** (cosmetic / optional).

---

## 1. What Gemini got right (verified)

| Claim | Status | Evidence |
|---|---|---|
| 5 files, 1,984 rows × 84 columns each | **[V] correct** | `df.shape == (1984, 84)` for all five |
| Date range 2009-12-31 → 2017-11-15, strictly increasing, 0 duplicates | **[V] correct** | `is_monotonic_increasing=True`, `duplicated().sum()==0` |
| Gap distribution {1:1556, 2:16, 3:357, 4:53, 5:1} | **[V] correct** | `Date.diff().dt.days.value_counts()` |
| 82 float64 + 2 string columns (`Date`, `Name`) | **[V] correct** | dtype counts |
| Null totals: 3,029 (S&P/NASDAQ/NYSE/RUSSELL), 3,031 (DJI) | **[V] correct** | |
| 45 columns with explicit NaN, per-column counts | **[V] correct** | Counts match exactly (CAD 483, JPY 479, … EMA_200 199) |
| 80 common columns, 86 in the union; peer naming `GSPC` vs `S&P` | **[V] correct** | Set intersection/union |
| No target column; S&P up-fraction 54.61% (1,083/1,983) | **[V] correct** | 1 flat day (diff == 0) is labelled 0 |
| Peer-index columns are same-day returns of the peer index | **[V] correct (Gemini implied it but did not prove it)** | `S&P.csv['DJI'] == DJI.csv['mom']` exactly (max diff 0.0); `DJI.csv['GSPC'] == S&P.csv['mom']` exactly |
| `mom` = 1-day simple return; `mom1` = `mom` lagged 1 day; `ROC_k` = 100·(C_t/C_{t−k} − 1) | **[V] correct** | max abs diff ≈ 5e-11 |
| `EMA_k` are price **levels** (EWM, adjust=False) | **[V] correct** | EMA_10 vs `Close.ewm(span=10, adjust=False)` mean abs diff 2.7e-4 |
| Shared macro/FX/commodity columns are identical across the five files | **[V] correct** | max abs diff 0 for all 64 non-index-specific common columns |

The "1,984 not 1,985" remark about the UCI landing page is **[G] unverified**. I did not check the landing-page text. Drop the remark or cite the page.

---

## 2. Verified errors in Gemini's data dictionary

### D-1 · HIGH · Five spread formulas are wrong
Gemini defines TE5 = DGS10−DGS5, TE6 = DGS5−DTB3, DE4 = DBAA−DTB3, DE5 = DAAA−DGS10 and DE6 = DAAA−DTB3. I tested each formula against the data, and these five do not match. The formulas that do reproduce the columns (max abs diff < 1e-15) are:

| Column | **Correct definition [V]** | Gemini's (wrong) definition |
|---|---|---|
| TE1 | DGS10 − DTB4WK | same ✓ |
| TE2 | DGS10 − DTB3 | same ✓ |
| TE3 | DGS10 − DTB6 | same ✓ |
| **TE5** | **DTB3 − DTB4WK** | DGS10 − DGS5 ✗ |
| **TE6** | **DTB6 − DTB4WK** | DGS5 − DTB3 ✗ |
| DE1 | DBAA − DAAA | same ✓ |
| DE2 | DBAA − DGS10 | same ✓ |
| **DE4** | **DBAA − DTB6** | DBAA − DTB3 ✗ |
| **DE5** | **DBAA − DTB3** | DAAA − DGS10 ✗ |
| **DE6** | **DBAA − DTB4WK** | DAAA − DTB3 ✗ |

**Implementation impact:** You must fix the dictionary and the report. Any economic interpretation of TE5/TE6 (for example, "10Y−5Y slope") is wrong.

### D-2 · HIGH · `CTB3M`, `CTB6M`, `CTB1Y` are relative changes, not yields
- **Evidence [V]:** CTB3M mean −0.0017, std 0.143, 42.4% exact zeros, min −1.0. A yield level could not look like this. The first rows hold −0.100, −0.0556, −0.1176, which are ratios of consecutive rates.
- **Correction:** Describe these columns as "daily relative change of the Canadian T-bill rate". Do not call them levels.

### D-3 · MEDIUM · `Volume` is the daily relative change of volume, not a z-score
- **Evidence [V]:** mean 0.0186, std 0.212, min −0.761, max 3.23, about 49% positive. That shape is consistent with ΔV/V. A z-score would have mean 0 and std 1.
- **Correction:** Call it "volume change". The `High_Volume_Surge` item then means "large *increase* in volume vs. the previous day", not "high volume".

### D-4 · MEDIUM · Units are mixed: decimal fraction vs percent
- **Evidence [V]:**
  - Decimal-fraction returns (std ≈ 0.005–0.016): `mom`, the peer indices, `AAPL`…`XOM`, `FCHI`, `FTSE`, `GDAXI`, `HSI`, `SSEC`, `GBP`, `JPY`, `CAD`, `CNY`.
  - Percent returns (std ≈ 0.5–2.7): `Dollar index`, `Dollar index-F`, `EUR`, `AUD`, `NZD`, `CHF`, `Brent`, `WIT-oil`, all `*-F` futures, `XAU`, `XAG`.
- **Impact:** Sign-based items are unaffected. Distance-based methods (DBSCAN, PCA) are dominated by the percent-scale columns unless you scale them. Descriptive tables must state the units for each column.

### D-5 · LOW · Other dictionary text
- "Rows 0 to 198" for EMA_200 is correct.
- The claim "Volatility: Target Index Volume" mislabels volume as volatility.
- `Name` for NYSE is `NYA`. Gemini lists this correctly.

---

## 3. CRITICAL hidden missingness that Gemini missed (sentinel values)

Gemini's audit counts only `NaN`. The data also encodes missing observations as **0** and as **−1.0**. A −1.0 return means a −100% move, which is impossible for an index, a commodity or a rate.

### D-6 · CRITICAL · US rate block = 0 on US bond-market holidays
- **Evidence [V]:** On 15 dates, DGS10 = DTB3 = DAAA = DBAA = 0 and all TE/DE spreads are 0. All 15 dates are Columbus Day or Veterans Day (2010-10-11, 2010-11-11, … 2017-10-09).
- **Impact:** Each of those days gets a fake one-day crash of the yield curve to 0 bp. That corrupts the TE2 sign, the DE1 median, descriptive statistics, Plot 7 (spurious spikes to zero), DBSCAN (these days become guaranteed "noise" outliers) and NB Gaussian variances.
- **Correction:** For each column in {DTB4WK, DTB3, DTB6, DGS5, DGS10, DAAA, DBAA, TE*, DE*}, set rows where DGS10 == 0 (or a row-wise all-zero rate block) to NaN, then forward-fill. Forward-fill is correct here because these are **levels**.

### D-7 · CRITICAL · `Gold` = −1.0 on 40 London bank holidays
- **Evidence [V]:** Gold = −1.0 on 40 dates: Easter Monday, early-May, spring and late-August bank holidays, Christmas Eve, and New Year's Eve, every year. Gold also has 65 exact zeros. Gold's mean is −0.020/day **only because of these artefacts**, while its max is +0.0496. Gemini's own descriptive table would report gold as falling 2% per day on average.
- **Impact:** Every "Gold" statistic, Plot 10 (the hexbin axis would be stretched to −100%), the DBSCAN outliers and the `Gold_Rally/Decline` item are corrupted.
- **Correction:** Set `Gold == -1.0` to NaN. Then treat it as a *return* (see §4).

### D-8 · HIGH · `Oil` −1.0 (2017-07-03) and `CTB*` −1.0 on Canadian holidays
- **Evidence [V]:**
  - Oil has one −1.0, on the day before US Independence Day.
  - CTB3M is −1.0 on 15 dates (Canadian Thanksgiving / Remembrance Day), and CTB6M/CTB1Y behave the same way.
- **Correction:** Use one sentinel rule for all relative-change columns: a value ≤ −0.99 becomes NaN.

### D-9 · MEDIUM · Exact zeros in return columns
- **Evidence [V]:** `Dollar index` has 35 zeros (1.8%), XAG 1.8%, CNY 3.2%. Some of these are genuine "no change" days and some are holidays. They cannot be told apart reliably.
- **Correction:** Keep the zeros. For transaction items, map 0 to *no item* (neither Up nor Down) rather than to "Down" (see TRANSACTION_DESIGN_REVIEW).

---

## 4. Preprocessing decisions — audit

### P-1 · Removal of the first 200 observations — **ACCEPT, with a corrected justification**
- EMA_200 is NaN for rows 0–198 [V]. The first fully non-null row is index 201, but that is driven by FX and other nulls, not by the warm-up.
- Dropping rows 0–199 is defensible. The precise minimum is 199 rows. After dropping 200, 1,784 rows remain (1,783 labelled).
- **Caveat [V]:** The dataset's EMAs are seeded at the start of the file (adjust=False). EMA_200 at row 199 still carries initialisation bias, whose weight decays as (199/201)^n. Say this in the report. It is an approximation, not an exact "fully conditioned" state.
- The drop must be applied **before** the chronological split. Every train-derived threshold is then computed on rows 200…split−1.

### P-2 · Blanket forward-fill — **REJECT as specified (HIGH)**
Gemini says forward-fill "replicates real-world trading reality" for every column. That is true only for **levels**. Most of this dataset consists of **returns**.

| Column type | Examples | Correct imputation | Why |
|---|---|---|---|
| Price/rate **levels** | Close, EMA_*, DTB*, DGS*, DAAA, DBAA, TE*, DE* | forward-fill (after the sentinel→NaN step) | Last known level is the information set |
| **Returns / relative changes** | mom*, ROC*, FX, commodities, futures, foreign indices, bellwethers, peer indices, Volume, CTB* | **fill with 0** (a closed market has no price change) **or leave NaN and let the item be absent** | Forward-filling a return duplicates yesterday's move. It creates fake autocorrelation and double-counts shocks. |

- **Evidence that this matters [V]:** SSEC has a run of 57 consecutive NaNs in 2017. A forward-fill would copy one day's Shanghai return 57 times. FX columns are NaN about 60 days per year (24%).
- **Recommendation:**
  - For NB and DBSCAN: fill returns with 0. Add a binary `<col>_missing` indicator only if analysis shows the missingness pattern is informative.
  - For transactions: NaN means the item is absent.
  - `bfill` stays prohibited. Gemini is correct on that.

### P-3 · Feature selection — **ACCEPT the concept; specify it concretely (MEDIUM)**
- **Must be fitted on train only:** MI ranking, the correlation filter (|r| > 0.90) and any k-best.
- **Mandatory exclusions for NB/DBSCAN [V]:** `Close`, `EMA_10/20/50/200` (non-stationary levels). The 2016–17 test period has S&P 2,000–2,600, which lies outside most of the training range, so a model would extrapolate. Replace these columns with stationary transforms: `Close/EMA_k − 1`.
- **Redundant groups to collapse:**
  - {Gold, XAU, gold-F}
  - {Oil, Brent, WIT-oil}
  - {DTB4WK, DTB3, DTB6}
  - {Dollar index, Dollar index-F}
  - futures vs. their cash index
  - {TE1, TE2, TE3}, which are nearly collinear
- **"k = 20" is arbitrary.** Choose k by time-series cross-validation inside train (`TimeSeriesSplit`). Do not choose it on the test set.

### P-4 · Scaling — **PARTIALLY REJECT (MEDIUM)**
- GaussianNB is invariant to per-feature affine scaling. Each class has its own per-feature μ and σ, and only `var_smoothing` (1e-9 · max variance) is scale-sensitive. RobustScaler before GaussianNB is harmless but unnecessary. Do not claim it improves NB.
- DBSCAN and PCA **require** scaling because of the mixed units (D-4). Use StandardScaler or RobustScaler fitted on **train only**. Clustering is unsupervised, so a full-sample fit is defensible if DBSCAN is purely descriptive. Choose one option and state it.
- BernoulliNB needs binarisation thresholds. Use 0 for returns (sign) or train medians.

### P-5 · Financial return construction — **clarify**
- Most columns are **already** returns. Do not re-difference them; `pct_change()` of a return is meaningless.
- Only levels (Close, EMAs, rates, spreads) may be differenced: Δrate in bp, log or simple return of Close.
- Simple returns are already given. Log returns are only needed for the drawdown / cumulative plots: `cumprod(1+mom)`.

---

## 5. Target construction — **VERIFIED, with two additions**
- y_t = 1[Close_{t+1} > Close_t] is equivalent to 1[mom_{t+1} > 0] [V]. It matches Hoseinzade & Haratizadeh (2019).
- The last row has no label. Drop it for supervised work but keep it for ARM.
- **Boundary purge (LOW):** The label of the last training row uses the first test day's Close. Drop that single row (a 1-day embargo) to make the split strictly non-overlapping.
- **Split facts [V]** (S&P, after dropping 200 rows and the last row, 80/20):
  - Train: 2010-10-18 … 2016-06-16 (1,426 rows), up-fraction 54.28%.
  - Test: 2016-06-17 … 2017-11-14 (357 rows), up-fraction **55.46%**.
  - The majority-class baseline is therefore 55.5% accuracy. See ALGORITHM_CORRECTNESS_AUDIT §NB.

---

## 6. Cross-market consistency [V]
- The 64 macro/FX/commodity/bellwether/foreign columns are byte-identical across files. The five "markets" therefore share all but 16 columns: Close, Volume, 12 technicals, and 4 peers.
- **Implication:** Five-market NB results are **not** five independent experiments. Report them as correlated replications.
- DJI has 2 more NaNs than the others. They are in its own index-specific columns. This does not matter.
- **Harmonisation map:** `GSPC` → `SP500`, `S&P` → `SP500`, `IXIC` → `NASDAQ`, `RUT` → `RUSSELL`, `NYSE` → `NYSE`, `DJI` → `DJI`.

---

## 7. Unsupported empirical claims (to delete or verify)
| Claim | Location | Verdict |
|---|---|---|
| "1,984 not 1,985 as stated on UCI" | DATASET_ANALYSIS_PLAN §1 | Unverified; delete or cite |
| "Volume: standardized z-score" | §3 | False (D-3) |
| "CTB: yields" | §3 | False (D-2) |
| "ffill replicates real-world trading reality" | §4 | False for returns (P-2) |
| "economically intuitive relationships e.g. {USD_Depreciating, Tech_Leaders_Advancing} ⇒ Market_Up" | §6 | A result asserted before any mining was run; delete |
| "0 remaining nulls" as an acceptance criterion | MVP | Not achievable by ffill alone. Leading NaNs after the drop need 0-fill (returns) and the sentinels must be handled first |
