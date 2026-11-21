# MISSING REQUIREMENTS AND COVERAGE GAPS

The requirement texts come from `planning/REQUIREMENTS_TRACEABILITY.md` (REQ-01 … REQ-12). The exam paper itself is not in the project folder. **Confirm these texts against the original question paper.** REQ-09 is missing from the RTM's numbering, so either a requirement was dropped or the numbering is wrong. Check this first.

| Req | Coverage in Gemini plan | Gap | Severity | Fix |
|---|---|---|---|---|
| **REQ-09 (missing ID)** | — | RTM jumps from REQ-08 to REQ-10 | **HIGH** | Recover the requirement from the exam paper |
| REQ-01 Exhaustive descriptive analytics | Moments, JB, correlation | Statistics would be computed on sentinel-corrupted columns (Gold mean −2%/day). No per-market table. Units are not stated | HIGH | Compute after sentinel cleaning. Add a five-market table, a units column, and missingness by year |
| REQ-02A Noisy data removal | NaN + warm-up | **Sentinel 0 / −1.0 values are not handled** (55+ corrupted rows) | **CRITICAL** | DATASET audit §3 |
| REQ-02B Task-relevant attributes | MI + correlation | Must be train-only. Level features must be excluded for NB | HIGH | DATASET P-3 |
| REQ-02C Transformation | scaling + binning | Transaction items defined wrongly (TE2 degenerate, Gold sentinel, same-day target) | HIGH | TRANSACTION review §4 |
| REQ-03 ≥ 10 unique plots "discussed in the course" | 12 plots | Some (hexbin, ACF/PACF, underwater drawdown) may **not** be course-taught types. The course labs list scatter, histogram, pie, bar, line, box, violin, heatmap, pairplot, dendrogram, k-distance, PCA scatter | MEDIUM | Make sure ≥ 10 plots come from the course's taught types. Keep the others as extras |
| REQ-04 Inferences for every plot | yes | Several inferences are **pre-written** ("capturing volatility regimes", "demonstrating Amdahl's law") | MEDIUM | Write inferences only after the figure exists |
| REQ-05A FP-Growth rules | yes | No predictive-vs-contemporaneous separation. No conviction formula | HIGH | ALGORITHM audit A |
| REQ-05B NB / DBSCAN "whichever is relevant" | both | A relevance justification is required; state it explicitly | LOW | ALGORITHM audit D (last row) |
| REQ-05C Pincer MFIs | yes | Recovery procedure missing. Wrong citation | **CRITICAL** | ALGORITHM audit B |
| REQ-06 Outputs incl. dendrograms | HAC companion | "Classification rules" — NB has no rules. Show per-class feature likelihood tables (μ, σ per class) and the top-|log-likelihood-ratio| features as the "rules" | MEDIUM | Add an NB parameter table |
| REQ-07 Two techniques on PySpark | FPGrowth + NB | NB default modelType fails on negative inputs | **CRITICAL** | PYSPARK §1 |
| REQ-08 Performance enhancement from PySpark | benchmark | Replication-based synthetic data is invalid. Crossover pre-asserted. Memory measurement wrong | HIGH | PYSPARK §2 |
| REQ-10 Interpretations and conclusions | yes | Must include the negative/near-null findings (NB ≈ baseline; predictive-rule lift ≈ 1) | MEDIUM | |
| REQ-11 PDF with parameters (support, confidence…) | ReportLab | "18 mandatory chapters" has no source. The test's PDF size > 2 MB is arbitrary | LOW | Define the chapter list from the exam requirements |
| REQ-12 Source files | yes | — | — | Include `requirements.lock`, data SHA-256s, and the seed in the README |

## Internal inconsistencies in Gemini's plan
1. "At least 10 plots" (exam) vs "exactly 12" (acceptance criteria) vs "fig01–fig17" (architecture).
2. NB tolerance ±1% (roadmap) vs ±2% (test plan).
3. "1,784 transactions" (TEST-PS-05) vs 1,983 labelled rows (dataset plan). Both are correct for different subsets; state which subset each step uses.
4. "Lin & Kedem" (architecture) vs "Lin & Dunham" (master brain).
5. Plot 10 hexbin (USD vs Gold) and Plot 7 (yield spreads) are directly corrupted by the sentinels.
6. "Plot 11 testing day-of-week anomalies": a violin plot does not test anything. Add Kruskal–Wallis.

## Statistical formula checks for the visualizations
- Annualised volatility = σ_daily·√252, on the rolling 30-day (or 21-day) std of `mom`. Correct if `mom` is used, **not** `Close`.
- Drawdown_t = W_t / max_{s≤t} W_s − 1, where W_t = ∏(1 + mom). Correct formula. Do not use `Close` (same thing, but the dividends caveat applies to neither).
- Normalised performance = 100·Close_t / Close_{t0}, with t0 a common start date for all five indices.
- Jarque–Bera = n/6·(S² + (K−3)²/4). Report the p-value. Expect rejection.
- ACF of r vs r²: add Ljung–Box Q at lags 10 and 20.
- Correlation heatmap: Spearman is preferred for fat-tailed returns. Exclude levels (spurious correlation of trending series).
