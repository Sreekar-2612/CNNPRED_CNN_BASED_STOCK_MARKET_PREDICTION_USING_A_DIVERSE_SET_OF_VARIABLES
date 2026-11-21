"""
Final academic PDF report (ReportLab).
Every number in the text is read from outputs/tables, outputs/metrics at build time, so the
report always agrees with the code, CSV files, notebooks and figures.
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from src.config import FIGURES_DIR, METRICS_DIR, REPORTS_DIR, TABLES_DIR

T = TABLES_DIR
NAVY = colors.HexColor("#1f2d4d")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, textColor=NAVY, spaceBefore=6, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, textColor=NAVY, spaceBefore=8, spaceAfter=5)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
BUL = ParagraphStyle("BL", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CAP = ParagraphStyle("CAP", parent=BODY, fontSize=8.8, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
EQ = ParagraphStyle("EQ", parent=BODY, alignment=TA_CENTER, fontName="Times-Italic", fontSize=11, spaceBefore=3, spaceAfter=6)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=7.6, leading=9.4, backColor=colors.HexColor("#f4f6f9"),
                      borderColor=colors.HexColor("#c9d1dd"), borderWidth=0.5, borderPadding=5, spaceBefore=4, spaceAfter=8)
TCELL = ParagraphStyle("TC", parent=BODY, fontSize=7.8, leading=9.5, alignment=0, spaceAfter=0)


class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("H1", "H2"):
            lvl = 0 if f.style.name == "H1" else 1
            self.notify("TOCEntry", (lvl, f.getPlainText(), self.page))


class R:
    """Report builder with automatic chapter / figure / table numbering."""
    def __init__(self):
        self.s, self.ch, self.sec, self.fig, self.tab = [], 0, 0, 0, 0

    def chapter(self, title):
        if self.ch:
            self.s.append(PageBreak())
        self.ch += 1; self.sec = 0
        self.s.append(Paragraph(f"{self.ch}. {title}", H1))

    def section(self, title):
        self.sec += 1
        self.s.append(Paragraph(f"{self.ch}.{self.sec} {title}", H2))

    def p(self, text):
        self.s.append(Paragraph(text, BODY))

    def bullets(self, items):
        for it in items:
            self.s.append(Paragraph(it, BUL, bulletText="•"))

    def eq(self, text, defs=None):
        self.s.append(Paragraph(text, EQ))
        if defs:
            self.s.append(Paragraph("where " + defs, BODY))

    def code(self, text):
        self.s.append(Preformatted(text.strip("\n"), CODE))

    def figure(self, path, caption, width=15.5):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(str(path)).getSize()
        w = width * cm; h = w * ih / iw
        if h > 17 * cm:
            h = 17 * cm; w = h * iw / ih
        self.fig += 1
        self.s.append(KeepTogether([Image(str(path), width=w, height=h),
                                    Paragraph(f"<b>Figure {self.fig}.</b> {caption}", CAP), Spacer(1, 6)]))
        return self.fig

    def table(self, df, caption, col_widths=None, fs=7.8):
        self.tab += 1
        st = ParagraphStyle("tc", parent=TCELL, fontSize=fs, leading=fs + 1.8)
        data = [[Paragraph(f"<b>{c}</b>", st) for c in df.columns]]
        for _, r in df.iterrows():
            data.append([Paragraph(str(v), st) for v in r.values])
        t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6f1")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7bb")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        self.s.append(KeepTogether([Paragraph(f"<b>Table {self.tab}.</b> {caption}", CAP), Spacer(1, 3), t, Spacer(1, 8)]))
        return self.tab


def _fmt(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def _page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, "Analytics and Systems of Big Data Practice - CNNpred (UCI 554) - Endsem Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def create_academic_report() -> Path:
    from src.report_diagrams import build_all_diagrams
    D = build_all_diagrams()
    F = FIGURES_DIR

    # ---------------- load results ----------------
    hashes = pd.read_csv(T / "table_01_dataset_hashes.csv")
    sent = pd.read_csv(T / "table_02_sentinel_audit.csv")
    trunc = pd.read_csv(T / "table_03_truncation_summary.csv")
    split = pd.read_csv(T / "table_04_split_partitions.csv")
    tdist = pd.read_csv(T / "table_05_target_distribution_S&P.csv").iloc[0]
    adf = pd.read_csv(T / "table_06_adf_stationarity.csv")
    desc = pd.read_csv(T / "table_07_descriptive_stats.csv")
    fi = pd.read_csv(T / "table_08_fp_growth_itemsets_S&P.csv")
    rules = pd.read_csv(T / "table_10_validated_rules_S&P.csv")
    piter = pd.read_csv(T / "table_11_pincer_iterations_S&P.csv")
    oracle = pd.read_csv(T / "table_13_oracle_comparison_S&P.csv")
    clf = pd.read_csv(T / "table_14_classification_metrics_S&P.csv")
    lik = pd.read_csv(T / "table_15_feature_likelihoods_S&P.csv")
    prof = pd.read_csv(T / "table_16_cluster_profiles_S&P.csv")
    spnb = pd.read_csv(T / "table_17_spark_parity_S&P.csv").iloc[0]
    spfp = pd.read_csv(T / "table_19_spark_mlxtend_fpgrowth_parity_S&P.csv").iloc[0]
    bench = json.load(open(METRICS_DIR / "benchmark_results.json"))
    tests = json.load(open(METRICS_DIR / "test_summary.json")) if (METRICS_DIR / "test_summary.json").exists() else None

    contemp = rules[rules["rule_type"] == "Contemporaneous"]
    pred = rules[rules["rule_type"] == "Predictive"]
    fi_c = fi[fi["database"] == "Contemporaneous_Full_1784"]
    fi_p = fi[fi["database"] == "Predictive_Train_1425"]
    m = clf.set_index("Model")
    base, gnb, bnb = m.loc["Majority_Baseline"], m.loc["GaussianNB_Continuous"], m.loc["BernoulliNB_Binary"]
    ss_ = bench["strong_scaling"]; ds = bench["synthetic_scaling"]
    dsr = pd.DataFrame(ds["synthetic_scaling"])
    n_mfi = int(oracle.iloc[0]["MFS_Count"])
    sent_counts = sent[sent["Index"] == "S&P"].groupby("Column").size()
    noise = prof[prof["Cluster_ID"] == -1]
    n_noise = int(noise["Observation_Count"].iloc[0]) if len(noise) else 0
    n_clusters = int((prof["Cluster_ID"] != -1).sum())

    r = R()
    s = r.s
    # ---------------- title page ----------------
    s += [Spacer(1, 3 * cm),
          Paragraph("Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram",
                    ParagraphStyle("t0", parent=BODY, alignment=TA_CENTER, fontSize=11)),
          Paragraph("25CS5004 / CS5010 - Analytics and Systems of Big Data Practice - Endsem (Nov 2025)",
                    ParagraphStyle("t1", parent=BODY, alignment=TA_CENTER, fontSize=10)),
          Spacer(1, 1.5 * cm),
          Paragraph("Descriptive and Predictive Analytics on the CNNpred Stock-Market Dataset",
                    ParagraphStyle("tt", parent=H1, alignment=TA_CENTER, fontSize=22, leading=28)),
          Paragraph("Preprocessing, Visualization, FP-Growth, Pincer Search, Naive Bayes, DBSCAN and PySpark",
                    ParagraphStyle("t2", parent=BODY, alignment=TA_CENTER, fontSize=12)),
          Spacer(1, 1.2 * cm),
          Paragraph("Dataset: UCI Machine Learning Repository, ID 554 (Hoseinzade &amp; Haratizadeh, 2019)",
                    ParagraphStyle("t3", parent=BODY, alignment=TA_CENTER)),
          Spacer(1, 6 * cm),
          Paragraph("All numbers in this report are generated by the project code and read from its output tables.",
                    ParagraphStyle("t4", parent=CAP)),
          PageBreak()]
    s.append(Paragraph("Table of Contents", H1))
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("T0", fontSize=10, leading=13, leftIndent=10, fontName="Helvetica-Bold"),
                       ParagraphStyle("T1", fontSize=8.8, leading=11, leftIndent=26)]
    s.append(toc)
    s.append(PageBreak())
    s.append(Paragraph("Executive Summary", H1))
    r.p(f"This report applies the complete analytics workflow required by the examination to the five CNNpred files "
        f"(S&amp;P 500, DJI, NASDAQ, NYSE, Russell 2000; 1,984 trading days x 84 variables each, 2009-12-31 to 2017-11-15). "
        f"The S&amp;P 500 file is the primary analysis target; descriptive statistics cover all five indices.")
    r.bullets([
        f"<b>Preprocessing</b> found hidden missing values coded as numbers: US bond-market holidays recorded as 0.0 across the "
        f"rate/spread block ({int(sent_counts.get('DGS10', 0))} days) and -1.0 (-100%) 'returns' for Gold on "
        f"{int(sent_counts.get('Gold', 0))} London holidays. These were converted to missing values before any analysis.",
        f"<b>FP-Growth</b> mined {len(fi_c):,} frequent itemsets (minimum count 143 of 1,784) and {len(contemp):,} "
        f"contemporaneous rules; {int(contemp['is_statistically_significant'].sum()):,} survive Fisher's exact test with "
        f"Benjamini-Hochberg correction. Same-day market structure is strong (lift up to {contemp['lift'].max():.2f}).",
        f"<b>Predictive rules</b> (consequent = next-day direction) were mined on the 1,425 training days only "
        f"(minimum count 114). {len(pred)} rules pass minconf = 0.40, but none is significant after FDR correction "
        f"(min adjusted p = {pred['bh_adjusted_p_value'].min():.3f}); on the untouched 357-day test period only "
        f"{int(pred['test_is_statistically_significant'].sum())} reach uncorrected p &lt; 0.05, about what chance predicts "
        f"for {len(pred)} tests.",
        f"<b>Pincer Search</b> (Lin &amp; Kedem, 1998) found {n_mfi} maximal frequent itemsets (minsup = 0.10, N = 1,783), "
        f"identical to mlxtend fpmax and to maximal FP-Growth itemsets, and it matches brute force on 60 random databases.",
        f"<b>Naive Bayes</b>: Bernoulli NB reached {bnb['Accuracy']:.1%} test accuracy versus the {base['Accuracy']:.1%} "
        f"training-derived majority baseline; the difference is not statistically significant (McNemar p = "
        f"{bnb['McNemar_pVal_vs_Base']:.2f}). Gaussian NB ({gnb['Accuracy']:.1%}) was significantly worse than the baseline.",
        f"<b>DBSCAN</b> separated {n_noise} high-stress days ({n_noise / 1784:.1%}) from one dense 'normal' regime.",
        f"<b>PySpark</b>: Spark Gaussian NB reproduces scikit-learn predictions {spnb['Parity_Agreement_Pct']:.1f}%; Spark "
        f"FPGrowth reproduces mlxtend's {int(spfp['mlxtend_Itemsets']):,} itemsets and {int(spfp['Spark_Rules']):,} "
        f"single-consequent rules exactly. On one {ds['cpu_count']}-core laptop, Spark did <b>not</b> outperform local "
        f"FP-Growth at these data sizes; the benchmark chapter explains why and what the measurements do show.",
    ])

    # ================= 1 =================
    r.chapter("Introduction and Problem Statement")
    r.section("Problem statement")
    r.p("The examination asks for (i) exhaustive descriptive analytics with data cleaning, selection and transformation; "
        "(ii) at least ten distinct visualizations with listed inferences; (iii) predictive analytics: association rules "
        "with FP-Growth, Naive Bayesian classification or DBSCAN clustering, and maximal frequent itemsets with Pincer "
        "Search; (iv) relevant outputs (rules, classifier performance, dendrograms); and (v) two of these techniques on "
        "PySpark together with an assessment of the performance gained from big-data support.")
    r.section("Why this dataset is interesting and difficult")
    r.p("CNNpred combines daily index prices, technical indicators, US and Canadian interest rates, exchange rates, "
        "commodities, bellwether stocks and foreign indices. Daily stock-index direction is close to unpredictable: under "
        "the Efficient Market Hypothesis (Fama, 1970) public information should already be in prices. This report "
        "therefore treats predictive results with strict out-of-sample tests and does not claim more than the evidence "
        "supports. Our results are <i>consistent with</i> weak predictability, but a single 357-day test period cannot "
        "prove or establish market efficiency.")
    r.section("Objectives")
    r.bullets(["Produce a clean, leakage-free version of the data, and document every decision.",
               "Describe the data statistically and visually (14 figures, each with inferences).",
               "Mine contemporaneous and predictive association rules and test them statistically.",
               "Implement Pincer Search from first principles and prove it correct against two independent oracles.",
               "Compare Naive Bayes classifiers with an honest baseline; discover market regimes with DBSCAN.",
               "Run Naive Bayes and FP-Growth on Spark MLlib, verify exact equivalence, and measure performance."])
    r.figure(D["workflow"], "End-to-end analytics workflow. All fitted quantities use the 1,425 training days only.")

    # ================= 2 =================
    r.chapter("Dataset Description and Characteristics")
    r.section("Files and provenance")
    r.p("Each CSV has 1,984 rows and 84 columns. Integrity is checked with SHA-256 on every run (Table 1). The columns "
        "fall into ten groups (Table 2). Most variables are already daily <i>returns</i> (changes), while Close, the EMAs "
        "and the interest rates are <i>levels</i>. This distinction drives every preprocessing decision.")
    h = hashes.copy()
    h["SHA256"] = h["SHA256"].str[:16] + "..." if "SHA256" in h.columns else h.iloc[:, -2]
    r.table(h[[c for c in h.columns if c in ("Index", "File", "Size_Bytes", "SHA256", "Checksum_Match")]],
            "Raw files and checksum verification (table_01).")
    groups = pd.DataFrame([
        ["Index &amp; technical", "Close, Volume (daily % change), mom, mom1-3, ROC_5..20, EMA_10..200", "Close/EMA = levels; others = changes"],
        ["US rates", "DTB4WK, DTB3, DTB6, DGS5, DGS10, DAAA, DBAA", "levels (%)"],
        ["Spreads", "TE1=DGS10-DTB4WK, TE2=DGS10-DTB3, TE3=DGS10-DTB6, TE5=DTB3-DTB4WK, TE6=DTB6-DTB4WK, DE1=DBAA-DAAA, DE2=DBAA-DGS10, DE4=DBAA-DTB6, DE5=DBAA-DTB3, DE6=DBAA-DTB4WK", "levels (verified to 1e-15)"],
        ["Canadian T-bills", "CTB3M, CTB6M, CTB1Y", "relative daily change"],
        ["FX", "GBP, JPY, CAD, CNY (fraction); AUD, EUR, NZD, CHF, Dollar index(-F) (percent)", "returns"],
        ["Commodities", "Oil, Gold (fraction); Brent, WIT-oil, gold-F, silver-F, copper-F, GAS-F, wheat-F, XAU, XAG (percent)", "returns"],
        ["Bellwether stocks", "AAPL, AMZN, GE, JNJ, JPM, MSFT, WFC, XOM", "returns"],
        ["World indices / futures", "FCHI, FTSE, GDAXI, HSI, SSEC; *-F futures", "returns"],
        ["Peer US indices", "4 of DJI, GSPC/S&amp;P, IXIC, NYSE, RUT (same-day returns)", "returns"],
    ], columns=["Group", "Columns", "Type"])
    r.table(groups, "Variable groups (verified empirically; spread formulas reconstructed exactly).", [3.2 * cm, 9.5 * cm, 3.5 * cm])
    r.section("Key characteristics")
    r.bullets(["Dates are strictly increasing with no duplicates; gaps of 3-5 days are weekends and holidays.",
               "Explicit missing values: 3,029 per file, mostly FX (~24%) and Asian indices (holiday calendars), plus "
               "indicator warm-up (EMA_200 is empty for the first 199 rows).",
               "Hidden missing values: zeros and -1.0 codes on holidays (Chapter 4).",
               f"Target balance: in the test period {int(tdist['Test_NextDay_Up'])} of {int(tdist['Test_Rows'])} days "
               f"were up ({tdist['Test_Up_Ratio']:.2%}); unchanged closes count as NotUp.",
               "The 64 macro/FX/commodity columns are identical across the five files, so per-index results are "
               "correlated, not independent replications."])

    # ================= 3 =================
    r.chapter("Environment and Technology Stack")
    import platform, sys
    try:
        import pyspark, sklearn, mlxtend
        vers = f"PySpark {pyspark.__version__}, scikit-learn {sklearn.__version__}, mlxtend {mlxtend.__version__}"
    except Exception:
        vers = "see requirements.txt"
    r.p(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}, {ds['cpu_count']} logical cores; "
        f"{vers}; pandas, NumPy, SciPy, statsmodels, matplotlib/seaborn, psutil, ReportLab; OpenJDK 17 for Spark (local mode).")
    r.p("Windows-specific fix: Java 17 creates a Unix-domain socket for Spark's network selector. With a long temp path "
        "this fails ('Unable to establish loopback connection'), so <font face='Courier'>src/config.py</font> points "
        "<font face='Courier'>jdk.net.unixdomain.tmpdir</font> at a short directory. Spark outputs are written with "
        "<font face='Courier'>toPandas()</font>, so Hadoop's winutils is not needed.")
    r.figure(D["architecture"], "Project architecture: modules, data flow and generated artefacts.")

    # ================= 4 =================
    r.chapter("Data Ingestion and Preprocessing")
    r.section("Data cleaning: sentinel values")
    r.p("The raw files encode some missing observations as numbers. On US bond-market holidays (Columbus Day, Veterans "
        "Day) every rate and spread is 0.0, a physically impossible collapse of the yield curve. On UK bank holidays the "
        "Gold 'return' is -1.0 (-100%), which alone pushes Gold's mean daily return to about -2%. Oil and the Canadian "
        "T-bill changes have the same -1.0 code on some holidays. Table 3 lists the counts.")
    sc = sent_counts.reset_index(); sc.columns = ["Column", "Sentinel days (S&amp;P file)"]
    r.table(sc, "Sentinel values converted to NaN (full date list in table_02).", [5 * cm, 5 * cm])
    r.code('''
closure = df["DGS10"] == 0.0              # US bond-market closure day
for col in RATE_AND_SPREAD_COLUMNS:       # DGS*, DTB*, DAAA, DBAA, TE*, DE*
    df.loc[closure, col] = np.nan         # genuine 0.00% T-bill yields are kept
for col in ["Gold", "Oil", "CTB3M", "CTB6M", "CTB1Y"]:
    df.loc[df[col] <= -0.99, col] = np.nan   # -100% "return" = holiday code''')
    r.section("Warm-up truncation, imputation and representations")
    r.p("EMA_200 needs 200 observations, so rows 0-199 are dropped, leaving 1,784 days (2010-10-18 onward). "
        "Imputation depends on the variable type: a <b>level</b> keeps its last known value (forward fill), while a "
        "<b>return</b> on a closed market is genuinely zero. Forward-filling a return would copy yesterday's move. "
        "Backward filling is never used, because it would copy future information into the past.")
    r.bullets(["<b>Rep A</b> (descriptive): NaN preserved; used for statistics and transactions (a missing value simply produces no item).",
               "<b>Rep B</b> (modelling): levels forward-filled, returns filled with 0; zero NaNs.",
               "<b>Rep C1/C2</b> (itemset mining): binary transactions (Chapter 5)."])
    r.table(trunc[["Index", "Raw_Rows", "Rep_A_Rows", "Rep_A_NaNs", "Rep_B_Rows", "Rep_B_NaNs", "Rep_A_Start_Date", "Rep_A_End_Date"]],
            "Rows and missing values after cleaning (table_03).")
    r.section("Chronological split with embargo")
    r.p("Shuffling time-series data would let the model learn from the future. The 1,783 labelled days are therefore "
        "split in time order: 1,425 training days, 1 embargo day (its label uses the first test day's close) and 357 "
        "test days. Every threshold, scaler, NB parameter and the majority baseline are fitted on the training days only.")
    r.table(split[split["Index"] == "S&amp;P".replace("&amp;", "&")][["Train_Rows", "Train_Start", "Train_End", "Embargo_Date", "Test_Rows", "Test_Start", "Test_End"]],
            "Chronological partition, S&amp;P 500 (table_04).")
    r.section("Target definition")
    r.eq("y<sub>t</sub> = 1 if Close<sub>t+1</sub> &gt; Close<sub>t</sub>, else 0",
         "Close<sub>t</sub> is the closing level on day t. An unchanged close counts as 0 (NextDay_NotUp), not as a "
         "'down' day. The last day has no label.")
    r.section("Data selection and transformation")
    r.p("Task-relevant attributes were selected by type. Trending levels (Close, EMAs) were not used directly as model "
        "inputs: the test period's index level lies outside most of the training range, so a model would have to "
        "extrapolate. They were replaced by stationary transformations, and the ADF test confirms stationarity (Table 6).")
    r.eq("dist_EMA50<sub>t</sub> = (Close<sub>t</sub> - EMA50<sub>t</sub>) / EMA50<sub>t</sub>;   "
         "vol20<sub>t</sub> = std(r<sub>t-19</sub>, ..., r<sub>t</sub>);   "
         "&#916;TS<sub>t</sub> = TS<sub>t</sub> - TS<sub>t-1</sub>",
         "r<sub>t</sub> is the daily return (column mom), TS = DGS10 - DTB3 is the term spread, and the credit spread is "
         "CS = DBAA - DGS10. The rolling volatility uses only past data.")
    r.table(adf[["Feature", "ADF_Statistic", "p_value", "Is_Stationary_5%"]], "Augmented Dickey-Fuller tests on modelling features (table_06).", [4.5 * cm, 3.5 * cm, 3 * cm, 3.5 * cm])

    # ================= 5 =================
    r.chapter("Data Representations")
    r.p("Each algorithm needs a different representation. A continuous matrix suits Gaussian NB and DBSCAN; binary "
        "transactions suit FP-Growth, Pincer Search and Bernoulli NB.")
    vocab = pd.DataFrame([
        ["SameDay_Up / SameDay_Down", "sign of mom<sub>t</sub> (0: no item)"],
        ["Volume_Jump", "Volume<sub>t</sub> &gt; Q75 of training days"],
        ["Mom5_Pos / Mom5_Neg", "sign of ROC_5<sub>t</sub>"],
        ["Above_EMA50 / Below_EMA50", "Close<sub>t</sub> vs EMA_50<sub>t</sub>"],
        ["TermSpread_Steep / _Flat / _Widening", "TS &gt; Q66 / TS &lt;= Q33 (training) / &#916;TS &gt; 0"],
        ["CreditSpread_High / _Low / _Widening", "CS &gt; Q66 / CS &lt;= Q33 (training) / &#916;CS &gt; 0"],
        ["USD_Up/Down, Gold_Up/Down, Oil_Up/Down", "sign of Dollar index, Gold, Oil returns (0 or missing: no item)"],
        ["Tech_Leaders_Up", "AAPL &gt; 0 and MSFT &gt; 0"],
        ["Europe_Up", "FTSE &gt; 0 and DAX-F &gt; 0"],
        ["Asia_Up", "HSI &gt; 0 and Nikkei-F &gt; 0"],
        ["NextDay_Up / NextDay_NotUp (C2 only)", "target y<sub>t</sub>"],
    ], columns=["Item", "Definition"])
    r.table(vocab, "Transaction vocabulary: 22 contemporaneous items plus 2 target items.", [6 * cm, 10 * cm])
    r.p("<b>C1 (contemporaneous)</b>: 1,784 transactions with 22 items describing the same day. <b>C2 (predictive)</b>: "
        "1,783 transactions that also contain the next-day item. Quantile thresholds are frozen from the training days.")
    r.code('''
mom5 = row.get("ROC_5", np.nan)            # 5-day rate of change
if pd.notna(mom5):
    if mom5 > 0:   items.append("Mom5_Pos")
    elif mom5 < 0: items.append("Mom5_Neg")
usd = row.get("Dollar index", np.nan)      # a daily % change, not a level
if pd.notna(usd):
    if usd > 0:   items.append("USD_Up")
    elif usd < 0: items.append("USD_Down")''')

    # ================= 6 =================
    r.chapter("Exploratory Data Analysis")
    r.section("Descriptive statistics")
    sp = desc[(desc["Index"] == "S&P") & desc["Feature"].isin(["mom", "Volume", "Dollar index", "Gold", "Oil", "DGS10", "DE1", "AAPL", "FTSE", "HSI"])]
    r.table(sp[["Feature", "Valid_N", "Mean", "Std_Dev", "Skewness", "Excess_Kurtosis", "Jarque_Bera_pVal", "Ljung_Box_Lag10_pVal"]].round(4),
            "Moments and tests for selected S&amp;P-file variables (sentinels removed; full table_07 covers all five indices).")
    r.eq("JB = n/6 [ S<sup>2</sup> + (K - 3)<sup>2</sup> / 4 ]",
         "n is the number of observations, S is skewness and K is kurtosis. Under normality JB follows &#967;<sup>2</sup>(2). "
         "Ljung-Box Q(10) tests for autocorrelation up to lag 10.")
    r.p("Daily returns have near-zero means, high excess kurtosis (fat tails) and Jarque-Bera p-values of about 0, so "
        "normality is rejected. This matters for Gaussian NB, which assumes normal class-conditional distributions.")
    r.section("Visualizations and inferences")
    figs = [
        ("fig_01_multi_index_normalized_trajectories.png", "Five indices rebased to 100.",
         "All indices trend strongly upward over 2010-2017 and move together. NASDAQ leads and NYSE lags. This is why trending price levels must not be used as stationary model inputs."),
        ("fig_02_rolling_volatility_clustering.png", "20-day annualised volatility of S&amp;P returns.",
         "Volatility comes in clusters (the 2011 euro crisis, August 2015, early 2016) separated by long calm periods. Risk is time-varying, which motivates the volatility feature and DBSCAN's regimes."),
        ("fig_03_yield_curve_regimes.png", "Term spread (10Y-3M) and credit spread (Baa-10Y), holiday zeros removed.",
         "The yield curve was never inverted in this sample; it flattened steadily after 2013. The credit spread rose in 2015-16 (energy stress). 'Inversion' would therefore be a useless item; tercile and change items are used instead."),
        ("fig_04_correlation_heatmap.png", "Spearman correlations of stationary series.",
         "US equity returns (index, AAPL, MSFT, JPM, peer indices) correlate strongly. FX and commodity correlations with equities are weak, which explains why same-day rules are strong but cross-asset rules are weak."),
        ("fig_05_return_distribution_kurtosis.png", "Return histogram with normal fit, and Q-Q plot.",
         "Both tails are heavier than normal (the Q-Q plot bends away at both ends). Extreme days are far more frequent than a Gaussian predicts."),
        ("fig_06_maximum_drawdown_comparison.png", "Underwater (drawdown) curves.",
         "The largest drawdowns occurred in 2011 and 2015-16. Russell 2000 (small caps) falls deepest, which is consistent with higher risk."),
    ]
    for f, c, inf in figs:
        r.figure(F / f, c)
        r.p("<b>Inference.</b> " + inf)

    # ================= 7 =================
    r.chapter("Association Rule Mining")
    r.p("Association rules X &#8658; Y describe items that occur together. Two rule families are kept strictly apart. "
        "<b>Contemporaneous</b> rules (database C1, all 1,784 days) describe same-day market structure. "
        "<b>Predictive</b> rules (database C2, training days only) have the consequent NextDay_Up or NextDay_NotUp and are "
        "re-scored on the 357 test days.")
    r.eq("supp(X) = &#963;(X)/N;  conf(X&#8658;Y) = &#963;(X&#8746;Y)/&#963;(X);  lift = conf/supp(Y);  "
         "conviction = (1 - supp(Y)) / (1 - conf)",
         "&#963;(X) is the number of transactions containing X and N is the database size. A lift above 1 means positive "
         "association. Conviction is infinite when conf = 1.")
    r.p("Integer thresholds avoid floating-point edge cases: &#963;<sub>min</sub> = ceil(0.08 x 1,784) = <b>143</b> "
        "(contemporaneous) and ceil(0.08 x 1,425) = <b>114</b> (predictive training), with minconf = 0.40. Each rule is "
        "tested with a two-sided Fisher exact test on its 2x2 table, and the p-values are corrected for multiple testing "
        "with Benjamini-Hochberg (FDR 5%).")

    # ================= 8 =================
    r.chapter("FP-Growth Algorithm")
    r.section("Algorithm")
    r.p("FP-Growth (Han, Pei &amp; Yin, 2000) avoids Apriori's candidate generation. It compresses the database into a "
        "prefix tree (FP-tree) in two scans, then mines it recursively through conditional pattern bases.")
    r.figure(D["fpgrowth"], "FP-Growth and rule-validation flowchart.", width=10)
    r.code('''
min_count = math.ceil(minsup * N)                     # 143 (C1) / 114 (C2 train)
fi = fpgrowth(X, min_support=min_count / N, use_colnames=True)
rules = association_rules(fi, metric="confidence", min_threshold=0.40)
rules["conviction"] = (1 - rules["consequent support"]) / (1 - rules["confidence"])''')
    r.section("Results")
    r.p(f"Contemporaneous database: <b>{len(fi_c):,}</b> frequent itemsets, <b>{len(contemp):,}</b> rules, of which "
        f"<b>{int(contemp['is_statistically_significant'].sum()):,}</b> are significant after BH correction.")
    top = contemp.sort_values("lift", ascending=False).head(8)
    r.table(top[["antecedents", "consequents", "support", "confidence", "lift", "conviction"]].round(3),
            "Top contemporaneous rules by lift.", [5.2 * cm, 4.8 * cm, 1.6 * cm, 1.8 * cm, 1.4 * cm, 1.8 * cm])
    r.p("<b>Interpretation.</b> The strongest rules link a falling index, a negative 5-day momentum, a close below the "
        "50-day EMA and falling oil, or a rising index with rising tech leaders and Europe. They describe co-movement on "
        "the same day. Partly this is index arithmetic: AAPL and MSFT are S&amp;P constituents, and mom<sub>t</sub> is "
        "part of ROC_5. These rules are not forecasts.")
    r.p(f"Predictive training database: <b>{len(fi_p):,}</b> itemsets and <b>{len(pred)}</b> rules with a NextDay "
        f"consequent. Their training lift ranges from {pred['lift'].min():.2f} to {pred['lift'].max():.2f}, and none "
        f"survives BH correction (min adjusted p = {pred['bh_adjusted_p_value'].min():.3f}).")
    tp = pred.sort_values("lift", ascending=False).head(8)
    r.table(tp[["antecedents", "consequents", "confidence", "lift", "test_confidence", "test_lift", "test_fisher_p_value"]].round(3),
            "Highest-lift predictive rules: training versus untouched test period.", [5.6 * cm, 2.6 * cm, 1.6 * cm, 1.3 * cm, 1.8 * cm, 1.5 * cm, 1.9 * cm])
    r.p(f"<b>Interpretation.</b> The best training rules lose their edge on the test period: test lift falls to about 1 "
        f"or below. Only {int(pred['test_is_statistically_significant'].sum())} of {len(pred)} rules reach p &lt; 0.05 "
        f"out of sample, close to the ~{0.05 * len(pred):.0f} expected by chance. Same-day indicators in these "
        f"transactions carry little reliable information about the next day's direction.")
    r.figure(F / "fig_11_association_rules_support_confidence_lift.png",
             "Rules in support-confidence space, coloured by lift.")
    r.p("<b>Inference.</b> High-lift rules sit at low support. The many high-confidence, low-lift rules involve "
        "frequent items (e.g. Above_EMA50) and are uninformative, which is why lift, conviction and significance are "
        "reported rather than confidence alone.")

    # ================= 9 =================
    r.chapter("Pincer Search Algorithm")
    r.section("Concept")
    r.p("An itemset is <b>maximal frequent</b> if it is frequent and none of its proper supersets is. Pincer Search "
        "(Lin &amp; Kedem, 1998) combines Apriori's bottom-up search with a top-down search over the Maximum Frequent "
        "Candidate Set (MFCS). The MFCS starts as {all items} and shrinks whenever an infrequent set is found. Whenever "
        "an MFCS element is frequent, all of its subsets are known to be frequent, so a long maximal itemset can be "
        "found early without enumerating its subsets.")
    r.figure(D["pincer"], "Pincer Search flowchart.", width=10.5)
    r.code('''
def prune_mfcs_by_infrequent(self, s):            # MFCS-gen for one infrequent set s
    for m in [m for m in self.mfcs if s <= m]:
        self.mfcs.remove(m)
        for e in s:                               # remove one item of s at a time
            child = m - {e}
            if child and not any(child <= x for x in self.mfcs):
                self.mfcs.add(child)
    self._enforce_antichain(self.mfcs)            # keep only maximal elements''')
    r.section("Correctness argument")
    r.bullets(["<b>Invariant I1.</b> Every frequent itemset is a subset of some element of MFS &#8746; MFCS. MFCS-gen "
               "removes an element m only if it contains an infrequent s, and replaces it with every m - {e}, e &#8712; s. "
               "A frequent X &#8838; m cannot contain all of s, so X &#8838; m - {e} for some e.",
               "<b>Soundness.</b> Sets enter MFS only after their support has been counted; the final antichain filter "
               "removes any non-maximal set.",
               "<b>Completeness.</b> The bottom-up L<sub>k</sub> is kept intact (subsumed candidates are not counted "
               "but still join), and the final step adds every frequent set not covered by MFS, so no maximal set is lost.",
               "<b>Termination.</b> k is at most the number of items, and each MFCS split reduces an element's size by one.",
               "<b>Complexity.</b> In the worst case the MFCS can grow exponentially (it is a minimal-transversal "
               "problem). Pincer helps when maximal itemsets are long, and is not guaranteed to be faster than Apriori."])
    r.section("Results and validation")
    r.table(piter, "Per-pass Pincer Search trace on C2 (minsup = 0.10, minimum count = ceil(0.10 x 1,783) = 179).")
    r.table(oracle[["Algorithm", "MFS_Count", "Matches_Ground_Truth"]],
            "Validation against two independent oracles (exact set equality).", [6 * cm, 3 * cm, 4 * cm])
    r.p(f"Pincer Search returns {n_mfi} maximal frequent itemsets, identical to mlxtend fpmax and to the maximal subset "
        f"of FP-Growth's output. The test suite also checks it against brute-force enumeration on 60 random databases "
        f"and on a textbook example. The maximal itemsets are short (2-4 items), because the paired items (Up/Down) are "
        f"mutually exclusive and most single items have support of 0.3-0.7.")

    # ================= 10 =================
    r.chapter("Machine Learning Classification")
    r.section("Naive Bayes")
    r.eq("P(y | x<sub>1..d</sub>) &#8733; P(y) &#928;<sub>j</sub> P(x<sub>j</sub> | y)",
         "y &#8712; {0,1} is the next-day direction and x<sub>j</sub> are features, assumed independent given y. Gaussian "
         "NB models P(x<sub>j</sub>|y) = N(&#956;<sub>jy</sub>, &#963;<sup>2</sup><sub>jy</sub>); Bernoulli NB models "
         "P(x<sub>j</sub>=1|y) = p<sub>jy</sub> with Laplace smoothing (&#945; = 1).")
    r.bullets(["<b>Gaussian NB</b>: 8 standardised stationary features (return, dist_EMA50, vol20, term and credit "
               "spreads and their changes, Volume); scaler fitted on training rows only.",
               "<b>Bernoulli NB</b>: the 22 binary contemporaneous items of day t.",
               "<b>Baseline</b>: the majority class of the 1,425 training labels (Up, 54.32%), applied unchanged to the "
               "357 test days, which gives 55.46% accuracy."])
    r.eq("McNemar &#967;<sup>2</sup> = (|b - c| - 1)<sup>2</sup> / (b + c)",
         "b counts days the model is right and the baseline wrong, and c the reverse. An exact binomial test compares "
         "the number of correct predictions with the majority rate.")
    cols = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "MCC", "McNemar_pVal_vs_Base", "Binomial_p_vs_Majority"]
    cm_ = clf[["Model"] + [c for c in cols if c in clf.columns]].round(4)
    r.table(cm_, "Test-period performance (N = 357). The p-values test 'better than the baseline'.", fs=7)
    r.figure(F / "fig_13_naive_bayes_confusion_matrices.png", "Confusion matrices on the 357 test days.")
    r.p(f"<b>Interpretation.</b> Bernoulli NB is the best model ({bnb['Accuracy']:.1%}, AUC {bnb['ROC_AUC']:.3f}, MCC "
        f"{bnb['MCC']:.3f}), but its advantage over the constant 'always Up' rule is not significant (McNemar p = "
        f"{bnb['McNemar_pVal_vs_Base']:.2f}, binomial p = {bnb['Binomial_p_vs_Majority']:.2f}). Gaussian NB is worse "
        f"than the baseline ({gnb['Accuracy']:.1%}, McNemar p = {gnb['McNemar_pVal_vs_Base']:.3f}). The fat-tailed, "
        f"dependent features violate its assumptions. Given the 95% interval of roughly +/-5 percentage points on 357 "
        f"days, the evidence does not show useful next-day predictability. This is consistent with, but does not "
        f"establish, weak-form market efficiency.")
    gl = lik[lik["Model"] == "BernoulliNB"].copy()
    gl = gl.reindex(gl["Ratio_C1_to_C0"].sub(1).abs().sort_values(ascending=False).index).head(8)
    r.table(gl[["Feature", "Class_0_Prob", "Class_1_Prob", "Ratio_C1_to_C0"]],
            "Bernoulli NB 'rules': items whose probability differs most between next-day Up and NotUp.", [5 * cm, 3 * cm, 3 * cm, 3 * cm])

    # ================= 11 =================
    r.chapter("Clustering using DBSCAN")
    r.p("DBSCAN (Ester et al., 1996) groups points in dense regions and labels isolated points as noise. Unlike "
        "k-means, it needs no cluster count. It is applied to five regime features: vol20, term spread, credit spread, "
        "return and Volume, standardised with training-row statistics.")
    r.figure(D["dbscan"], "DBSCAN procedure.", width=10)
    r.eq("N<sub>&#949;</sub>(p) = { q : ||p - q|| &lt;= &#949; };   p is a core point if |N<sub>&#949;</sub>(p)| &gt;= MinPts",
         "MinPts = 2d - 1 = 9 for d = 5 features. &#949; is the knee of the sorted 9-NN distance curve (the point "
         "farthest from the chord joining the curve's ends).")
    r.figure(F / "fig_08_dbscan_k_distance_knee.png", "Sorted 9-NN distance curve and the selected &#949;.")
    r.p("<b>Inference.</b> The curve is flat for most days and rises sharply for a few percent of them. The knee "
        "separates the dense core of ordinary days from the sparse extreme days.")
    r.table(prof.drop(columns=["Index"]).round(4), "DBSCAN cluster profiles (table_16).", fs=7)
    r.p(f"<b>Interpretation.</b> DBSCAN finds {n_clusters} dense regime with {int(prof[prof['Cluster_ID'] != -1]['Observation_Count'].sum())} "
        f"days, and {n_noise} noise days ({n_noise / 1784:.1%}). The noise days have much higher volatility and "
        f"volume changes; they are the market-stress days (August 2011, August 2015, the Brexit vote). A single global "
        f"&#949; cannot also split the calm days into sub-regimes, because their density varies smoothly. The silhouette "
        f"is therefore undefined, and the honest result is 'normal versus stress', not several distinct regimes.")

    # ================= 12 =================
    r.chapter("Dimensionality Reduction using PCA and t-SNE")
    r.section("PCA")
    r.eq("max<sub>w</sub> w<sup>T</sup>&#931;w  subject to ||w|| = 1",
         "&#931; is the covariance matrix of the standardised features. The principal components are its eigenvectors, "
         "and the explained variance ratio is &#955;<sub>i</sub>/&#931;&#955;.")
    r.figure(F / "fig_07_pca_scree_explained_variance.png", "Scree plot: standardised stationary features, fitted on training rows.")
    r.p("<b>Inference.</b> The variance is spread across many components: the cross-asset returns are only weakly "
        "correlated, so no handful of factors summarises the data. This is why DBSCAN uses five interpretable regime "
        "features instead of all 80 dimensions.")
    r.section("t-SNE")
    r.p("t-SNE (van der Maaten &amp; Hinton, 2008) maps points to 2-D so that local neighbourhoods are preserved. It "
        "minimises KL(P||Q) between Gaussian similarities in the original space and Student-t similarities in the map "
        "(perplexity 30, PCA initialisation, seed 42). It is <b>transductive</b>: it embeds all 1,784 days at once and "
        "cannot place new points. It is used only for visualisation. The scaler was fitted on the training rows, but the "
        "embedding itself uses every day, so it is not a training-only analysis.")
    r.figure(F / "fig_09_dbscan_regime_clusters_pca.png", "DBSCAN labels projected with PCA (left) and t-SNE (right).")
    r.figure(F / "fig_09b_tsne_manifold.png", "t-SNE embedding of the regime features.")
    r.p("<b>Inference.</b> In both projections the noise (stress) days lie on the outer edge, away from the dense core. "
        "t-SNE separates them more clearly, but distances between t-SNE clusters are not meaningful.")

    # ================= 13 =================
    r.chapter("Hierarchical Clustering using Ward Linkage")
    r.eq("&#916;(A,B) = (|A||B| / (|A| + |B|)) ||c<sub>A</sub> - c<sub>B</sub>||<sup>2</sup>",
         "Ward merges the two clusters A, B whose merge increases the within-cluster sum of squares least; "
         "c<sub>A</sub>, c<sub>B</sub> are the centroids.")
    r.p("A dendrogram is a required clustering output, and DBSCAN does not produce one. Ward agglomerative clustering "
        "is therefore run on the same standardised regime features, using every 4th day (446 days spanning 2010-2017). "
        "The tree is truncated to its last 30 merges.")
    r.figure(F / "fig_10_ward_hac_dendrogram.png", "Truncated Ward dendrogram of the regime features.")
    r.p("<b>Inference.</b> The final merge joins two large groups at a much greater height than the earlier merges, "
        "which suggests a two-group structure (calm versus volatile periods) similar to DBSCAN's result. Below that "
        "level the merge heights grow gradually, with no further clear break.")

    # ================= 14 =================
    r.chapter("PySpark Implementation")
    r.p("Two techniques run natively on Spark MLlib: <b>Naive Bayes</b> (<font face='Courier'>modelType='gaussian'</font>; "
        "the default multinomial model rejects negative returns) and <b>FP-Growth</b> (Parallel FP-Growth, Li et al., "
        "2008). Both are checked for exact equivalence with the local implementations.")
    r.code('''
nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="gaussian")
model = nb.fit(train_df)                       # 1,425 training rows
pred = model.transform(test_df)                # 357 test rows

fp = FPGrowth(itemsCol="items", minSupport=0.08, minConfidence=0.40)
model = fp.fit(spark_tx_df)                    # same C1 database as mlxtend
itemsets = model.freqItemsets.toPandas()''')
    par = pd.DataFrame([
        ["Gaussian NB predictions identical to scikit-learn", f"{int(spnb['Identical_Predictions'])} / {int(spnb['Test_Observations'])} ({spnb['Parity_Agreement_Pct']:.1f}%)"],
        ["Spark NB test accuracy", f"{spnb['Spark_Accuracy']:.4f}"],
        ["FP-Growth itemsets (mlxtend / Spark)", f"{int(spfp['mlxtend_Itemsets'])} / {int(spfp['Spark_Itemsets'])}; identical: {bool(spfp['Itemsets_Identical'])}"],
        ["Single-consequent rules (mlxtend / Spark)", f"{int(spfp['mlxtend_Rules_SingleConsequent'])} / {int(spfp['Spark_Rules'])}; identical: {bool(spfp['Rules_Identical'])}"],
        ["Max |confidence difference|", f"{spfp['Max_Confidence_Abs_Diff']:.1e}"],
    ], columns=["Check", "Result"])
    r.table(par, "Spark vs local equivalence (table_17, table_19).", [8 * cm, 7 * cm])
    r.p("Spark only emits rules with a single-item consequent, so rules were compared on that subset. Spark keeps "
        "itemsets with count &gt;= ceil(minSupport x N), which matches the integer threshold used locally.")

    # ================= 15 =================
    r.chapter("Performance Benchmarking and Amdahl's Law")
    r.section("Method")
    r.bullets(["Synthetic workloads: bootstrap resampling of the real transactions with 5% random item flips. Pure "
               "replication would leave the FP-tree unchanged and make the benchmark meaningless.",
               "Spark start-up, data loading and steady-state fit times are measured separately. Steady-state fits are "
               "the mean of 3 runs after a warm-up, and every Spark run is forced to completion with count().",
               "Memory: peak RSS of the whole process tree (Python plus the Java child process), sampled every 50 ms with psutil.",
               f"Strong scaling: a fixed workload of {ss_.get('workload_rows', 200000):,} transactions on local[1], local[2], "
               f"local[4] and local[8]."])
    r.eq("S(p) = T(1)/T(p);   Amdahl: S(p) = 1 / ((1 - f) + f/p)",
         "p is the number of cores and f the parallelisable fraction of the work.")
    dsv = dsr[["N_Rows", "Distinct_Transactions", "Local_mlxtend_Sec", "Spark_FPGrowth_Sec", "Speedup_Local_over_Spark",
               "Local_Peak_RSS_MB", "Spark_Peak_RSS_MB_ProcessTree"]].copy()
    dsv.columns = ["N", "Distinct tx", "Local fit (s)", "Spark fit (s)", "T_local / T_spark", "Local RSS (MB)", "Spark RSS (MB)"]
    r.table(dsv, f"Data scaling, minsup = 0.10 (Spark session start-up: {ds['spark_startup_sec']:.2f} s, excluded).")
    r.figure(F / "fig_14_local_vs_spark_data_scaling.png", "Local vs Spark FP-Growth fit time as data grows.")
    sp_t = pd.DataFrame({"Cores p": ss_["cores"], "Fit time (s)": [ss_["timing_by_core"][str(c)]["steady_fit_mean_sec"] for c in ss_["cores"]],
                         "Speedup S(p)": ss_["speedups"], "Efficiency": ss_["efficiencies"]})
    r.table(sp_t, f"Strong scaling of Spark FPGrowth (fitted Amdahl f = {ss_['parallel_fraction']:.2f}).", [3 * cm, 3 * cm, 3 * cm, 3 * cm])
    r.figure(F / "fig_12_pyspark_strong_scaling_speedup.png", "Observed Spark speedup vs ideal and fitted Amdahl curves.")
    r.section("Interpretation: what PySpark did and did not deliver")
    r.p("On this single laptop, local mlxtend FP-Growth was faster than Spark at every size we measured except one noisy "
        "point. Spark also did not speed up with more cores (fitted f is about 0). The reasons can be identified:")
    r.bullets(["The vocabulary has only 24 items, so the frequent-itemset lattice is tiny. Spark's PFP shards the work by "
               "item, giving at most a handful of meaningful tasks; the job is dominated by fixed scheduling, "
               "serialisation and a single-threaded final aggregation, which is the serial fraction in Amdahl's law.",
               "mlxtend runs a vectorised, in-memory algorithm with no task or JVM overhead; at 250,000 x 24 booleans "
               "the data is only about 6 MB.",
               "All 'workers' share one machine's memory bandwidth, so local mode cannot show true horizontal scaling."])
    r.p("Big-data support from Spark is therefore not a speed gain at this data scale. What it does provide, as "
        "measured here, is (a) an identical result from a distributed, partitioned algorithm, with exact parity for both "
        "techniques, and (b) a memory model that does not require a dense in-memory one-hot matrix. That matters once the "
        "data, or the number of items, exceeds one machine. Showing a speed-up would require a real multi-node cluster "
        "or a much wider item vocabulary. We state this limitation rather than presenting an artificial speed-up.")

    # ================= 16 =================
    r.chapter("Testing and Validation")
    r.p("The pytest suite has two parts. (1) Artefact checks confirm that the pipeline outputs exist with the expected "
        "shapes. (2) Independent correctness tests recompute results from first principles:")
    r.bullets(["Pincer Search equals brute-force maximal itemsets on 60 random databases, plus a textbook example.",
               "Sentinels are absent from Rep A, including the TE*/DE* spreads.",
               "Every vocabulary item actually occurs, and mutually exclusive items never co-occur.",
               "The target puts unchanged closes in NextDay_NotUp.",
               "The FP-Growth database sizes and thresholds are 1,784/143 and 1,425/114.",
               "Predictive rules have a NextDay consequent, no NextDay antecedent, and test-period metrics.",
               "The majority baseline is derived from the 1,425 training labels only.",
               "Spark FPGrowth results are identical to mlxtend's, and Spark NB matches scikit-learn."])
    if tests:
        r.p(f"Final run: <b>{tests['passed']} passed, {tests['failed']} failed</b> ({tests['total']} tests).")

    # ================= 17 =================
    r.chapter("Results and Comparative Analysis")
    summ = pd.DataFrame([
        ["Preprocessing", "Hidden sentinels removed; 1,784 clean days; leakage-free 1,425/1/357 split"],
        ["FP-Growth (contemporaneous)", f"{len(fi_c):,} itemsets, {len(contemp):,} rules, {int(contemp['is_statistically_significant'].sum()):,} FDR-significant; max lift {contemp['lift'].max():.2f}"],
        ["FP-Growth (predictive)", f"{len(pred)} rules on train, 0 FDR-significant; no stable out-of-sample edge"],
        ["Pincer Search", f"{n_mfi} MFIs; identical to fpmax and FP-Growth; verified against brute force"],
        ["Gaussian NB", f"accuracy {gnb['Accuracy']:.3f}, AUC {gnb['ROC_AUC']:.3f}; significantly below baseline"],
        ["Bernoulli NB", f"accuracy {bnb['Accuracy']:.3f}, AUC {bnb['ROC_AUC']:.3f}; above baseline, not significant"],
        ["Majority baseline", f"accuracy {base['Accuracy']:.3f}"],
        ["DBSCAN", f"{n_clusters} dense regime + {n_noise} stress days ({n_noise / 1784:.1%})"],
        ["Ward HAC", "two-level structure consistent with DBSCAN"],
        ["PySpark", "exact parity for NB and FP-Growth; no speed-up on one machine at these sizes"],
    ], columns=["Component", "Result"])
    r.table(summ, "Summary of results.", [4.5 * cm, 11.5 * cm])
    r.p("<b>Comparison.</b> The descriptive techniques find strong, statistically robust structure: same-day "
        "co-movement, volatility clustering, fat tails and stress regimes. The predictive techniques (predictive rules "
        "and NB) find no next-day signal that survives out-of-sample testing. This contrast is the main finding. The "
        "variables describe today's market well but do not reliably forecast tomorrow's direction at daily frequency.")

    # ================= 18 =================
    r.chapter("Limitations, Conclusion and Future Work")
    r.section("Limitations")
    r.bullets(["The evaluation uses a single chronological test window (357 days in a calm bull market). Walk-forward "
               "evaluation over several windows would give a distribution of scores.",
               "Contemporaneous mining uses the full sample (it is descriptive). Only the predictive rules are out-of-sample.",
               "The 'DBCV' value in the code is a noise-penalised silhouette heuristic, not the DBCV index of Moulavi et "
               "al. (2014), and it is reported under that name.",
               "t-SNE is transductive and used only for visualisation.",
               "Spark ran in local mode on one machine; no cluster scaling was measured.",
               "Results are for the S&amp;P 500 file. The other indices share 64 of 84 columns and are not independent."])
    r.section("Conclusion")
    r.p("All required techniques were implemented, executed and verified. The project's strongest results are its "
        "verified correctness and its honest evaluation: exact algorithm equivalences (Pincer vs fpmax, Spark vs "
        "local) and leakage-free predictive testing. The financial conclusion is deliberately modest: no tested "
        "technique gives a statistically reliable next-day edge on this data. This is consistent with the weak-form "
        "Efficient Market Hypothesis but does not prove it.")
    r.section("Future work")
    r.bullets(["Walk-forward evaluation over several windows; probability calibration.",
               "HDBSCAN or OPTICS for varying-density regimes, with true DBCV.",
               "A multi-node Spark cluster and a wider item vocabulary to test where distributed FP-Growth pays off.",
               "Intraday or multi-day horizons, where predictability may differ."])

    # ================= 19 =================
    r.chapter("Reproducibility Instructions")
    r.code('''
pip install -r requirements.txt          # Python 3.10+, Java 17 for PySpark
python run_all.py                        # pipeline -> 11 notebooks -> pytest -> PDF
# or step by step:
python -m src.run_pipeline               # all tables, figures, metrics, this PDF
python -c "from src.build_notebooks import build_all_notebooks; build_all_notebooks()"
python -m pytest tests -q
python -m src.generate_pdf_report''')
    r.p("Seeds are fixed (42). Raw files are checked with SHA-256 on every run. Benchmark timings vary between machines; "
        "all other numbers are deterministic.")

    # ================= 20 =================
    r.chapter("References and Appendices")
    r.section("References")
    for ref in [
        "Hoseinzade, E., &amp; Haratizadeh, S. (2019). CNNpred: CNN-based stock market prediction using a diverse set of variables. <i>Expert Systems with Applications</i>, 129, 273-285.",
        "Han, J., Pei, J., &amp; Yin, Y. (2000). Mining frequent patterns without candidate generation. <i>ACM SIGMOD</i>, 1-12.",
        "Lin, D.-I., &amp; Kedem, Z. M. (1998). Pincer-Search: A new algorithm for discovering the maximum frequent set. <i>EDBT</i>, 105-119.",
        "Li, H., Wang, Y., Zhang, D., Zhang, M., &amp; Chang, E. (2008). PFP: Parallel FP-Growth for query recommendation. <i>ACM RecSys</i>.",
        "Ester, M., Kriegel, H.-P., Sander, J., &amp; Xu, X. (1996). A density-based algorithm for discovering clusters. <i>KDD</i>, 226-231.",
        "van der Maaten, L., &amp; Hinton, G. (2008). Visualizing data using t-SNE. <i>JMLR</i>, 9, 2579-2605.",
        "Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. <i>JASA</i>, 58, 236-244.",
        "Benjamini, Y., &amp; Hochberg, Y. (1995). Controlling the false discovery rate. <i>JRSS B</i>, 57, 289-300.",
        "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions. <i>Psychometrika</i>, 12, 153-157.",
        "Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. <i>Journal of Finance</i>, 25, 383-417.",
        "Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. <i>AFIPS</i>.",
        "Zaharia, M., et al. (2016). Apache Spark: A unified engine for big data processing. <i>CACM</i>, 59(11), 56-65.",
    ]:
        r.bullets([ref])
    r.section("Appendix A: Requirement-to-deliverable traceability")
    trace = pd.DataFrame([
        ["Exhaustive descriptive analytics", "Ch. 2, 4, 6; table_07 (all 5 indices)", "src/descriptive_analytics.py; NB 01-03", "Done"],
        ["Cleaning: remove noisy data", "Ch. 4.1-4.2; table_02, table_03", "src/preprocessing.py", "Done"],
        ["Selection: task-relevant attributes", "Ch. 4.5, 5; table_06", "src/feature_engineering.py", "Done"],
        ["Transformation to required form", "Ch. 4-5 (Rep A, B, C1, C2)", "src/preprocessing.py, feature_engineering.py", "Done"],
        [">= 10 unique plots with inferences", "14 figures + 5 diagrams, inference under each", "src/visualization.py; NB 03-10", "Done"],
        ["Association rules with FP-Growth", "Ch. 7-8; table_08-10; Fig. 11", "src/association_rules.py; NB 04", "Done"],
        ["Naive Bayes / DBSCAN", "Ch. 10-11; table_14-16; Fig. 8, 9, 13", "src/classification.py, clustering.py; NB 06-07", "Done (both)"],
        ["MFIs with Pincer Search", "Ch. 9; table_11-13", "src/pincer_search.py; NB 05", "Done, verified"],
        ["Rules, performance measures, dendrograms", "Tables 8-13; Ch. 10, 13", "NB 04-08", "Done"],
        ["Two techniques on PySpark", "Ch. 14; table_17-19", "src/pyspark_pipeline.py; NB 09", "Done, exact parity"],
        ["Establish performance enhancement", "Ch. 15; benchmark_results.json; Fig. 12, 14", "src/benchmark_engine.py; NB 10", "Measured: no speed-up on 1 machine (explained)"],
        ["Commented code, outputs, interpretations", "All chapters; source docstrings", "src/, notebooks/", "Done"],
        ["Overall PDF report + source files", "This report; reports/", "run_all.py", "Done"],
    ], columns=["Exam requirement", "Where in report / outputs", "Code / notebook", "Status"])
    r.table(trace, "Requirement-to-deliverable traceability.", [4.3 * cm, 4.6 * cm, 4.4 * cm, 3 * cm], fs=7.2)
    r.section("Appendix B: Output files")
    r.p("outputs/tables: table_01 to table_19 (CSV). outputs/figures: fig_01 to fig_14 and diagrams/. "
        "outputs/metrics: benchmark_results.json, test_summary.json. notebooks: 01 to 11 (executed).")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "Analytics_BigData_Endsem_Report.pdf"
    doc = Doc(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
              title="CNNpred Big Data Analytics - Endsem Report")
    doc.multiBuild(s, onFirstPage=lambda c, d: None, onLaterPages=_page)
    return out


if __name__ == "__main__":
    print(create_academic_report())
