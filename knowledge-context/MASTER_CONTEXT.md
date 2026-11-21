# 🧠 MASTER_CONTEXT.md: AUTONOMOUS AI CODING AGENT OPERATIONAL DIRECTIVE
## Complete Big Data Curriculum Synthesis & Rapid Hackathon Execution Context

> **Target Audience:** Autonomous AI Coding Agents, Machine Learning Engineers, and LLM Planners.  
> **Core Objective:** Provide complete, dense, and instantly actionable technical context across all 12 lab works, Task 2, and the Semester Big Data Project so an AI agent can execute, debug, combine, and scale any data science solution without inspecting 50+ separate repository files.

---

## 1. WORKSPACE ARCHITECTURE & MASTER REPOSITORY MANIFEST

**Root Workspace:** `c:/Users/Sreekar_Tirumala/Documents/College/Big Data`  
**Hackathon Working Directory:** `c:/Users/Sreekar_Tirumala/Documents/College/Big Data/lab_hackathon`

```
c:/Users/Sreekar_Tirumala/Documents/College/Big Data/
├── BIG_DATA_LAB_1.ods, BIG_DATA_LAB_2.ods        # Labs 1 & 2: Spreadsheets, descriptive stats & pivot tables
├── Supply_Chain_Data_Analysis.xlsx               # Baseline logistics dataset (100 rows, 24 cols)
├── task2/                                        # Task 2 (Asgn 2): Tableau workbooks (.twb) & visual reports
├── LAB3/                                         # Lab 3: Multi-domain EDA (4 datasets, 55-cell notebook)
│   ├── Data_analysis.ipynb                       # Titanic, Olympics, REF 2014, Supply Chain visual profiling
│   └── supply_chain_data.csv                     # 100-row logistics benchmark
├── lab4/                                         # Lab 4: Preprocessing & Binning
│   ├── lab4_solution.py                          # 321 lines: Min-Max, Z-Score, Decimal Scaling, Equal-Freq Bins
│   └── task2/lab4_task2_complete_analysis.ipynb  # IMPS & NFS banking ecosystem statistics analysis
├── LAB5/                                         # Lab 5: Relational SQL Pipeline & Feature Engineering
│   ├── northwind.db                              # SQLite database (7 relational tables, 2,155 order lines)
│   └── Northwind_Preprocessing_Final.ipynb       # SQL joins, Regex parsing, Tukey IQR capping, Log1p skew
├── lab6/                                         # Lab 6: Baseline FIM (Apriori vs FP-Growth)
│   ├── fim_test.py                               # MLxtend benchmark on FoodMart retail data
│   └── foodmartFIM.txt                           # 4,141 retail transactions, 1,559 items
├── Assignment_7_FIM_All_Files/                   # Lab 7: Advanced FIM Architectures
│   └── assignment7_fim.py                        # 244 lines: Dynamic Itemset Counting (DIC), Hash Apriori (DHP), ECLAT
├── lab8/                                         # Lab 8: Advanced FIM Partitioning & Search Spaces
│   ├── fim_algorithms.py                         # 412 lines: Standard Apriori, SON Partitioning, AClose, Pincer Search
│   └── FIM_Algorithms_Test.ipynb                 # Verification suite comparing all 4 algorithms
├── lab 9/                                        # Lab 9: Supervised Decision Trees (CART)
│   ├── src/train_decision_tree.py                # 357 lines: ColumnTransformer, 5-fold CV, CCP pruning, path tracing
│   └── online_shoppers_decision_tree.py          # UCI Online Shoppers (12,330 rows, Revenue target ~15.5%)
├── lab10/                                        # Lab 10: Probabilistic & Synthetic Unsupervised Learning
│   ├── 01_naive_bayes_online_shoppers.ipynb      # GaussianNB, BernoulliNB, ComplementNB (imbalance tuning)
│   ├── 02_naive_bayes_supply_chain.ipynb         # Multiclass Naive Bayes on carrier logistics & quality inspection
│   └── 03_clustering_synthetic_dataset.ipynb     # K-Means, HAC (Ward/Single/Complete/Avg), Cophenetic correlation
├── lab11/                                        # Lab 11: Advanced Customer Segmentation & Metric Spaces
│   └── online_shoppers_clustering.ipynb          # Automated Knee-Detection DBSCAN, 13 Distance Metrics, GMM, OPTICS
├── lab12/                                        # Lab 12: Distributed Big Data & Metaheuristics
│   ├── task1_genetic_algorithm.py                # 305 lines: Continuous SGA (Sphere, Rastrigin, Rosenbrock, Poly)
│   ├── task2_tsp_ga.py                           # 321 lines: Permutation TSP GA (OX1, PMX, 2-Opt Inversion, Swap)
│   ├── task3_fpgrowth_dbscan.py                  # 463 lines: Serial vs Parallel FP-Growth & Spatial DBSCAN with DSU
│   ├── task4_classification_clustering.py       # 245 lines: 6-Classifier evaluation clustered in 7D metric space
│   ├── pyspark_fpgrowth_dbscan_cluster.py        # Production PySpark MLlib FPGrowth & DBSCAN script
│   └── Big_Data_Lab12.ipynb                      # Master consolidated interactive notebook
├── BIG_DATA_PROJECT/urban-service-intelligence/  # Production PySpark Batch & Structured Streaming Pipeline
└── lab_hackathon/                                # Master Technical Knowledge Base & Tactical Playbooks
    ├── LAB_INDEX.md                              # Chronological catalog linking all files and schemas
    ├── ALGORITHMS_GUIDE.md                       # Complete algorithmic catalog, formulas & complexities
    ├── CODE_SNIPPETS_AND_REUSABLE_MODULES.md     # Standalone production modules with I/O examples
    ├── ENV_AND_LIBRARIES.md                      # Package pins, pip commands, Windows & PySpark setup
    ├── IMPLEMENTATION_AUDIT.md                   # Critical bug audit, code fixes & cross-lab combinations
    ├── HACKATHON_PLAYBOOK.md                     # Rapid triage, streaming watchers & problem adaptors
    └── MASTER_CONTEXT.md                         # This executive autonomous agent operational directive
```

---

## 2. COMPACT ALGORITHMIC CHEAT SHEET

| Algorithm | Paradigm | Complexity | Best Suited For | File Pointer |
|---|---|---|---|---|
| **Vertical ECLAT** | Pattern Mining | $O(N \log N)$ set ops | Fast basket mining, TID intersections, rules | [`assignment7_fim.py#L103`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L103) |
| **FP-Growth** | Pattern Mining | $O(N)$ (2 scans) | Dense retail transactions, avoids candidate generation | [`task3_fpgrowth_dbscan.py#L25`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task3_fpgrowth_dbscan.py#L25) |
| **SON Partitioning** | Distributed FIM | Exactly 2 passes | Horizontal distributed candidate generation | [`fim_algorithms.py#L64`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L64) |
| **AClose** | Lossless FIM | Polynomial in $|G|$ | Closed itemsets, minimal non-redundant generators | [`fim_algorithms.py#L137`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L137) |
| **Pincer Search** | Bidirectional FIM | Pruned top-down | Maximal frequent itemsets (MFIs) via MFCS | [`fim_algorithms.py#L234`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L234) |
| **CART (Gini/Entropy)** | Supervised Tree | $O(D \cdot N \log N)$ | Interpretable decisions, cost-complexity pruning (CCP) | [`train_decision_tree.py#L50`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/src/train_decision_tree.py#L50) |
| **Complement Naive Bayes** | Probabilistic ML | $O(N \cdot D)$ (ultra-fast) | Severe class imbalance (< 15% positive), text/clickstream | [`01_naive_bayes_online_shoppers.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/01_naive_bayes_online_shoppers.ipynb) |
| **Automated Knee DBSCAN** | Density Clustering | $O(N \log N)$ with KD-tree | Arbitrary shapes, automatic $\varepsilon$ discovery, noise isolation | [`online_shoppers_clustering.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb) |
| **Spatial DBSCAN + DSU** | Distributed Density | $O(N / P)$ + boundary merge | Distributed spatial clustering, merges halos via Disjoint Set Union | [`task3_fpgrowth_dbscan.py#L225`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task3_fpgrowth_dbscan.py#L225) |
| **HAC (Ward Linkage)** | Hierarchical | $O(N^2 \log N)$ | Multi-scale dendrograms, cophenetic correlation | [`03_clustering_synthetic_dataset.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/03_clustering_synthetic_dataset.ipynb) |
| **Simple GA (SGA)** | Metaheuristic | $O(G \cdot P \cdot D)$ | Continuous multi-modal optimization, feature selection | [`task1_genetic_algorithm.py#L30`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task1_genetic_algorithm.py#L30) |
| **Permutation TSP GA** | Combinatorial | $O(G \cdot P \cdot N)$ | Vehicle routing, dispatch sequencing (OX1 + 2-Opt) | [`task2_tsp_ga.py#L35`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task2_tsp_ga.py#L35) |
| **PySpark MLlib FPGrowth** | Distributed Engine | Linear horizontal scaling | Multi-gigabyte transaction datasets on cluster cores | [`pyspark_fpgrowth_dbscan_cluster.py#L14`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/pyspark_fpgrowth_dbscan_cluster.py#L14) |

---

## 3. INSTANT PLUG-AND-PLAY PRODUCTION CODE SNIPPETS

### 3.1 Robust Preprocessing Pipeline (IQR Capping + Log1p + ColumnTransformer)
```python
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_production_preprocessor(df: pd.DataFrame, num_cols: list, cat_cols: list, skew_thresh: float = 1.0):
    df_clean = df.copy()
    # 1. Tukey IQR Outlier Capping
    for col in num_cols:
        q25, q75 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        iqr = q75 - q25
        df_clean[col] = df_clean[col].clip(lower=q25 - 1.5 * iqr, upper=q75 + 1.5 * iqr)
    # 2. Skewness Rectification (np.log1p)
    for col in num_cols:
        if df_clean[col].min() >= 0 and abs(skew(df_clean[col].dropna())) > skew_thresh:
            df_clean[col] = np.log1p(df_clean[col])
    # 3. Leakage-Free ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )
    return df_clean, preprocessor
```

### 3.2 High-Speed Vertical ECLAT & Association Rules
```python
import math
from itertools import combinations
from collections import defaultdict

def fast_eclat_and_rules(transactions: list, min_support: float = 0.05, min_confidence: float = 0.6):
    n = len(transactions)
    min_count = math.ceil(min_support * n)
    tidsets = defaultdict(set)
    for tid, t in enumerate(transactions):
        for item in t:
            tidsets[item].add(tid)
    items = [(frozenset([x]), tids) for x, tids in tidsets.items() if len(tids) >= min_count]
    items.sort(key=lambda p: len(p[1]))
    frequent = {item: len(tids) / n for item, tids in items}

    def recurse(prefix, suffix):
        for i, (item, tids) in enumerate(suffix):
            newset = prefix | item
            frequent[newset] = len(tids) / n
            new_suffix = [(it2, inter) for it2, tids2 in suffix[i+1:] if len(inter := tids & tids2) >= min_count]
            if new_suffix:
                recurse(newset, new_suffix)

    for i, (item, tids) in enumerate(items):
        new_suffix = [(it2, inter) for it2, tids2 in items[i+1:] if len(inter := tids & tids2) >= min_count]
        if new_suffix:
            recurse(item, new_suffix)

    rules = []
    for itemset, sup in frequent.items():
        if len(itemset) >= 2:
            for r in range(1, len(itemset)):
                for ant in combinations(itemset, r):
                    A = frozenset(ant)
                    if A in frequent and (conf := sup / frequent[A]) >= min_confidence:
                        rules.append({'A': set(A), 'B': set(itemset - A), 'sup': sup, 'conf': conf})
    return frequent, rules
```

### 3.3 Automated Knee-Detection DBSCAN
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

def auto_dbscan(X: np.ndarray, min_samples: int = None):
    k = min_samples if min_samples else 2 * X.shape[1]
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, -1])
    knee = KneeLocator(np.arange(len(k_dist)), k_dist, curve='convex', direction='increasing').knee
    eps = k_dist[knee] if knee is not None else float(np.median(k_dist))
    db = DBSCAN(eps=eps, min_samples=k).fit(X)
    return db, eps, db.labels_
```

---

## 4. CRITICAL PITFALLS & IMPLEMENTATION DEFECTS TO AVOID

1. **Avoid Custom DIC in `assignment7_fim.py`:** Line 193 rescans transactions `trans[0:end]` when new candidates form, converting DIC into quadratic passes. Use `Module 4 Vertical ECLAT` or `mlxtend.fpgrowth` instead.
2. **Avoid Set Bloat in ECLAT:** On large datasets ($> 100\text{k}$ rows), raw `set` TID-lists consume excessive memory; convert TIDs to sorted NumPy arrays or integer bit-masks.
3. **Avoid Unhandled Categories in Test Folds:** Never call `pd.get_dummies` independently on train and test sets. Always use `OneHotEncoder(handle_unknown='ignore')`.
4. **Avoid Data Leakage:** Never fit scalers or compute IQR fences on the complete dataset before train/test split. Fit strictly on `X_train`.
5. **Guard Against Zero Variance in Normalization:** Always check `denom = max - min` and `std != 0` before division in custom normalization routines.
6. **Watch the High-D Euclidean Singularity Effect:** In high-dimensional one-hot encoded spaces ($D > 20$), Euclidean distance concentrates tightly around the mean, breaking DBSCAN hyperspheres. Apply PCA/UMAP or use Manhattan/Cosine distances.
7. **PySpark on Windows:** If `winutils.exe` is missing, ensure you avoid local checkpointing to unmapped paths or set `HADOOP_HOME`.

---

## 5. HACKATHON TACTICAL ADAPTATIONS MATRIX

| Problem Category | Key Indicators | Recommended Pipeline Architecture | Evaluation Target |
|---|---|---|---|
| **E-Commerce Basket Recommendations** | Retail orders, POS logs, co-purchasing | Lab 5 SQL Joins $\rightarrow$ Module 4 Vertical ECLAT $\rightarrow$ AClose Filtering | Support $\ge 0.02$, Conf $\ge 0.6$, Lift |
| **Fraud & Anomaly Detection** | Rare events ($< 2\%$), skewed financial amounts | Module 3 IQR Capping + Log1p $\rightarrow$ Lab 10 `ComplementNB` / Lab 9 Pruned CART | PR-AUC, Macro-F1, Recall |
| **High-Velocity IoT Telemetry** | Time-series sensor streams, multi-channel metrics | Module 6 Automated DBSCAN (Noise isolation) $\rightarrow$ Rolling Z-Score Fences | False Alarm Rate, Latency $< 50\text{ms}$ |
| **Customer Segmentation** | Clickstream behavioral counts, session durations | Module 3 Log1p $\rightarrow$ PCA (3D) $\rightarrow$ Module 6 DBSCAN / Lab 10 K-Means | Silhouette Score $> 0.6$, Davies-Bouldin |
| **Route & Dispatch Optimization** | Waypoints, delivery coordinates, latency matrices | Lab 11 $L_1$ Distance $\rightarrow$ Module 9 Permutation GA (OX1 + 2-Opt Inversion) | Total Tour Distance Minimization |

---

## 6. ONE-COMMAND REPRODUCIBILITY ENVIRONMENT

```powershell
pip install numpy==2.2.6 scipy==1.15.3 pandas==2.3.3 scikit-learn==1.7.2 matplotlib==3.11.2 seaborn==0.13.2 mlxtend==0.23.1 kneed==0.8.5 pyspark==3.5.3 pyarrow==25.0.1 openpyxl==3.1.5 python-docx==1.2.0 joblib==1.6.0
```

*For comprehensive mathematical derivations, consult [`ALGORITHMS_GUIDE.md`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab_hackathon/ALGORITHMS_GUIDE.md). For complete code templates and input/output examples, consult [`CODE_SNIPPETS_AND_REUSABLE_MODULES.md`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab_hackathon/CODE_SNIPPETS_AND_REUSABLE_MODULES.md).*
