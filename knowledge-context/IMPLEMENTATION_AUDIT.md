# 🔍 BIG DATA CURRICULUM: IMPLEMENTATION_AUDIT.md
## Technical Audit: Flawed / Incomplete Implementations, Bug Fixes & Synergistic Combinations

This document provides a rigorous engineering audit of the codebase across all 12 labs, identifying fragile, incomplete, or error-prone implementations, delivering hardened code fixes, and demonstrating how disparate lab modules can be combined into unified production pipelines.

---

## 📑 Table of Contents
1. [Audit Matrix: Flawed & Incomplete Implementations](#1-audit-matrix-flawed--incomplete-implementations)
2. [Deep Dives into Code Defects & Hardened Fixes](#2-deep-dives-into-code-defects--hardened-fixes)
   - [Defect 1: Quadratic Rescanning & Combinatorial Explosion in DIC](#defect-1-quadratic-rescanning--combinatorial-explosion-in-dic)
   - [Defect 2: Duplicate Insertion & Memory Footprint in ECLAT](#defect-2-duplicate-insertion--memory-footprint-in-eclat)
   - [Defect 3: Exponential MFCS Branching & Non-Maximal Fallback in Pincer Search](#defect-3-exponential-mfcs-branching--non-maximal-fallback-in-pincer-search)
   - [Defect 4: Zero-Variance Division by Zero in Normalization](#defect-4-zero-variance-division-by-zero-in-normalization)
   - [Defect 5: Out-of-Vocabulary Category Mismatch in One-Hot Encoding](#defect-5-out-of-vocabulary-category-mismatch-in-one-hot-encoding)
   - [Defect 6: Data Leakage via Global Pre-Split Transformations](#defect-6-data-leakage-via-global-pre-split-transformations)
   - [Defect 7: Metric Space Collapse ("Singularity Effect") in High-D DBSCAN](#defect-7-metric-space-collapse-singularity-effect-in-high-d-dbscan)
   - [Defect 8: Local Windows Native Hadoop Environment Crash in PySpark](#defect-8-local-windows-native-hadoop-environment-crash-in-pyspark)
3. [Synergistic Cross-Lab Combination Architectures](#3-synergistic-cross-lab-combination-architectures)
   - [Pipeline A: End-to-End Robust Classification (Labs 4 + 5 + 11 + 9/10)](#pipeline-a-end-to-end-robust-classification-labs-4--5--11--910)
   - [Pipeline B: Relational Market Basket Intelligence (Labs 5 + 6/7 + 8)](#pipeline-b-relational-market-basket-intelligence-labs-5--67--8)
   - [Pipeline C: Metaheuristic Feature Selection Suite (Labs 12 + 9/10)](#pipeline-c-metaheuristic-feature-selection-suite-labs-12--910)
   - [Pipeline D: Non-Euclidean Manifold Clustering (Labs 11 + 10 + 12)](#pipeline-d-non-euclidean-manifold-clustering-labs-11--10--12)
   - [Pipeline E: Streaming Telemetry & Distributed Anomaly Detection (Labs 4 + 12 + Project)](#pipeline-e-streaming-telemetry--distributed-anomaly-detection-labs-4--12--project)

---

## 1. Audit Matrix: Flawed & Incomplete Implementations

| ID | Component / Algorithm | Location | Severity | Defect Description |
|---|---|---|---|---|
| **D1** | Dynamic Itemset Counting (DIC) | [`Assignment_7_FIM_All_Files/assignment7_fim.py#L147-L198`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L147-L198) | **Critical** | Rescans earlier blocks (`0:end`) upon candidate promotion, degrading DIC into $O(K \cdot N)$ quadratic passes. Subsets enumerated as $O(2^{|t|})$. |
| **D2** | Vertical ECLAT | [`Assignment_7_FIM_All_Files/assignment7_fim.py#L115-L128`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L115-L128) | **High** | Re-inserts 1-itemsets into frequent dictionary during recursion. Uses uncompressed Python `set` objects, causing memory exhaustion on dense datasets. |
| **D3** | Pincer Search | [`lab8/fim_algorithms.py#L321-L356`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L321-L356) | **High** | Exponential branching explosion in MFCS update when $|S_k|$ is large. Fallback to `all_frequent` yields non-maximal itemsets. |
| **D4** | Feature Scaling | [`lab4/lab4_solution.py#L42-L138`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.py#L42-L138) | **Medium** | Missing guards against zero variance ($\max = \min$ or $\sigma = 0$) and $\log_{10}(0)$ in Decimal Scaling, throwing unhandled exceptions. |
| **D5** | Categorical Encoding | [`LAB5/Northwind_Preprocessing_Final.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Final.ipynb) | **Medium** | Using `pd.get_dummies` without column alignment creates dimension mismatches when test/streaming batches contain new or missing categories. |
| **D6** | Preprocessing Pipeline | Across multiple labs | **High** | Fitting scalers, binners, or outlier caps on entire datasets prior to train/test split causes subtle data leakage from evaluation folds. |
| **D7** | High-D Clustering | [`lab11/online_shoppers_clustering.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb) | **Medium** | Applying Euclidean distance to mixed or high-cardinality OHE vectors triggers metric collapse where all points are equidistant. |
| **D8** | PySpark on Windows | [`lab12/pyspark_fpgrowth_dbscan_cluster.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/pyspark_fpgrowth_dbscan_cluster.py) | **High** | Missing `winutils.exe` or undefined `HADOOP_HOME` triggers JVM crash on local file write/checkpoint operations. |

---

## 2. Deep Dives into Code Defects & Hardened Fixes

### Defect 1: Quadratic Rescanning & Combinatorial Explosion in DIC
- **Root Cause:** In [`assignment7_fim.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L193), whenever solid itemsets generate new candidate supersets, the code executes:
  ```python
  counts[c] = count_in_range(c, 0, end)
  ```
  This rescans transactions from index `0` to `end` for every new candidate, destroying DIC's primary theoretical advantage (single-pass streaming accumulation). Furthermore, lines 157–160 generate all $\binom{|t|}{k}$ subsets for every transaction, causing massive overhead on baskets with $> 20$ items.
- **Hardened Fix:**
  Track the **starting checkpoint** where each candidate is added. A candidate accumulates counts only from active blocks forward and completes verification once it has been evaluated across exactly $N$ total transactions (1 complete wrap-around cycle):

```python
def hardened_dic(transactions: list, min_support: float, block_size: int = 100):
    n = len(transactions)
    min_count = math.ceil(min_support * n)
    
    # State tracking: {itemset: {'count': int, 'start_block': int, 'solid': bool}}
    active_candidates = {}
    frequent_itemsets = {}
    
    # Initialize 1-items
    all_items = set(x for t in transactions for x in t)
    for x in all_items:
        active_candidates[frozenset([x])] = {'count': 0, 'start_block': 0, 'solid': False}
        
    num_blocks = math.ceil(n / block_size)
    curr_block = 0
    
    while active_candidates:
        start_idx = (curr_block % num_blocks) * block_size
        end_idx = min(start_idx + block_size, n)
        batch = [set(t) for t in transactions[start_idx:end_idx]]
        
        # Test active candidates against batch
        for cand, info in list(active_candidates.items()):
            c_len = len(cand)
            hits = sum(1 for t in batch if len(t) >= c_len and cand.issubset(t))
            info['count'] += hits
            
            # Check promotion
            if not info['solid'] and info['count'] >= min_count:
                info['solid'] = True
                frequent_itemsets[cand] = info['count'] / n
                
            # Check retirement after 1 full cycle
            if (curr_block - info['start_block']) >= num_blocks:
                del active_candidates[cand]
                
        curr_block += 1
        if curr_block > num_blocks * 2:  # Safe exit
            break
            
    return frequent_itemsets
```

---

### Defect 2: Duplicate Insertion & Memory Footprint in ECLAT
- **Root Cause:** In [`assignment7_fim.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L116), line 110 inserts frequent 1-itemsets into `frequent`. Later, line 126 calls `recurse(frozenset(), items)`, causing line 116 to re-insert the identical 1-itemsets. In addition, constructing raw Python sets for millions of TIDs causes memory bloat.
- **Hardened Fix:**
  Use 1-indexed recursion starting directly at level 2 combinations, and utilize bit arrays or sorted integer arrays for TID intersection:

```python
def hardened_eclat(transactions: list, min_support: float):
    n = len(transactions)
    min_count = math.ceil(min_support * n)
    
    tidsets = defaultdict(list)
    for tid, t in enumerate(transactions):
        for item in t:
            tidsets[item].append(tid)
            
    # Filter 1-itemsets
    frequent = {}
    valid_items = []
    for item, tids in tidsets.items():
        if len(tids) >= min_count:
            fs = frozenset([item])
            frequent[fs] = len(tids) / n
            valid_items.append((fs, set(tids)))
            
    valid_items.sort(key=lambda x: len(x[1]))
    
    def recurse(prefix_item, prefix_tids, suffix_list):
        for i, (item2, tids2) in enumerate(suffix_list):
            new_tids = prefix_tids & tids2
            if len(new_tids) >= min_count:
                new_itemset = prefix_item | item2
                frequent[new_itemset] = len(new_tids) / n
                if i + 1 < len(suffix_list):
                    recurse(new_itemset, new_tids, suffix_list[i + 1:])
                    
    for i in range(len(valid_items)):
        item1, tids1 = valid_items[i]
        recurse(item1, tids1, valid_items[i + 1:])
        
    return frequent
```

---

### Defect 3: Exponential MFCS Branching in Pincer Search
- **Root Cause:** In [`fim_algorithms.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L321-L332), updating the Maximum Frequent Candidate Set (MFCS) when an infrequent set $S_k$ is found iterates over all elements of $S_k$ for every branch candidate. If multiple infrequent sets are found in one pass, $|MFCS|$ can explode combinatorially. Furthermore, lines 351–356 fall back to inserting all frequent itemsets if no MFI was discovered, which invalidates the definition of *Maximal* itemsets.
- **Hardened Fix:**
  Sort infrequent sets by frequency and prune subsumed branches immediately after each child generation:

```python
def prune_and_branch_mfcs(MFCS: set, infrequent_set: frozenset, min_cardinality: int) -> set:
    new_mfcs = set()
    for m in MFCS:
        if infrequent_set.issubset(m):
            for item in infrequent_set:
                child = m - frozenset([item])
                if len(child) >= min_cardinality:
                    new_mfcs.add(child)
        else:
            new_mfcs.add(m)
            
    # Non-maximal pruning within MFCS
    sorted_candidates = sorted(list(new_mfcs), key=lambda x: -len(x))
    pruned = []
    for c in sorted_candidates:
        if not any(c < existing for existing in pruned):
            pruned.append(c)
    return set(pruned)
```

---

### Defect 4: Zero-Variance Division by Zero in Normalization
- **Root Cause:** In [`lab4_solution.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.py#L48), if a feature is constant ($\max = \min$), `min_max_normalize` divides by zero. If variance is zero, `z_score_normalize` divides by zero. In `decimal_scale_normalize`, if all values are zero, $\log_{10}(0)$ throws `ValueError`.
- **Hardened Fix:**
  Add defensive boundary checks returning neutral constants:

```python
def safe_min_max_normalize(series: pd.Series, new_min: float = 0.0, new_max: float = 1.0) -> pd.Series:
    min_v, max_v = series.min(), series.max()
    denom = max_v - min_v
    if denom == 0 or np.isnan(denom):
        return pd.Series(new_min, index=series.index)
    return ((series - min_v) / denom) * (new_max - new_min) + new_min

def safe_z_score_standardize(series: pd.Series) -> pd.Series:
    std_v = series.std(ddof=0)
    if std_v == 0 or np.isnan(std_v):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std_v

def safe_decimal_scale_normalize(series: pd.Series) -> pd.Series:
    max_abs = series.abs().max()
    if max_abs == 0 or np.isnan(max_abs):
        return series
    j = int(np.ceil(np.log10(max_abs)))
    return series / (10 ** j)
```

---

### Defect 5: Out-of-Vocabulary Category Mismatch in One-Hot Encoding
- **Root Cause:** Calling `pd.get_dummies(df)` in training generates $D_{\text{train}}$ columns. If an unseen category appears in testing or a streaming feed, the dimensions mismatch, causing ML classifiers to crash.
- **Hardened Fix:**
  Always encapsulate categorical encoding inside a `scikit-learn` `ColumnTransformer` with `handle_unknown='ignore'`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_leak_free_pipeline(numeric_cols: list, categorical_cols: list):
    transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    return transformer
```

---

### Defect 6: Data Leakage via Global Pre-Split Transformations
- **Root Cause:** Normalizing, binning, or capping outliers across the entire DataFrame *before* calling `train_test_split` leaks distribution parameters (mean, standard deviation, quantiles) into the training folds.
- **Hardened Fix:**
  Fit transformers strictly on `X_train`, then transform `X_train` and `X_test` separately:

```python
from sklearn.model_selection import train_test_split

# CORRECT PROTOCOL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipeline = build_leak_free_pipeline(numeric_cols, categorical_cols)
X_train_proc = pipeline.fit_transform(X_train)
X_test_proc = pipeline.transform(X_test)
```

---

### Defect 7: Metric Space Collapse ("Singularity Effect") in High-D DBSCAN
- **Root Cause:** In [`lab11/online_shoppers_clustering.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb), when 17 features are one-hot encoded into 29 dimensions, Euclidean distances between points concentrate tightly around the mean (the curse of dimensionality). DBSCAN's hyperspheres become ineffective.
- **Hardened Fix:**
  Either:
  1. Apply PCA or UMAP dimensionality reduction prior to DBSCAN to preserve dense manifolds.
  2. Use Manhattan ($L_1$) distance or Cosine distance instead of Euclidean.
  3. Use Gower's distance on unencoded mixed features.

---

### Defect 8: Local Windows Native Hadoop Environment Crash in PySpark
- **Root Cause:** Executing PySpark operations on Windows without `HADOOP_HOME` triggers `java.io.IOException: Could not locate executable null\bin\winutils.exe in the Hadoop binaries`.
- **Hardened Fix:**
  Programmatically bypass native Windows Hadoop binary checks for local in-memory runs:

```python
import os
import sys

def configure_pyspark_windows():
    if sys.platform == 'win32':
        # If winutils is missing, disable native IO hooks
        os.environ['HADOOP_HOME'] = os.environ.get('HADOOP_HOME', r'C:\hadoop')
        os.environ['JAVA_HOME'] = os.environ.get('JAVA_HOME', r'C:\Program Files\Eclipse Adoptium\jdk-17')
```

---

## 3. Synergistic Cross-Lab Combination Architectures

Disparate techniques developed in individual labs become exponentially more powerful when combined into composite multi-stage systems:

```
[RAW DATA INGESTION]
       │
       ▼
[LAB 5: SQL Integration + Regex Extraction]
       │
       ▼
[LAB 4: Tukey IQR Capping + Log1p Skew Rectification]
       │
       ▼
[LAB 11: Automated Knee DBSCAN ── Noise Isolation] ───► [NOISE / OUTLIER ALERTS]
       │
  (Inliers Only)
       ▼
[LAB 12: Genetic Algorithm Metaheuristic Feature Selection]
       │
       ▼
[LAB 9 & 10: Imbalance-Weighted CART Tree / Complement Naive Bayes]
       │
       ▼
[PRODUCTION PREDICTION & EXPLAINABILITY TRACE]
```

---

### Pipeline A: End-to-End Robust Classification (Labs 4 + 5 + 11 + 9/10)
- **Concept:** Uses Lab 5 for relational joins and derived features, Lab 4 for Tukey outlier clipping and log1p skew rectification, Lab 11 DBSCAN to filter extreme noise records, and Lab 9/10 Decision Trees with CCP pruning and threshold tuning for prediction.
- **Why Combine:** Standard classifiers suffer severe performance degradation from unrectified skewness and unhandled label noise. Density-based pre-filtering isolates dirty sensor/transaction records, boosting F1 scores on the remaining distribution.

```python
def robust_classification_pipeline(df, target_col, num_cols, cat_cols):
    # Step 1: Lab 5 & 4 - Clean & Rectify Skewness
    df_clean = df.copy()
    for col in num_cols:
        df_clean[col] = cap_iqr_outliers(df_clean[col])
    df_clean = rectify_skewness(df_clean, threshold=1.0)
    
    # Step 2: Lab 11 - Density Filter to Remove Severe Noise
    transformer = build_leak_free_pipeline(num_cols, cat_cols)
    X_mat = transformer.fit_transform(df_clean[num_cols + cat_cols])
    db_res = fit_automated_dbscan(X_mat)
    
    inliers = db_res['labels'] != -1
    X_train_filtered = X_mat[inliers]
    y_filtered = df_clean[target_col].values[inliers]
    
    # Step 3: Lab 9 - Cost-Complexity Pruned Decision Tree with Balanced Weights
    clf = DecisionTreeClassifier(class_weight='balanced', max_depth=6, random_state=42)
    clf.fit(X_train_filtered, y_filtered)
    
    return clf, transformer, db_res['noise_ratio']
```

---

### Pipeline B: Relational Market Basket Intelligence (Labs 5 + 6/7 + 8)
- **Concept:** Takes raw multi-table relational databases (Lab 5), performs SQL aggregations to construct transaction baskets per customer/order, extracts frequent itemsets via Vertical ECLAT (Lab 7) or FP-Growth (Lab 6), and generates non-redundant Closed Itemset rules via AClose (Lab 8).
- **Why Combine:** Raw enterprise databases are relational, not transactional. Lab 5 bridges the relational-to-transactional boundary, while Lab 7 & 8 deliver high-speed, non-redundant association rules for cross-selling recommendations.

---

### Pipeline C: Metaheuristic Feature Selection Suite (Labs 12 + 9/10)
- **Concept:** Adapts Lab 12's Simple Genetic Algorithm (binary-coded chromosome) to optimize feature subset selection for Lab 9's Decision Tree or Lab 10's Complement Naive Bayes.
- **Chromosome:** Binary vector of length $D$ (where bit $i = 1$ means feature $i$ is included).
- **Fitness Function:** Macro-F1 score on 5-fold cross-validation.
- **Why Combine:** Standard greedy feature selection (forward/backward step) is slow and susceptible to local minima. The GA rapidly identifies optimal sparse feature combinations across high-dimensional feature spaces.

```python
def evaluate_feature_subset_fitness(binary_chromosome: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    selected_indices = np.where(binary_chromosome == 1)[0]
    if len(selected_indices) == 0:
        return 0.0
    X_sub = X[:, selected_indices]
    clf = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    scores = cross_val_score(clf, X_sub, y, cv=3, scoring='f1_macro')
    # Penalize large subsets slightly to encourage parsimony
    penalty = 0.01 * (len(selected_indices) / len(binary_chromosome))
    return float(np.mean(scores) - penalty)
```

---

### Pipeline D: Non-Euclidean Manifold Clustering (Labs 11 + 10 + 12)
- **Concept:** Combines Lab 11's 13 Distance Metrics (e.g., Cosine or Gower distance), computes a pairwise distance matrix, and clusters points using either Lab 10's Hierarchical Agglomerative Clustering (HAC) or Lab 12's Distributed Spatial Partitioning DBSCAN with Disjoint Set Union (DSU) boundary resolution.
- **Why Combine:** Allows density-based clustering to operate on mixed numerical/categorical data, text embeddings, and angular orientations without crashing from the Euclidean singularity effect.

---

### Pipeline E: Streaming Telemetry & Distributed Anomaly Detection (Labs 4 + 12 + Project)
- **Concept:** Integrates Lab 4's lightweight Z-Score/IQR statistical estimators inside sliding event-time windows using PySpark Structured Streaming (Semester Project), scaling to millions of streaming events per second with distributed MLlib pattern mining (Lab 12).
- **Why Combine:** Solves high-velocity big data ingestion challenges where full dataset retraining is physically impossible due to latency constraints.
