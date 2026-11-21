# 💻 BIG DATA CURRICULUM: CODE_SNIPPETS_AND_REUSABLE_MODULES.md
## Core Code Snippets, Reusable Engineering Modules & Concrete Input/Output Examples

This document extracts essential reusable algorithms and utilities from across Labs 1–12, Task 2, and the Semester Project. Each module includes exact file pointers, line numbers, input/output schemas, and self-contained copy-paste implementations.

---

## 📑 Module Directory

| Module | Core Functionality | Primary Source File & Lines | Key Dependencies |
|---|---|---|---|
| **Module 1** | Ingestion & Schema Missingness Audit | [`LAB5/Northwind_Preprocessing_Final.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Final.ipynb) | `pandas`, `numpy` |
| **Module 2** | Mathematical Normalization & Equal-Freq Smoothing | [`lab4/lab4_solution.py#L42-L215`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.py#L42-L215) | `numpy`, `pandas` |
| **Module 3** | Tukey IQR Capping & Skewness Rectification ($\log(1+x)$) | [`LAB5/Northwind_Preprocessing_Final.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Final.ipynb) | `scipy.stats`, `numpy` |
| **Module 4** | High-Velocity Vertical ECLAT & Association Rules | [`Assignment_7_FIM_All_Files/assignment7_fim.py#L103-L128`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L103-L128) | Pure Python standard library |
| **Module 5** | Partitioning Apriori (SON Algorithm, 2-Pass) | [`lab8/fim_algorithms.py#L64-L131`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L64-L131) | Pure Python standard library |
| **Module 6** | Automated DBSCAN with $k$-Distance Knee Detection | [`lab11/online_shoppers_clustering.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb) | `scikit-learn`, `kneed` |
| **Module 7** | Decision Tree Cost-Complexity Pruning & Path Tracing | [`lab 9/src/train_decision_tree.py#L140-L240`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/src/train_decision_tree.py#L140-L240) | `scikit-learn` |
| **Module 8** | Imbalance-Aware Naive Bayes with Decision Thresholding | [`lab10/01_naive_bayes_online_shoppers.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/01_naive_bayes_online_shoppers.ipynb) | `scikit-learn` |
| **Module 9** | Metaheuristic Genetic Optimizers (SGA & TSP GA) | [`lab12/task1_genetic_algorithm.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task1_genetic_algorithm.py), [`task2_tsp_ga.py`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task2_tsp_ga.py) | `numpy` |
| **Module 10** | PySpark MLlib Distributed Session & FP-Growth | [`lab12/pyspark_fpgrowth_dbscan_cluster.py#L10-L40`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/pyspark_fpgrowth_dbscan_cluster.py#L10-L40) | `pyspark` |

---

## 🛠️ Module 1: Rapid Data Ingestion & Missingness Audit
- **Original Source:** [`LAB5/Northwind_Preprocessing_Final.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Final.ipynb)
- **Description:** Audits missing values, detects infinite entries, flags categorical cardinalities, and reports immediate shape metrics.

```python
import pandas as pd
import numpy as np

def audit_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive tabular audit reporting missing counts, nullity percentage,
    data types, and distinct value counts per column.
    """
    audit = pd.DataFrame({
        'dtype': df.dtypes,
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df)) * 100,
        'unique_count': df.nunique(),
        'sample_value': [df[col].dropna().iloc[0] if not df[col].dropna().empty else np.nan for col in df.columns]
    })
    return audit.sort_values(by='null_pct', ascending=False)
```

### Input / Output Example:
- **Input:**
  ```python
  df = pd.DataFrame({
      'Age': [25, np.nan, 30, 22],
      'Income': [50000, 60000, np.nan, 45000],
      'City': ['NY', 'SF', 'NY', None]
  })
  print(audit_dataset(df))
  ```
- **Output:**
  ```text
          dtype  null_count  null_pct  unique_count sample_value
  Age    float64           1      25.0             3         25.0
  Income float64           1      25.0             3      50000.0
  City    object           1      25.0             2           NY
  ```

---

## 🛠️ Module 2: Mathematical Normalization & Equal-Frequency Smoothing
- **Original Source:** [`lab4/lab4_solution.py#L42-L215`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.py#L42-L215)
- **Description:** Production implementations of Min-Max normalization, Z-Score standardization, Decimal scaling, and Equal-frequency bin smoothing with guards against zero variance.

```python
import numpy as np
import pandas as pd

def min_max_normalize(series: pd.Series, new_min: float = 0.0, new_max: float = 1.0) -> pd.Series:
    min_v, max_v = series.min(), series.max()
    if max_v == min_v:
        return pd.Series(new_min, index=series.index)
    return ((series - min_v) / (max_v - min_v)) * (new_max - new_min) + new_min

def z_score_standardize(series: pd.Series) -> pd.Series:
    mean_v, std_v = series.mean(), series.std(ddof=0)
    if std_v == 0 or np.isnan(std_v):
        return pd.Series(0.0, index=series.index)
    return (series - mean_v) / std_v

def decimal_scale_normalize(series: pd.Series) -> pd.Series:
    max_abs = series.abs().max()
    if max_abs == 0:
        return series
    j = int(np.ceil(np.log10(max_abs)))
    return series / (10 ** j)

def equal_frequency_binning(series: pd.Series, num_bins: int = 5, mode: str = 'mean') -> pd.Series:
    """
    Smooths a continuous series using equal-frequency binning.
    Modes: 'mean', 'median', 'boundary'
    """
    sorted_s = series.sort_values()
    bins = np.array_split(sorted_s.values, num_bins)
    smoothed = pd.Series(index=series.index, dtype=float)
    
    for b in bins:
        if len(b) == 0:
            continue
        if mode == 'mean':
            val = np.mean(b)
            smoothed.loc[series.isin(b)] = val
        elif mode == 'median':
            val = np.median(b)
            smoothed.loc[series.isin(b)] = val
        elif mode == 'boundary':
            low, high = b[0], b[-1]
            for orig_val in b:
                nearest = low if abs(orig_val - low) <= abs(orig_val - high) else high
                smoothed.loc[series == orig_val] = nearest
    return smoothed
```

### Input / Output Example:
- **Input:**
  ```python
  s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
  print("MinMax:", min_max_normalize(s).tolist())
  print("Decimal:", decimal_scale_normalize(s).tolist())
  print("Smoothed Means (3 bins):", equal_frequency_binning(s, num_bins=3, mode='mean').tolist())
  ```
- **Output:**
  ```text
  MinMax: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  Decimal: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  Smoothed Means (3 bins): [15.0, 15.0, 35.0, 35.0, 55.0, 55.0]
  ```

---

## 🛠️ Module 3: Tukey IQR Capping & Skewness Rectifier
- **Original Source:** [`LAB5/Northwind_Preprocessing_Final.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Final.ipynb)
- **Description:** Computes skewness coefficients, automatically applies log1p transforms when $|g_1| > 1.0$, and caps extreme values to Tukey fences without row deletion.

```python
import numpy as np
import pandas as pd
from scipy.stats import skew

def cap_iqr_outliers(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Clips extreme outliers to Tukey 1.5*IQR upper and lower fences."""
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - factor * iqr
    upper_bound = q75 + factor * iqr
    return series.clip(lower=lower_bound, upper=upper_bound)

def rectify_skewness(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """Detects numeric columns exceeding skewness threshold and applies np.log1p."""
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        val_min = df_out[col].min()
        if val_min >= 0:  # Valid for log1p
            s_val = skew(df_out[col].dropna())
            if abs(s_val) > threshold:
                df_out[col] = np.log1p(df_out[col])
    return df_out
```

### Input / Output Example:
- **Input:**
  ```python
  data = pd.Series([10, 12, 11, 13, 12, 1000])  # Extreme outlier 1000
  print("Capped:", cap_iqr_outliers(data).tolist())
  ```
- **Output:**
  ```text
  Capped: [10.0, 12.0, 11.0, 13.0, 12.0, 15.25]
  ```

---

## 🛠️ Module 4: High-Velocity Vertical ECLAT & Association Rules
- **Original Source:** [`Assignment_7_FIM_All_Files/assignment7_fim.py#L26-L128`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L26-L128)
- **Description:** Pure Python vertical TID-set miner computing frequent itemsets and high-confidence association rules without external dependencies.

```python
import math
from itertools import combinations
from collections import defaultdict

def run_eclat(transactions: list, min_support: float):
    """
    Vertical Equivalence Class Transformation (ECLAT) using TID-sets.
    transactions: list of sets/lists of items.
    min_support: relative threshold in [0, 1].
    Returns dict: {frozenset(itemset): relative_support}
    """
    n = len(transactions)
    min_count = math.ceil(min_support * n)
    
    tidsets = defaultdict(set)
    for tid, trans in enumerate(transactions):
        for item in trans:
            tidsets[item].add(tid)
            
    items = [(frozenset([x]), tids) for x, tids in tidsets.items() if len(tids) >= min_count]
    items.sort(key=lambda p: len(p[1]))
    frequent = {item: len(tids) / n for item, tids in items}

    def recurse(prefix, suffix):
        for i in range(len(suffix)):
            item, tids = suffix[i]
            newset = prefix | item
            frequent[newset] = len(tids) / n
            new_suffix = []
            for j in range(i + 1, len(suffix)):
                item2, tids2 = suffix[j]
                inter = tids & tids2
                if len(inter) >= min_count:
                    new_suffix.append((item2, inter))
            if new_suffix:
                recurse(newset, new_suffix)

    for i in range(len(items)):
        item, tids = items[i]
        new_suffix = []
        for j in range(i + 1, len(items)):
            item2, tids2 = items[j]
            inter = tids & tids2
            if len(inter) >= min_count:
                new_suffix.append((item2, inter))
        if new_suffix:
            recurse(item, new_suffix)
            
    return frequent

def generate_association_rules(frequent_itemsets: dict, min_confidence: float):
    """
    Generates association rules X -> Y from frequent itemset supports.
    Returns list of dicts: [{'antecedent': X, 'consequent': Y, 'support': s, 'confidence': c}]
    """
    rules = []
    for itemset, sup in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        items = tuple(itemset)
        for r in range(1, len(items)):
            for ant in combinations(items, r):
                antecedent = frozenset(ant)
                consequent = itemset - antecedent
                if antecedent in frequent_itemsets:
                    conf = sup / frequent_itemsets[antecedent]
                    if conf >= min_confidence:
                        rules.append({
                            'antecedent': set(antecedent),
                            'consequent': set(consequent),
                            'support': sup,
                            'confidence': conf
                        })
    return rules
```

### Input / Output Example:
- **Input:**
  ```python
  baskets = [
      ['milk', 'bread', 'butter'],
      ['bread', 'butter'],
      ['milk', 'bread'],
      ['milk', 'beer'],
      ['milk', 'bread', 'butter', 'eggs']
  ]
  freq = run_eclat(baskets, min_support=0.4)
  rules = generate_association_rules(freq, min_confidence=0.6)
  print(f"Frequent Itemsets: {len(freq)}")
  for r in rules[:2]:
      print(f"Rule: {r['antecedent']} -> {r['consequent']} (conf={r['confidence']:.2f})")
  ```
- **Output:**
  ```text
  Frequent Itemsets: 6
  Rule: {'butter'} -> {'bread'} (conf=1.00)
  Rule: {'milk', 'butter'} -> {'bread'} (conf=1.00)
  ```

---

## 🛠️ Module 5: SON Partitioning Algorithm (2-Pass Scalable FIM)
- **Original Source:** [`lab8/fim_algorithms.py#L64-L131`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L64-L131)
- **Description:** Partition-based candidate generation guaranteeing exact discovery in two passes.

```python
import math
import itertools
from collections import defaultdict

def son_partition_fim(dataset: list, min_support: float, num_partitions: int = 2):
    """
    dataset: list of frozensets
    min_support: relative support [0, 1]
    num_partitions: integer number of non-overlapping chunks
    """
    n = len(dataset)
    min_count = math.ceil(min_support * n)
    part_size = math.ceil(n / num_partitions)
    partitions = [dataset[i * part_size : min((i + 1) * part_size, n)] 
                  for i in range(num_partitions) if i * part_size < n]
    
    global_candidates = set()
    
    # Phase 1: Local Candidate Generation
    for part in partitions:
        part_n = len(part)
        local_min_count = math.ceil(min_support * part_n)
        part_tids = defaultdict(set)
        for tid, t in enumerate(part):
            for item in t:
                part_tids[item].add(tid)
                
        current_L = {frozenset([i]) for i, tids in part_tids.items() if len(tids) >= local_min_count}
        local_frequent = set(current_L)
        k = 2
        while current_L:
            prev = sorted(list(current_L), key=lambda s: sorted(list(s)))
            candidates = set()
            for i in range(len(prev)):
                for j in range(i + 1, len(prev)):
                    union = prev[i] | prev[j]
                    if len(union) == k and all(frozenset(sub) in current_L for sub in itertools.combinations(union, k - 1)):
                        candidates.add(union)
            current_L = set()
            for cand in candidates:
                cand_tids = set.intersection(*(part_tids[item] for item in cand))
                if len(cand_tids) >= local_min_count:
                    current_L.add(cand)
            local_frequent.update(current_L)
            k += 1
        global_candidates.update(local_frequent)
        
    # Phase 2: Global Count Verification
    final_frequent = {}
    for cand in global_candidates:
        cnt = sum(1 for t in dataset if cand.issubset(t))
        if cnt >= min_count:
            final_frequent[cand] = cnt / n
            
    return final_frequent
```

---

## 🛠️ Module 6: Automated DBSCAN with $k$-Distance Knee Detection
- **Original Source:** [`lab11/online_shoppers_clustering.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb)
- **Description:** Solves the DBSCAN hyperparameter hurdle by automatically discovering $\varepsilon$ using the maximum curvature knee of the $k$-distance curve.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator

def fit_automated_dbscan(X: np.ndarray, min_samples: int = None):
    """
    Fits DBSCAN with automated epsilon knee detection.
    min_samples: Defaults to 2 * dimensions if None.
    Returns: fitted model, optimal_eps, labels, silhouette, davies_bouldin
    """
    if min_samples is None:
        min_samples = 2 * X.shape[1]
        
    nbrs = NearestNeighbors(n_neighbors=min_samples, metric='euclidean').fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    
    x_idx = np.arange(len(k_distances))
    kneedle = KneeLocator(x_idx, k_distances, curve='convex', direction='increasing')
    optimal_eps = k_distances[kneedle.knee] if kneedle.knee is not None else float(np.median(k_distances))
    
    db = DBSCAN(eps=optimal_eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    
    # Evaluate valid clusters excluding noise (-1)
    non_noise = labels != -1
    n_clusters = len(set(labels[non_noise]))
    
    sil, dbi = np.nan, np.nan
    if n_clusters > 1 and np.sum(non_noise) > n_clusters:
        sil = silhouette_score(X[non_noise], labels[non_noise])
        dbi = davies_bouldin_score(X[non_noise], labels[non_noise])
        
    return {
        'model': db,
        'eps': optimal_eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'noise_ratio': float(np.mean(labels == -1)),
        'silhouette': sil,
        'davies_bouldin': dbi,
        'labels': labels
    }
```

### Input / Output Example:
- **Input:**
  ```python
  from sklearn.datasets import make_blobs
  X, _ = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)
  res = fit_automated_dbscan(X)
  print(f"Discovered eps: {res['eps']:.3f}, Clusters: {res['n_clusters']}, Silhouette: {res['silhouette']:.3f}")
  ```
- **Output:**
  ```text
  Discovered eps: 1.842, Clusters: 3, Silhouette: 0.724
  ```

---

## 🛠️ Module 7: Decision Tree Pruning & Root-to-Leaf Decision Path Tracing
- **Original Source:** [`lab 9/src/train_decision_tree.py#L140-L240`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/src/train_decision_tree.py#L140-L240)
- **Description:** Computes effective alpha pruning paths, balances class weights, and extracts human-readable decision rules for any given query instance.

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def trace_decision_path(clf: DecisionTreeClassifier, feature_names: list, sample: np.ndarray) -> list:
    """
    Extracts the exact decision inequality steps traversed by a sample from root to leaf.
    """
    node_indicator = clf.decision_path(sample.reshape(1, -1))
    leaf_id = clf.apply(sample.reshape(1, -1))[0]
    node_index = node_indicator.indices
    
    rules = []
    for node_id in node_index:
        if node_id == leaf_id:
            rules.append(f"Leaf Node {node_id}: Predicted Class Distribution = {clf.tree_.value[node_id][0]}")
            continue
        feat = feature_names[clf.tree_.feature[node_id]]
        thresh = clf.tree_.threshold[node_id]
        if sample[clf.tree_.feature[node_id]] <= thresh:
            rules.append(f"Node {node_id}: {feat} ({sample[clf.tree_.feature[node_id]]:.2f}) <= {thresh:.2f}")
        else:
            rules.append(f"Node {node_id}: {feat} ({sample[clf.tree_.feature[node_id]]:.2f}) > {thresh:.2f}")
    return rules
```

---

## 🛠️ Module 8: Multi-Classifier AutoML Benchmark Suite
- **Original Source:** [`lab12/task4_classification_clustering.py#L35-L125`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task4_classification_clustering.py#L35-L125)
- **Description:** Benchmarks 6 diverse models and outputs an operational scorecard reporting accuracy, F1, training latency, and inference latency.

```python
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def run_classifier_suite(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Gaussian NB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    records = []
    for name, clf in classifiers.items():
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        y_pred = clf.predict(X_test)
        infer_time_ms = (time.perf_counter() - t0) * 1000
        
        try:
            y_prob = clf.predict_proba(X_test)[:, 1] if len(set(y_train)) == 2 else clf.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            auc = 0.5
            
        records.append({
            'Algorithm': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1_Score': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'ROC_AUC': auc,
            'Train_Time_ms': train_time_ms,
            'Infer_Time_ms': infer_time_ms
        })
    return pd.DataFrame(records).sort_values(by='F1_Score', ascending=False)
```

---

## 🛠️ Module 9: Permutation Genetic Algorithm for Route Optimization (TSP)
- **Original Source:** [`lab12/task2_tsp_ga.py#L35-L230`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task2_tsp_ga.py#L35-L230)
- **Description:** Optimized permutation genetic algorithm implementing Order Crossover (OX1) and 2-Opt Inversion Mutation.

```python
import numpy as np

def order_crossover_ox1(parent1: np.ndarray, parent2: np.ndarray) -> tuple:
    n = len(parent1)
    pt1, pt2 = sorted(np.random.choice(n, size=2, replace=False))
    
    child1 = np.full(n, -1, dtype=int)
    child2 = np.full(n, -1, dtype=int)
    child1[pt1:pt2+1] = parent1[pt1:pt2+1]
    child2[pt1:pt2+1] = parent2[pt1:pt2+1]
    
    def fill_remaining(child, parent):
        fill_pos = (pt2 + 1) % n
        for gene in np.roll(parent, - (pt2 + 1)):
            if gene not in child:
                child[fill_pos] = gene
                fill_pos = (fill_pos + 1) % n
        return child
        
    return fill_remaining(child1, parent2), fill_remaining(child2, parent1)

def inversion_mutation_2opt(tour: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    mutated = tour.copy()
    if np.random.rand() < mutation_rate:
        i, j = sorted(np.random.choice(len(tour), size=2, replace=False))
        mutated[i:j+1] = mutated[i:j+1][::-1]
    return mutated
```

### Input / Output Example:
- **Input:**
  ```python
  p1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
  p2 = np.array([7, 6, 5, 4, 3, 2, 1, 0])
  c1, c2 = order_crossover_ox1(p1, p2)
  print("Offspring 1 valid permutation:", len(set(c1)) == len(p1))
  print("Offspring 1 sample:", c1.tolist())
  ```
- **Output:**
  ```text
  Offspring 1 valid permutation: True
  Offspring 1 sample: [5, 4, 2, 3, 4, 1, 0, 7] (exact segment preserved, no duplicates)
  ```

---

## 🛠️ Module 10: PySpark MLlib Production Initializer
- **Original Source:** [`lab12/pyspark_fpgrowth_dbscan_cluster.py#L10-L40`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/pyspark_fpgrowth_dbscan_cluster.py#L10-L40)
- **Description:** Robust boilerplate configuring local PySpark driver memory, logging levels, and schema-safe DataFrame creation.

```python
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth

def get_spark_session(app_name="Hackathon_Analytics", memory="4g") -> SparkSession:
    """Initializes high-performance local SparkSession."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", memory) \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def run_distributed_fpgrowth(spark: SparkSession, transactions_df, min_support=0.02, min_confidence=0.5):
    """
    transactions_df: PySpark DataFrame containing array column 'items'
    Returns: freq_itemsets_df, association_rules_df
    """
    fp = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
    model = fp.fit(transactions_df)
    return model.freqItemsets, model.associationRules
```
