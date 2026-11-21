# ⚙️ BIG DATA CURRICULUM: ALGORITHMS_GUIDE.md
## Exhaustive Algorithmic Catalog, Formulations, Complexities & Implementation Pointers

This document provides a comprehensive technical reference for every algorithm implemented across Labs 1 through 12, Task 2, and the Semester Project. Each entry contains the formal mathematical definition, step-by-step algorithmic logic, time/space complexity bounds, original file references, and operational considerations.

---

## 📑 Table of Contents
1. [Frequent Itemset Mining (FIM) & Association Rules](#1-frequent-itemset-mining-fim--association-rules)
   - Standard Apriori
   - FP-Growth (Frequent Pattern Growth)
   - Direct Hashing & Pruning (DHP / Hash Apriori)
   - Dynamic Itemset Counting (DIC)
   - Vertical ECLAT (Equivalence Class Transformation)
   - Partitioning Variant of Apriori (SON Algorithm)
   - AClose (Frequent Closed Itemsets & Generators)
   - Pincer Search (Maximal Frequent Itemsets)
   - Distributed PySpark MLlib FP-Growth
2. [Supervised Machine Learning Algorithms](#2-supervised-machine-learning-algorithms)
   - CART Decision Trees (Gini, Entropy, Log Loss)
   - Minimal Cost-Complexity Pruning (CCP)
   - Root-to-Leaf Decision Path Tracing
   - Gaussian Naive Bayes (GaussianNB)
   - Bernoulli Naive Bayes (BernoulliNB)
   - Complement Naive Bayes (ComplementNB)
   - Multinomial Naive Bayes (MultinomialNB)
   - Multi-Classifier Evaluation Suite (SVM, RF, KNN, LogReg)
3. [Unsupervised Machine Learning & Clustering Algorithms](#3-unsupervised-machine-learning--clustering-algorithms)
   - K-Means Clustering & Elbow Method
   - Hierarchical Agglomerative Clustering (HAC) & Cophenetic Correlation
   - Density-Based Spatial Clustering (DBSCAN) & Automated Knee Detection
   - Distributed Spatial Partitioning DBSCAN with Disjoint Set Union (DSU)
   - Ordering Points To Identify Clustering Structure (OPTICS)
   - Gaussian Mixture Models (GMM) with Expectation-Maximization
4. [Evolutionary Algorithms & Metaheuristics](#4-evolutionary-algorithms--metaheuristics)
   - Simple Genetic Algorithm (SGA) for Continuous Optimization
   - Combinatorial Permutation Genetic Algorithm for TSP
5. [Data Preprocessing, Transformations & Feature Engineering](#5-data-preprocessing-transformations--feature-engineering)
   - Min-Max Normalization
   - Z-Score Standardization
   - Decimal Scaling Normalization
   - Equal-Frequency & Equal-Width Binning with Smoothing
   - IQR Outlier Detection & Tukey's Fence Capping
   - Skewness Detection & Logarithmic ($\log(1+x)$) Rectification
   - Multi-Table Relational SQL Integration & Regex Extraction
6. [The 13 Distance and Dissimilarity Metrics Framework](#6-the-13-distance-and-dissimilarity-metrics-framework)

---

## 1. Frequent Itemset Mining (FIM) & Association Rules

### 1.1 Standard Apriori Algorithm
- **Role:** Baseline level-wise candidate generation and database scanning algorithm.
- **Mathematical Principle:** 
  - Anti-monotonicity (Apriori Property): Any subset of a frequent itemset must also be frequent. If $X \subseteq Y$ and $\text{supp}(X) < \text{minsup}$, then $\text{supp}(Y) < \text{minsup}$.
  - Support of itemset $X$: $\text{supp}(X) = \frac{|\{t \in \mathcal{D} \mid X \subseteq t\}|}{|\mathcal{D}|}$.
- **Algorithmic Flow:**
  1. Scan database $\mathcal{D}$ to count support for 1-itemsets $C_1$, retain frequent 1-itemsets $L_1 = \{c \in C_1 \mid \text{supp}(c) \ge \text{minsup}\}$.
  2. For $k = 2, 3, \dots$ until $L_{k-1} = \emptyset$:
     - Candidate generation (Join step): $C_k = \{l_1 \cup l_2 \mid l_1, l_2 \in L_{k-1}, |l_1 \cap l_2| = k-2\}$.
     - Candidate pruning (Prune step): For each $c \in C_k$, if any $(k-1)$-subset of $c$ is not in $L_{k-1}$, remove $c$.
     - Support counting: Scan $\mathcal{D}$ to count occurrences of each $c \in C_k$.
     - Retain $L_k = \{c \in C_k \mid \text{supp}(c) \ge \text{minsup}\}$.
- **Complexity:**
  - Time: $O\left(k_{\max} \cdot |\mathcal{D}| \cdot \sum |C_k|\right)$. Highly exponential in worst-case dense data.
  - Space: $O(|C_k|)$ candidate storage in memory.
- **Original Source References:**
  - Baseline Oracle: [`lab8/fim_algorithms.py#L22-L58`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L22-L58)
  - Production Scikit/MLxtend Wrapper: [`lab6/fim_test.py#L32-L48`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/fim_test.py#L32-L48)

---

### 1.2 FP-Growth (Frequent Pattern Growth)
- **Role:** Non-candidate-generation frequent itemset mining via compact prefix-tree compression.
- **Algorithmic Flow:**
  1. Pass 1: Scan $\mathcal{D}$ to find frequent 1-items $L_1$ and sort them in descending order of frequency.
  2. Pass 2: Build FP-Tree by inserting each transaction's frequent items in descending frequency order. Maintain node links across identical items via a Header Table.
  3. Divide-and-Conquer Recursive Mining:
     - For each item $a_i$ in Header Table (bottom-up):
       - Traverse prefix paths from node links to form the *conditional pattern base* for $a_i$.
       - Construct conditional FP-tree from the conditional pattern base.
       - Recursively mine the conditional tree until tree has a single path or is empty.
- **Complexity:**
  - Time: $O(|\mathcal{D}|)$ database passes (exactly 2 scans) + $O(\text{Tree Size})$ recursive traversal.
  - Space: $O(|\text{Distinct Items}| \cdot |\mathcal{D}|)$ worst case; $O(\text{Compressed Nodes})$ average case.
- **Original Source References:**
  - MLxtend Benchmark: [`lab6/fim_test.py#L32-L48`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab6/fim_test.py#L32-L48)
  - Custom Pure-Python Engine: [`lab12/task3_fpgrowth_dbscan.py#L25-L135`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task3_fpgrowth_dbscan.py#L25-L135)

---

### 1.3 Direct Hashing & Pruning (DHP / Hash Apriori)
- **Role:** Reduces candidate 2-itemsets $C_2$ (the largest combinatorial bottleneck in Apriori) by early hash filtering during pass 1.
- **Mathematical Principle:**
  - A hash table $H_2$ of $B$ buckets is maintained during Pass 1.
  - For each transaction $t$, all $\binom{|t|}{2}$ 2-item subsets are hashed: $h(x, y) = (\text{id}_x \times 10 + \text{id}_y) \bmod B$. Bucket count $H_2[h(x, y)]$ is incremented.
  - At the end of Pass 1, if $H_2[h(x, y)] < \text{min\_count}$, the candidate pair $\{x, y\}$ cannot possibly be frequent and is pruned prior to generating $C_2$.
- **Complexity:**
  - Time: $O\left(|\mathcal{D}| \cdot \binom{|t|}{2}\right)$ hash updates in Pass 1, followed by significantly pruned candidate joins.
  - Space: $O(B)$ hash buckets (e.g., $B = 200,003$).
- **Original Source References:**
  - Implementation: [`Assignment_7_FIM_All_Files/assignment7_fim.py#L44-L100`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L44-L100)

---

### 1.4 Dynamic Itemset Counting (DIC)
- **Role:** Relaxes strict level-wise barrier synchronizations by adding candidate itemsets at periodic block checkpoints rather than only at full database pass boundaries.
- **Four Itemset States:**
  1. *Dashed Circle:* Suspected infrequent (count < minsup, not seen across full dataset).
  2. *Solid Circle:* Confirmed infrequent (count < minsup, seen across entire dataset).
  3. *Dashed Square:* Suspected frequent (count $\ge$ minsup, not seen across full dataset).
  4. *Solid Square:* Confirmed frequent (count $\ge$ minsup, seen across entire dataset).
- **Algorithmic Flow:**
  - Divide dataset into blocks of size $M$ (e.g., $M = 100$ transactions).
  - At every block boundary $k \cdot M$, evaluate counts of currently active candidates.
  - If any candidate achieves support threshold, promote it to frequent and generate its superset candidates immediately at the next block.
- **Original Source References:**
  - Implementation: [`Assignment_7_FIM_All_Files/assignment7_fim.py#L131-L200`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L131-L200)

---

### 1.5 Vertical ECLAT (Equivalence Class Transformation)
- **Role:** Vertical transaction representation algorithm replacing horizontal transaction scans with vertical Transaction ID (TID) set intersections.
- **Mathematical Principle:**
  - For each item $x$, construct TID set $\mathcal{T}(x) = \{\text{tid} \in \mathcal{D} \mid x \in t_{\text{tid}}\}$.
  - Support of itemset $X \cup Y$: $\mathcal{T}(X \cup Y) = \mathcal{T}(X) \cap \mathcal{T}(Y)$.
  - $|\mathcal{T}(X \cup Y)| \ge \text{minsup} \implies X \cup Y$ is frequent.
- **Complexity:**
  - Time: $O(|\mathcal{T}| \log |\mathcal{T}|)$ set intersection operations. No repeated scans of raw database.
  - Space: $O(|\text{Items}| \cdot |\mathcal{D}|)$ in memory.
- **Original Source References:**
  - Implementation: [`Assignment_7_FIM_All_Files/assignment7_fim.py#L103-L128`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/Assignment_7_FIM_All_Files/assignment7_fim.py#L103-L128)

---

### 1.6 Partitioning Variant of Apriori (SON Algorithm)
- **Role:** Guarantees frequent itemset discovery in exactly **2 database passes** regardless of maximum itemset length.
- **Mathematical Foundation:**
  - Theorem: If an itemset $X$ is globally frequent in $\mathcal{D}$ with support $s$, it must be locally frequent in at least one partition $P_i \subset \mathcal{D}$ with local support $s \cdot |P_i|$.
- **Two-Pass Protocol:**
  - **Pass 1 (Local Candidate Generation):** Partition $\mathcal{D}$ into $P$ non-overlapping slices: $\mathcal{D} = P_1 \cup P_2 \cup \dots \cup P_p$. Fit each slice entirely into RAM and mine local frequent itemsets $L^{(i)}$ at local threshold $s \cdot |P_i|$. Form global candidate set $C_{\text{global}} = \bigcup_{i=1}^p L^{(i)}$.
  - **Pass 2 (Global Validation):** Stream through entire dataset $\mathcal{D}$ in a single pass to count exact occurrences of each candidate in $C_{\text{global}}$. Filter candidates with global count $\ge s \cdot |\mathcal{D}|$.
- **Complexity:**
  - Database passes: Exactly 2.
  - Memory: $O(|P_i|)$ per partition, enabling linear horizontal scaling across distributed workers.
- **Original Source References:**
  - Implementation: [`lab8/fim_algorithms.py#L64-L131`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L64-L131)

---

### 1.7 AClose Algorithm (Frequent Closed Itemsets & Minimal Generators)
- **Role:** Eliminates redundancy in association rule mining by identifying the minimal lossless representation: Frequent Closed Itemsets (FCIs) and their Minimal Generators.
- **Definitions:**
  - *Galois Closure Operator:* For itemset $I$, closure $h(I) = \bigcap_{t \in \mathcal{T}(I)} t$. An itemset $I$ is closed if $h(I) = I$.
  - *Minimal Generator:* An itemset $g$ is a generator of closed set $c$ if $h(g) = c$ and no proper subset $g' \subset g$ has the same support: $\forall g' \subset g, \text{supp}(g') > \text{supp}(g)$.
- **Non-Generator Pruning Property:**
  - If a candidate $k$-itemset $c$ has any $(k-1)$-subset $s \subset c$ with $\text{supp}(s) = \text{supp}(c)$, then $c$ cannot be a minimal generator and is pruned immediately.
- **Original Source References:**
  - Implementation: [`lab8/fim_algorithms.py#L137-L228`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L137-L228)

---

### 1.8 Pincer Search Algorithm (Maximal Frequent Itemsets)
- **Role:** Bidirectional search mining Maximal Frequent Itemsets (MFIs) simultaneously from bottom-up (small itemsets) and top-down (large itemsets).
- **Core Concept:**
  - *MFI:* A frequent itemset that has no frequent supersets.
  - *MFCS (Maximum Frequent Candidate Set):* Top-down boundary representing potential maximal frequent itemsets. Initialized to $MFCS = \{\text{All frequent 1-items}\}$.
  - When an infrequent candidate $s \in C_k$ is identified during bottom-up search, any element in $MFCS$ containing $s$ must be split downward: $b' = b \setminus \{e\}$ for $e \in s$.
  - Any candidate $c \in C_k$ that is already a subset of a confirmed MFI is pruned from counting.
- **Original Source References:**
  - Implementation: [`lab8/fim_algorithms.py#L234-L363`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab8/fim_algorithms.py#L234-L363)

---

### 1.9 Distributed PySpark MLlib FP-Growth
- **Role:** Industrial distributed implementation running on Spark RDDs and DataFrames, parallelizing conditional FP-tree mining across cluster executor cores.
- **API Call:** `pyspark.ml.fpm.FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.5)`
- **Key Methods:** `.fit(df)`, `.freqItemsets`, `.associationRules`.
- **Original Source References:**
  - Implementation: [`lab12/pyspark_fpgrowth_dbscan_cluster.py#L14-L32`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/pyspark_fpgrowth_dbscan_cluster.py#L14-L32)

---

## 2. Supervised Machine Learning Algorithms

### 2.1 Classification and Regression Trees (CART)
- **Role:** Non-parametric recursive binary decision tree classifier.
- **Splitting Impurity Measures:**
  1. *Gini Impurity:*
     $$I_G(t) = 1 - \sum_{k=1}^C p_k^2$$
  2. *Shannon Entropy (Information Gain):*
     $$H(t) = -\sum_{k=1}^C p_k \log_2(p_k)$$
  3. *Log Loss (Cross-Entropy):*
     $$\mathcal{L}(t) = -\sum_{k=1}^C p_k \ln(p_k)$$
- **Split Optimization:** Select attribute $A$ and threshold $\theta$ that maximizes impurity reduction:
  $$\Delta I = I(\text{parent}) - \left(\frac{N_L}{N} I(\text{left}) + \frac{N_R}{N} I(\text{right})\right)$$
- **Original Source References:**
  - Training Script: [`lab 9/online_shoppers_decision_tree.py#L100-L240`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/online_shoppers_decision_tree.py#L100-L240)
  - Modular Pipeline: [`lab 9/src/train_decision_tree.py#L50-L160`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/src/train_decision_tree.py#L50-L160)

---

### 2.2 Minimal Cost-Complexity Pruning (CCP)
- **Role:** Post-pruning mechanism preventing overfitting without arbitrary heuristic depth cut-offs.
- **Mathematical Objective:**
  $$R_\alpha(T) = R(T) + \alpha |T|$$
  where $R(T)$ is total leaf misclassification error, $|T|$ is number of terminal leaves, and $\alpha \ge 0$ is the complexity tuning parameter.
- **Pruning Sequence:**
  - Compute effective alphas $\alpha_{\text{eff}} = \frac{R(t) - R(T_t)}{|T_t| - 1}$ for internal subtrees via `cost_complexity_pruning_path(X, y)`.
  - Evaluate pruned trees across $\alpha$ values via stratified cross-validation to select $\alpha^*$.
- **Original Source References:**
  - Implementation: [`lab 9/src/train_decision_tree.py#L140-L175`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/src/train_decision_tree.py#L140-L175)

---

### 2.3 Root-to-Leaf Decision Path Tracing
- **Role:** Explains individual black-box predictions by extracting the exact sequence of decision inequalities traversed from root to leaf.
- **Mechanism:** Leverages `tree_.decision_path(sample)` yielding sparse CSR indicator matrix of traversed node IDs.
- **Original Source References:**
  - Implementation: [`lab 9/src/train_decision_tree.py#L190-L240`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab%209/src/train_decision_tree.py#L190-L240)

---

### 2.4 Naive Bayes Classifiers
- **Role:** Probabilistic classification based on Bayes' Theorem with conditional feature independence assumption:
  $$P(y \mid x_1, \dots, x_d) \propto P(y) \prod_{i=1}^d P(x_i \mid y)$$

#### 2.4.1 Gaussian Naive Bayes (`GaussianNB`)
- Likelihood model:
  $$P(x_i \mid y) = \frac{1}{\sqrt{2\pi \sigma_{y,i}^2}} \exp\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2}\right)$$
- Numerical stabilization: Variance smoothing $\epsilon \cdot \max(\sigma^2)$ (default $\epsilon = 10^{-9}$).

#### 2.4.2 Bernoulli Naive Bayes (`BernoulliNB`)
- For binary features $x_i \in \{0, 1\}$:
  $$P(x_i \mid y) = P(i \mid y)^{x_i} (1 - P(i \mid y))^{1 - x_i}$$
- Additive Laplace Smoothing:
  $$\hat{P}(i \mid y) = \frac{N_{yi} + \alpha}{N_y + \alpha \cdot 2}$$

#### 2.4.3 Complement Naive Bayes (`ComplementNB`)
- Specifically formulated for severe class imbalance. Estimates model parameters using data from all classes *except* class $c$ (the complement class $\neg c$):
  $$\hat{\theta}_{\neg c, i} = \frac{\sum_{j \notin c} x_{ji} + \alpha}{\sum_{j \notin c} \sum_k x_{jk} + \alpha \cdot d}$$
- Decision rule assigns instance to class with minimal complement weight.
- **Original Source References:**
  - Imbalanced E-commerce: [`lab10/01_naive_bayes_online_shoppers.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/01_naive_bayes_online_shoppers.ipynb)
  - Multiclass Supply Chain: [`lab10/02_naive_bayes_supply_chain.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/02_naive_bayes_supply_chain.ipynb)

---

### 2.5 Multi-Classifier Evaluation Suite
- **Role:** Unified benchmark suite executing 6 standardized classifiers across continuous/discrete spaces:
  1. CART Decision Tree (`max_depth=5`)
  2. Gaussian Naive Bayes (`var_smoothing=1e-9`)
  3. K-Nearest Neighbors (`n_neighbors=5`, Euclidean metric)
  4. Support Vector Machine (`SVC(kernel='rbf', C=1.0, probability=True)`)
  5. Random Forest (`n_estimators=100, max_depth=8`)
  6. Logistic Regression (`penalty='l2', C=1.0, max_iter=1000`)
- **Evaluated Metrics:** Accuracy, Precision (macro), Recall (macro), F1-Score (macro), ROC-AUC (OvR), Training Time (ms), Latency (ms).
- **Original Source References:**
  - Implementation: [`lab12/task4_classification_clustering.py#L35-L125`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task4_classification_clustering.py#L35-L125)

---

## 3. Unsupervised Machine Learning & Clustering Algorithms

### 3.1 K-Means Clustering & The Elbow Method
- **Role:** Partition-based clustering minimizing within-cluster sum of squares (WCSS / Inertia):
  $$J = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|_2^2$$
- **Elbow Method:** Fits $K \in [2, 10]$, plotting $J(K)$. Optimal $K^*$ is the inflection point where second derivative of inertia is maximized.
- **Silhouette Coefficient:**
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  where $a(i)$ is mean intra-cluster distance, $b(i)$ is mean nearest-cluster distance. Range $[-1, +1]$.
- **Original Source References:**
  - Implementation: [`lab10/03_clustering_synthetic_dataset.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/03_clustering_synthetic_dataset.ipynb)

---

### 3.2 Hierarchical Agglomerative Clustering (HAC)
- **Role:** Bottom-up hierarchical clustering merging nearest pairs of clusters according to linkage criteria.
- **Linkage Functions:**
  - *Single Linkage:* $d(A, B) = \min_{x \in A, y \in B} d(x, y)$ (susceptible to chaining).
  - *Complete Linkage:* $d(A, B) = \max_{x \in A, y \in B} d(x, y)$ (produces compact spheres).
  - *Average Linkage:* $d(A, B) = \frac{1}{|A||B|} \sum_{x \in A} \sum_{y \in B} d(x, y)$.
  - *Ward Linkage:* Minimizes total within-cluster variance increase:
    $$\Delta \text{ESS}_{AB} = \frac{|A||B|}{|A| + |B|} \|\mu_A - \mu_B\|_2^2$$
- **Cophenetic Correlation Coefficient ($c$):**
  $$c = \frac{\sum_{i < j} (d_{ij} - \bar{d})(t_{ij} - \bar{t})}{\sqrt{\sum_{i<j}(d_{ij} - \bar{d})^2 \sum_{i<j}(t_{ij} - \bar{t})^2}}$$
  Measures preservation of pairwise Euclidean distances $d_{ij}$ in dendrogram cophenetic distances $t_{ij}$.
- **Original Source References:**
  - Implementation: [`lab10/03_clustering_synthetic_dataset.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/03_clustering_synthetic_dataset.ipynb)

---

### 3.3 Density-Based Spatial Clustering (DBSCAN) & Knee Detection
- **Role:** Discovers arbitrarily shaped clusters and isolates noise points based on local density reachability.
- **Definitions:**
  - $\varepsilon$-Neighborhood: $N_\varepsilon(p) = \{q \in D \mid \text{dist}(p, q) \le \varepsilon\}$.
  - Core Point: $|N_\varepsilon(p)| \ge \text{min\_samples}$.
  - Border Point: Not a core point, but $p \in N_\varepsilon(q)$ for some core point $q$.
  - Noise Point: Neither core nor border point (labeled $-1$).
- **Automated Parameter Selection Protocol:**
  1. Set $\text{min\_samples} = 2 \times \text{dimensions}$.
  2. Compute distance to the $k$-th nearest neighbor ($k = \text{min\_samples}$) for every point in dataset.
  3. Sort distances in ascending order to form the $k$-distance curve.
  4. Use `kneed.KneeLocator` to locate maximum curvature point $\varepsilon^*$.
- **Original Source References:**
  - Automated Knee DBSCAN: [`lab11/online_shoppers_clustering.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb)
  - Comparative Manifold Study: [`lab10/03_clustering_synthetic_dataset.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab10/03_clustering_synthetic_dataset.ipynb)

---

### 3.4 Distributed Spatial Partitioning DBSCAN with Disjoint Set Union (DSU)
- **Role:** Distributed DBSCAN overcoming $O(N^2)$ memory limits by spatial bounding-box partitioning and cross-boundary cluster merging.
- **Algorithmic Flow:**
  1. Decompose coordinate space into $P$ non-overlapping spatial bounding boxes with an $\varepsilon$-halo boundary margin.
  2. Run local DBSCAN independently within each partition in parallel worker processes.
  3. Reconcile boundary clusters: points located within distance $\le \varepsilon$ of partition borders that were assigned different local cluster labels are merged using a Disjoint Set Union (DSU / Union-Find) data structure.
  4. Relabel merged components to produce globally consistent cluster IDs.
- **Original Source References:**
  - Implementation: [`lab12/task3_fpgrowth_dbscan.py#L225-L355`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task3_fpgrowth_dbscan.py#L225-L355)

---

### 3.5 Gaussian Mixture Models (GMM)
- **Role:** Soft probabilistic density clustering modeling data as a mixture of $K$ multivariate Gaussian distributions:
  $$P(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
- **Optimization:** Expectation-Maximization (EM) algorithm iterating E-step (computing posterior responsibilities $\gamma_{ik}$) and M-step (updating $\pi_k, \mu_k, \Sigma_k$).
- **Model Selection:** Bayesian Information Criterion (BIC):
  $$\text{BIC} = -2 \ln \hat{L} + p \ln(n)$$
- **Original Source References:**
  - Implementation: [`lab11/online_shoppers_clustering.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb)

---

## 4. Evolutionary Algorithms & Metaheuristics

### 4.1 Simple Genetic Algorithm (SGA) for Continuous Optimization
- **Role:** Global metaheuristic search optimizing non-convex, multimodal continuous mathematical surfaces.
- **Test Surfaces Supported:**
  1. *Sphere 5D:* $f(x) = \sum_{i=1}^5 x_i^2$ (convex benchmark).
  2. *Rastrigin 5D:* $f(x) = 10 \cdot 5 + \sum_{i=1}^5 [x_i^2 - 10 \cos(2\pi x_i)]$ (highly multimodal).
  3. *Rosenbrock 5D:* $f(x) = \sum_{i=1}^4 [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]$ (narrow curved valley).
  4. *Custom Polynomial 5D:* Non-separable higher-order polynomial.
- **Genetic Architecture:**
  - Binary Chromosome: 16 bits per variable $\times$ 5 variables = 80 bits total length.
  - Decoding: $x_i = \text{low}_i + \frac{\text{integer\_value}}{2^{16} - 1} (\text{high}_i - \text{low}_i)$.
  - Fitness Mapping: Maximization objective $F(x) = \frac{1}{1.0 + f(x) - f_{\min}}$ or $-f(x)$.
  - Selection: Tournament Selection ($k=3$) and Roulette Wheel Selection.
  - Crossover: 1-Point, 2-Point, and Uniform Crossover ($p_c = 0.85$).
  - Mutation: Bit-flip mutation ($p_m = 0.02$).
  - Elitism: Elite count = 2 preserved unconditionally.
- **Original Source References:**
  - Implementation: [`lab12/task1_genetic_algorithm.py#L30-L240`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task1_genetic_algorithm.py#L30-L240)

---

### 4.2 Combinatorial Permutation Genetic Algorithm for TSP
- **Role:** Metaheuristic optimization for permutation sequence problems (Traveling Salesperson Problem).
- **Genetic Operators:**
  - Representation: Permutation vector of city indices $[c_0, c_1, \dots, c_{N-1}]$.
  - Fitness: $F = \frac{1000}{\text{Total Tour Distance}}$.
  - Specialized Crossovers:
    1. *Order Crossover (OX1):* Selects random segment from parent 1, copies into offspring; fills remaining positions from parent 2 starting from second cut point preserving cyclic order without duplicates.
    2. *Partially Mapped Crossover (PMX):* Selects random segment, builds bi-directional lookup mappings between segments, resolves gene conflicts outside segment.
  - Specialized Mutations:
    1. *2-Opt Inversion Mutation:* Reverses a random contiguous segment of the tour (untangles crossing path edges).
    2. *Swap Mutation:* Exchanges two arbitrary city positions.
- **Original Source References:**
  - Implementation: [`lab12/task2_tsp_ga.py#L35-L230`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab12/task2_tsp_ga.py#L35-L230)

---

## 5. Data Preprocessing, Transformations & Feature Engineering

### 5.1 Normalization & Scaling Formulations

| Method | Mathematical Formula | Output Range | Robust to Outliers? | Target Use Cases |
|---|---|---|---|---|
| **Min-Max** | $v' = \frac{v - \min(v)}{\max(v) - \min(v)} (new_{\max} - new_{\min}) + new_{\min}$ | $[0, 1]$ or $[\text{low}, \text{high}]$ | No | Neural networks, distance-based models with known bounded inputs |
| **Z-Score** | $z = \frac{v - \mu}{\sigma}$ | $\mu=0, \sigma=1$ ($(-\infty, +\infty)$) | Moderately | Algorithms assuming Gaussian distribution (Logistic Reg, SVM, PCA) |
| **Decimal Scaling** | $v' = \frac{v}{10^j}$ where $j = \lceil \log_{10}(\max \|v\|) \rceil$ | $(-1, 1)$ | No | Quick normalization preserving decimal representation without scale distortion |

- **Original Source References:**
  - Implementation: [`lab4/lab4_solution.py#L42-L138`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.py#L42-L138)

---

### 5.2 Equal-Frequency Binning & Smoothing
- **Bin Allocation:** Sort column $v_{(1)} \le v_{(2)} \le \dots \le v_{(N)}$. Partition into $B$ bins each having exactly $\lfloor N / B \rfloor$ elements.
- **Smoothing Modes:**
  1. *Bin Means:* Replace all values in bin $k$ with $\bar{v}_k = \frac{1}{|b_k|} \sum_{x \in b_k} x$.
  2. *Bin Medians:* Replace all values in bin $k$ with $\text{median}(b_k)$.
  3. *Bin Boundaries:* For each value $x \in b_k$, replace with $\arg\min_{e \in \{\min(b_k), \max(b_k)\}} |x - e|$.
- **Original Source References:**
  - Implementation: [`lab4/lab4_solution.py#L140-L215`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab4/lab4_solution.py#L140-L215)

---

### 5.3 Outlier Detection & Tukey's Fences Capping
- **Definition:**
  - $Q_1 = \text{25th percentile}, Q_3 = \text{75th percentile}$
  - $\text{IQR} = Q_3 - Q_1$
  - $\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}$
  - $\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$
- **Action:** Winsorization / Capping (clip values exceeding bounds to fence boundaries rather than dropping rows).
- **Original Source References:**
  - Implementation: [`LAB5/Northwind_Preprocessing_Final.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Final.ipynb)

---

### 5.4 Skewness Rectification ($\log(1+x)$)
- **Fisher-Pearson Standardized Skewness:**
  $$g_1 = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^3}{\left[\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2\right]^{3/2}}$$
- **Rule:** If $|g_1| > 1.0$, distribution is significantly skewed. Apply natural log transformation: $y = \ln(1 + x)$ (`np.log1p`).
- **Original Source References:**
  - Implementation: [`LAB5/Northwind_Preprocessing_Final.ipynb`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/LAB5/Northwind_Preprocessing_Final.ipynb)

---

## 6. The 13 Distance and Dissimilarity Metrics Framework

A formal metric on space $\mathcal{X}$ must satisfy four axioms for all $x, y, z \in \mathcal{X}$:
1. **Non-negativity:** $d(x, y) \ge 0$
2. **Identity of indiscernibles:** $d(x, y) = 0 \iff x = y$
3. **Symmetry:** $d(x, y) = d(y, x)$
4. **Triangle Inequality:** $d(x, z) \le d(x, y) + d(y, z)$

### 6.1 Numerical Metrics (8)

1. **Euclidean ($L_2$):**
   $$d_2(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$
2. **Manhattan ($L_1$, City Block):**
   $$d_1(x, y) = \sum_{i=1}^n |x_i - y_i|$$
   *Advantage:* Less sensitive to extreme coordinate outliers than $L_2$.
3. **Minkowski ($L_p$):**
   $$d_p(x, y) = \left(\sum_{i=1}^n |x_i - y_i|^p\right)^{1/p}$$
4. **Chebyshev ($L_\infty$, Chessboard):**
   $$d_\infty(x, y) = \max_{i=1 \dots n} |x_i - y_i|$$
5. **Mahalanobis Distance:**
   $$d_M(x, y) = \sqrt{(x - y)^T \Sigma^{-1} (x - y)}$$
   *Advantage:* Scale-invariant and accounts for inter-attribute covariance correlations.
6. **Cosine Distance:**
   $$d_{\cos}(x, y) = 1 - \frac{x \cdot y}{\|x\|_2 \|y\|_2}$$
   *Advantage:* Measures angular orientation rather than magnitude; standard for sparse text/item frequency vectors.
7. **Correlation Distance:**
   $$d_{\text{corr}}(x, y) = 1 - \frac{(x - \bar{x}) \cdot (y - \bar{y})}{\|x - \bar{x}\|_2 \|y - \bar{y}\|_2}$$
8. **Canberra Distance:**
   $$d_{\text{can}}(x, y) = \sum_{i=1}^n \frac{|x_i - y_i|}{|x_i| + |y_i|}$$
   *Advantage:* Weighted distance sensitive to relative differences near zero.

---

### 6.2 Categorical & Binary Metrics (4)

For two binary vectors $x, y \in \{0, 1\}^n$:
- $q = |\{i \mid x_i=1, y_i=1\}|$ (both 1)
- $r = |\{i \mid x_i=1, y_i=0\}|$ ($x$ is 1, $y$ is 0)
- $s = |\{i \mid x_i=0, y_i=1\}|$ ($x$ is 0, $y$ is 1)
- $t = |\{i \mid x_i=0, y_i=0\}|$ (both 0)

9. **Jaccard Distance:**
   $$d_J = 1 - \frac{q}{q + r + s}$$
   Ignores negative matches ($t$). Essential for market baskets where co-absence is uninformative.
10. **Dice / Sørensen-Dice Distance:**
    $$d_D = 1 - \frac{2q}{2q + r + s}$$
11. **Simple Matching Coefficient (SMC) Distance:**
    $$d_{\text{SMC}} = 1 - \frac{q + t}{q + r + s + t}$$
    Treats positive and negative co-occurrence with equal weight.
12. **Hamming Distance:**
    $$d_H(x, y) = \sum_{i=1}^n \mathbb{I}(x_i \ne y_i)$$

---

### 6.3 Mixed-Data Metric (1)

13. **Gower's General Dissimilarity Coefficient:**
    $$S_G(x, y) = \frac{\sum_{k=1}^p w_k s_{ijk}}{\sum_{k=1}^p w_k}$$
    where similarity $s_{ijk}$ is defined per attribute type:
    - Nominal/Binary: $s_{ijk} = 1$ if $x_{ik} = y_{jk}$, else $0$.
    - Numerical: $s_{ijk} = 1 - \frac{|x_{ik} - y_{jk}|}{R_k}$ (where $R_k$ is feature range).
    Dissimilarity: $d_G(x, y) = 1 - S_G(x, y)$.
- **Original Source References:**
  - Formal Taxonomy: [`lab11/online_shoppers_clustering.ipynb#Section-5`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/online_shoppers_clustering.ipynb)
  - Theoretical Report: [`lab11/Lab_11_Report_Version_2.pdf`](file:///c:/Users/Sreekar_Tirumala/Documents/College/Big%20Data/lab11/Lab_11_Report_Version_2.pdf)
