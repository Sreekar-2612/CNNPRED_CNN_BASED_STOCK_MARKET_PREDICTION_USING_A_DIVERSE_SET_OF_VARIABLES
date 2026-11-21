# PYSPARK AND BENCHMARK AUDIT

Items marked "verify at install" depend on the installed Spark build. I did not run Spark in this audit.

## 1. Native MLlib compatibility and algorithm equivalence

| Item | Gemini | Verdict | Severity |
|---|---|---|---|
| `pyspark.ml.fpm.FPGrowth` | yes | Native. Implements PFP (Li et al., 2008) | — |
| Spark FPGrowth `minSupport` semantics | "identical minSupport" | Spark keeps an itemset when **count ≥ ⌈minSupport·N⌉**. mlxtend compares the float support ≥ min_support. These can differ at exact boundaries. **Fix:** choose minsup so that minsup·N is not an integer, or compare by integer counts | MEDIUM |
| Spark association rules | "Spark rules match top local rules" | Spark generates rules with a **single-item consequent only**. mlxtend generates multi-item consequents. "Match" must be defined as: the itemset sets are equal exactly, and the rules with \|consequent\| = 1 are equal exactly (antecedent, consequent, confidence, lift to 1e-9). "Top rules match" is too weak — FP-Growth is deterministic, so demand **exact** equality | HIGH |
| Lift / support columns in Spark rules | assumed | `lift` exists in Spark ≥ 2.4. The rule `support` column exists in recent 3.x. **Verify at install.** Conviction is not provided; compute it in a post-processing step | LOW |
| Transactions | `array<string>` | Items must be unique per row, or Spark raises. Use sorted lists built from sets | LOW |
| `pyspark.ml.classification.NaiveBayes` | "Distributed NB" | The default `modelType='multinomial'` **rejects negative feature values**. Returns are negative half the time, so the default call fails. You **must** set `modelType='gaussian'` (Spark ≥ 3.0) to be equivalent to sklearn GaussianNB, or `'bernoulli'` with 0/1 features to match BernoulliNB | **CRITICAL** |
| Parameter equivalence NB | "accuracy within ±1% / ±2%" | Two different tolerances in two documents (roadmap ±1%, test plan ±2%). The models are mathematically the same (per-class MLE of μ, σ²). Differences come only from sklearn's `var_smoothing` (1e-9·max var) and Spark's `smoothing` (Laplace, which is irrelevant for gaussian). Expected agreement is ≈100% of predictions. **Correct test:** prediction agreement ≥ 99.5% and max \|Δ posterior\| < 1e-6, with var_smoothing set to 0 or tiny in sklearn | MEDIUM |
| Pipeline | VectorAssembler | Correct. Spark has no RobustScaler (use `RobustScaler` in Spark ≥ 3.0 or `StandardScaler`). For GaussianNB, scaling is irrelevant | LOW |
| "Distributed" | `local[*]` | `local[*]` is **multi-threaded on one machine**, not a cluster. Call it "parallel execution on Spark local mode with P cores". Make no claims about cluster scaling | MEDIUM |

## 2. Benchmark methodology

### B-1 · HIGH · Startup overhead isolation is under-specified
Time each phase separately:
1. JVM + SparkSession creation.
2. pandas → Spark DataFrame conversion (enable Arrow and log it).
3. First `fit` (includes JIT and codegen warm-up).
4. Steady-state `fit`, as the median of ≥ 5 runs after 1–2 warm-up runs.
5. Result materialisation (`.count()` / `.collect()`). Spark is lazy: if the output is never materialised, the timing is invalid.

- Report end-to-end time **and** algorithm-only time.
- Local baseline: the same phases where applicable (the mlxtend one-hot encoding counts as data preparation).
- Pin `spark.sql.shuffle.partitions` and the number of cores. Use `local[1]`, `local[2]`, `local[4]`, `local[P]` to get a strong-scaling curve. This is the defensible "performance enhancement from Big Data support".

### B-2 · CRITICAL · Synthetic scaling by pure replication is invalid for FP-Growth
- Replicating 1,984 transactions k times leaves **every relative support unchanged**, so the frequent itemsets are identical.
- An FP-tree merges identical paths, so tree size stays almost constant and only the counts grow. The benchmark would measure I/O and counting, not mining. It also flatters local FP-Growth.
- **Correction:**
  - Generate synthetic transactions by **bootstrap resampling with per-attribute perturbation**. For example, resample day-level records, add Gaussian noise (σ = 0.25·σ_train) to the underlying continuous variables, then re-discretise with the frozen train thresholds. Paths then diversify while the marginal distributions are preserved.
  - Alternatively, use an IBM Quest-style generator.
  - Report the number of distinct transactions at each N, and the number of frequent itemsets per N.
- **NB:** Use the same bootstrap + noise approach on the feature matrix. For classification, plain replication is acceptable because the NB cost is linear in N regardless.

### B-3 · HIGH · Crossover "N ≈ 50k–100k" stated before measurement
- No such number may appear in the plan. The crossover depends on cores, memory, Arrow use, and whether Spark start-up time is included.
- It may **not exist** for NB. sklearn GaussianNB is a vectorised O(N·d) pass and remains faster than Spark local up to tens of millions of rows on a laptop.
- It may exist for mlxtend FP-Growth, because of mlxtend's dense one-hot DataFrame memory and its single-threaded Python tree building.
- **Report the measured curve, including the possibility that no crossover occurs.**

### B-4 · HIGH · Memory measurement
- `tracemalloc` measures only Python-heap allocations in the driver process. It **does not see** NumPy buffers allocated outside the tracked allocator, the Arrow buffers, or the JVM heap.
- **Correction:**
  - For local runs, use peak RSS via `psutil` (sampled in a thread) or `resource`/`memory_profiler`.
  - For Spark, sum the RSS of the `java` child-process tree, plus Spark's own metrics: `spark.sparkContext.statusTracker()`, or `/api/v1/applications/.../executors` → `peakMemoryMetrics` on Spark ≥ 3.0 (this needs the UI or REST enabled; the plan disables the UI, so either re-enable it or rely on psutil).
  - Always report the JVM's configured `spark.driver.memory` cap.

### B-5 · MEDIUM · Amdahl's law is misapplied
- The stated model, T_Spark = T_init + T_DAG + cN/P versus T_local = c_local·N, is a **fixed-overhead linear cost model**, not Amdahl's law.
- Amdahl's law is S(P) = 1 / ((1−f) + f/P) at a fixed workload. Use it with the `local[P]` sweep: fit f from the measured speedups.
- Varying N is weak or Gustafson scaling. Name both correctly.

### B-6 · MEDIUM · Statistical rigour
- Use ≥ 5 repetitions, report median and IQR, fix the seeds, and run with no other heavy processes.
- Report the hardware: CPU model, number of cores, RAM, Java version, Spark version.

### B-7 · MEDIUM · Unsupported numbers
| Claim | Where | Verdict |
|---|---|---|
| JVM startup ~2–5 s vs local ~20 ms | MVP §3 | Plausible order of magnitude but unmeasured; mark as hypothesis |
| "mlxtend crashed with MemoryError at 10⁶; Spark 42 s" | MASTER_BRAIN (Lab 12) | A different dataset and machine; must not be cited as evidence here |
| "crossover typically 50k–100k" | MVP §3 | Delete |
| "Spark NB within ±1% / ±2%" | Roadmap / TEST plan | Inconsistent; replace (see table) |

## 3. Environment risks (Windows)
- **HADOOP_HOME mock:** In-memory MLlib usually works without `winutils.exe`. **Any file write** (`model.save`, `df.write.parquet`) fails without it. Either write outputs via pandas (`toPandas()`), or install a matching winutils. The roadmap's "write Spark tables to CSV" must use the pandas path.
- **Pinned versions (verify at install, do not trust the pins):**
  - `matplotlib==3.11.2`, `pyarrow==25.0.1`, `numpy==2.2.6` + `pyspark==3.5.3`: some of these may not exist as listed, or may be incompatible.
  - pyspark 3.5.x officially supports Python 3.8–3.11/3.12 and NumPy 1.x for the pandas-on-Spark paths; NumPy 2 support arrived in Spark 4.0.
  - **Recommendation:** pin the Python version, resolve the pins with `pip install` in a clean venv, then freeze. If pyspark 3.5 + NumPy 2 errors appear, use `numpy<2`.
- `mlxtend` is **not installed** in the current interpreter [V], so it needs installing.
- Hard-coded `JAVA_HOME` path (`jdk-17.0.21.9`) is unverified. Detect it at runtime instead.

## 4. Choice of the two PySpark techniques
FP-Growth + Naive Bayes is a correct and native choice. Do **not** use DBSCAN as the second PySpark technique: Spark MLlib has no DBSCAN, so a hand-rolled version would not be a "native" test-drive.
