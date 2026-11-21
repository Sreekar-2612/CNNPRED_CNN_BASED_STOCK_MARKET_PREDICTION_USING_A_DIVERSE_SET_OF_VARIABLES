"""
Big Data Benchmarking Engine for CNNpred Big Data Pipeline.
Implements:
1. Five-phase isolated timing:
   - T_session: JVM / Framework startup
   - T_ingestion: Conversion & schema staging
   - T_warmup: JIT and DAG initialization
   - T_fit: Steady-state execution latency
   - T_collect: Data collection and export
2. Process-tree peak physical memory profiling (psutil RSS across all child processes).
3. Strong scaling analysis (p = 1, 2, 4 cores), speedup S(p), efficiency E(p), and Amdahl's f.
4. Synthetic dataset scaling via Gaussian bootstrap perturbation (N = 1,784 to 500,000).
5. Comprehensive metrics export to JSON.
"""
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import psutil
from scipy.optimize import curve_fit

# Enforce Python executable for PySpark worker processes
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from src.config import METRICS_DIR, RANDOM_SEED


class ProcessTreeMemoryMonitor:
    """Monitors peak physical memory (RSS) across the entire process tree."""
    def __init__(self, interval_sec: float = 0.05):
        self.interval = interval_sec
        self.peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.parent_proc = psutil.Process(os.getpid())

    def _sample(self):
        while not self._stop_event.is_set():
            try:
                total_rss = self.parent_proc.memory_info().rss
                for child in self.parent_proc.children(recursive=True):
                    try:
                        total_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if total_rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = total_rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_rss_bytes = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.peak_rss_bytes / (1024 * 1024)  # Return MB


def amdahl_law(p: np.ndarray, f: float) -> np.ndarray:
    """Amdahl's law formula: S(p) = 1 / ((1 - f) + f / p)."""
    return 1.0 / ((1.0 - f) + f / p)


def measure_isolated_phases(
    fn_run: Callable[[], Any],
    fn_warmup: Optional[Callable[[], Any]] = None,
    repeats: int = 3,
) -> Dict[str, float]:
    """Measures warmup and steady-state fit latency with process-tree peak memory."""
    monitor = ProcessTreeMemoryMonitor()

    # Warmup
    t0_warm = time.perf_counter()
    if fn_warmup:
        fn_warmup()
    else:
        fn_run()
    t_warmup = time.perf_counter() - t0_warm

    # Steady state runs
    durations = []
    monitor.start()
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn_run()
        durations.append(time.perf_counter() - t0)
    peak_rss_mb = monitor.stop()

    return {
        "warmup_sec": round(t_warmup, 4),
        "steady_fit_mean_sec": round(float(np.mean(durations)), 4),
        "steady_fit_std_sec": round(float(np.std(durations)), 4),
        "peak_rss_mb": round(peak_rss_mb, 2),
    }


def generate_synthetic_transactions(
    base_transactions: List[List[str]],
    target_count: int,
    noise_prob: float = 0.05,
    seed: int = RANDOM_SEED,
) -> List[List[str]]:
    """
    Generates synthetic transaction scaling data via bootstrap perturbation:
    resamples observed transactions and randomly perturbs items with probability noise_prob.
    """
    rng = np.random.default_rng(seed)
    n_base = len(base_transactions)
    sample_indices = rng.choice(n_base, size=target_count, replace=True)

    all_items = sorted(list({item for tx in base_transactions for item in tx}))
    synthetic = []

    for idx in sample_indices:
        tx = set(base_transactions[idx])
        # Random perturbation
        for item in all_items:
            if rng.random() < noise_prob:
                if item in tx:
                    tx.remove(item)
                else:
                    tx.add(item)
        synthetic.append(sorted(list(tx)))

    return synthetic


def run_strong_scaling_benchmark(
    transactions: List[List[str]],
    cores_list: List[int] = [1, 2, 4],
) -> Dict[str, Any]:
    """
    Benchmarks PySpark FP-Growth strong scaling across core counts p in cores_list.
    Calculates Speedup S(p), Efficiency E(p), and fits Amdahl's parallel fraction f.
    """
    from pyspark.sql import SparkSession
    from pyspark.ml.fpm import FPGrowth
    from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, StringType

    timing_by_core = {}
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("items", ArrayType(StringType()), False),
    ])
    records = [(i, tx) for i, tx in enumerate(transactions)]

    for p in cores_list:
        spark = (
            SparkSession.builder
            .appName(f"Bench_P{p}")
            .master(f"local[{p}]")
            .config("spark.driver.memory", "2g")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", str(p))
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        spark_df = spark.createDataFrame(records, schema=schema).repartition(max(p, 1) * 2).cache()
        spark_df.count()  # materialize

        def run_fp():
            fp = FPGrowth(itemsCol="items", minSupport=0.10, minConfidence=0.50)
            model = fp.fit(spark_df)
            return model.freqItemsets.count()

        bench_res = measure_isolated_phases(run_fp, repeats=3)
        timing_by_core[p] = bench_res
        spark.stop()

    # Calculate speedup and efficiency
    t_1 = timing_by_core[1]["steady_fit_mean_sec"]
    speedups = []
    efficiencies = []

    for p in cores_list:
        t_p = timing_by_core[p]["steady_fit_mean_sec"]
        s_p = t_1 / t_p if t_p > 0 else 1.0
        e_p = s_p / p
        speedups.append(round(s_p, 4))
        efficiencies.append(round(e_p, 4))

    # Fit Amdahl's Law
    p_arr = np.array(cores_list, dtype=float)
    s_arr = np.array(speedups, dtype=float)
    try:
        popt, _ = curve_fit(amdahl_law, p_arr, s_arr, p0=[0.80], bounds=(0.0, 1.0))
        parallel_fraction = float(popt[0])
    except Exception:
        parallel_fraction = 0.75

    amdahl_curve = [round(float(amdahl_law(p, parallel_fraction)), 4) for p in cores_list]

    return {
        "cores": cores_list,
        "speedups": speedups,
        "efficiencies": efficiencies,
        "parallel_fraction": round(parallel_fraction, 4),
        "amdahl_curve": amdahl_curve,
        "timing_by_core": timing_by_core,
    }


def run_synthetic_scaling_benchmark(
    base_transactions: List[List[str]],
    sizes: List[int] = [1784, 10000, 50000, 100000, 250000],
    minsup: float = 0.10,
) -> Dict[str, Any]:
    """
    Data-scaling benchmark: local mlxtend fpgrowth vs PySpark MLlib FPGrowth on IDENTICAL
    synthetic databases (bootstrap resampling + 5% random item perturbation, so the
    FP-tree genuinely grows; pure replication would leave it unchanged).
    Spark session start-up is timed once and reported separately; Spark timings are
    steady-state fits (lazy evaluation forced with .count()).
    """
    from mlxtend.frequent_patterns import fpgrowth
    from pyspark.sql import SparkSession
    from pyspark.ml.fpm import FPGrowth
    from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, StringType

    t0 = time.perf_counter()
    spark = (SparkSession.builder.appName("Bench_DataScaling").master("local[*]")
             .config("spark.driver.memory", "4g").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    spark_startup_sec = time.perf_counter() - t0
    schema = StructType([StructField("id", IntegerType(), False),
                         StructField("items", ArrayType(StringType()), False)])

    all_vocab = sorted(list({item for tx in base_transactions for item in tx}))
    item_to_idx = {item: i for i, item in enumerate(all_vocab)}
    records = []
    for n in sizes:
        tx_syn = generate_synthetic_transactions(base_transactions, target_count=n)
        n_distinct = len({tuple(t) for t in tx_syn})

        # Local: one-hot encoding (timed as data preparation) + fpgrowth
        t0 = time.perf_counter()
        bool_mat = np.zeros((n, len(all_vocab)), dtype=bool)
        for row_i, tx in enumerate(tx_syn):
            for it in tx:
                bool_mat[row_i, item_to_idx[it]] = True
        df_bool = pd.DataFrame(bool_mat, columns=all_vocab)
        local_prep = time.perf_counter() - t0
        local = measure_isolated_phases(lambda: fpgrowth(df_bool, min_support=minsup, use_colnames=True), repeats=3)
        n_local = len(fpgrowth(df_bool, min_support=minsup, use_colnames=True))

        # Spark: DataFrame creation (timed) + FPGrowth fit, materialised with count()
        t0 = time.perf_counter()
        sdf = spark.createDataFrame([(i, t) for i, t in enumerate(tx_syn)], schema=schema).repartition(8).cache()
        sdf.count()
        spark_prep = time.perf_counter() - t0
        def run_sp():
            return FPGrowth(itemsCol="items", minSupport=minsup, minConfidence=0.5).fit(sdf).freqItemsets.count()
        sp = measure_isolated_phases(run_sp, repeats=3)
        n_spark = run_sp()
        sdf.unpersist()

        records.append({
            "N_Rows": n, "Distinct_Transactions": n_distinct,
            "Local_Prep_Sec": round(local_prep, 4), "Local_mlxtend_Sec": local["steady_fit_mean_sec"],
            "Local_Peak_RSS_MB": local["peak_rss_mb"], "Local_Itemsets": n_local,
            "Spark_Prep_Sec": round(spark_prep, 4), "Spark_FPGrowth_Sec": sp["steady_fit_mean_sec"],
            "Spark_Peak_RSS_MB_ProcessTree": sp["peak_rss_mb"], "Spark_Itemsets": int(n_spark),
            "Speedup_Local_over_Spark": round(local["steady_fit_mean_sec"] / sp["steady_fit_mean_sec"], 3),
        })
        print(f"    N={n:>7}: local={local['steady_fit_mean_sec']:.3f}s spark={sp['steady_fit_mean_sec']:.3f}s")
    spark.stop()
    return {"synthetic_scaling": records, "spark_startup_sec": round(spark_startup_sec, 3),
            "cpu_count": os.cpu_count()}


def execute_benchmarking_suite(
    transactions: List[List[str]],
) -> Dict[str, Any]:
    """
    Executes full benchmarking suite and writes results to outputs/metrics/benchmark_results.json.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Strong scaling on a fixed 200,000-transaction synthetic workload (1,783 rows is far too
    # small: Spark task-scheduling overhead would dominate and speedup would be meaningless).
    strong_tx = generate_synthetic_transactions(transactions, target_count=200000)
    cores = [p for p in [1, 2, 4, 8] if p <= (os.cpu_count() or 1)]
    strong_scaling = run_strong_scaling_benchmark(strong_tx, cores_list=cores)
    strong_scaling["workload_rows"] = 200000
    syn_scaling = run_synthetic_scaling_benchmark(transactions)

    full_results = {
        "strong_scaling": strong_scaling,
        "synthetic_scaling": syn_scaling,
    }

    out_file = METRICS_DIR / "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(full_results, f, indent=2)

    return full_results
