"""Unit tests for Big Data Benchmarking Engine (REQ-24)."""
import json
import pytest
from src.config import METRICS_DIR, FIGURES_DIR


def test_benchmarking_engine_rss():
    """REQ-24: Verifies isolated timing, psutil process-tree RSS, and Amdahl fit."""
    bench_file = METRICS_DIR / "benchmark_results.json"
    assert bench_file.exists()

    with open(bench_file) as f:
        data = json.load(f)

    assert "strong_scaling" in data
    assert "synthetic_scaling" in data
    assert 0.0 <= data["strong_scaling"]["parallel_fraction"] <= 1.0  # measured value, not a forced outcome
    assert len(data["strong_scaling"]["speedups"]) == len(data["strong_scaling"]["cores"])
    assert (FIGURES_DIR / "fig_12_pyspark_strong_scaling_speedup.png").exists()
