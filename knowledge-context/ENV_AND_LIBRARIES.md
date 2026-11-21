# 📦 BIG DATA CURRICULUM: ENV_AND_LIBRARIES.md
## Runtime Environments, Library Versions, Dependencies & Setup Commands

This document specifies the exact environment configurations, package versions, and installation protocols required to run every algorithm, script, and notebook from Labs 1 through 12, Task 2, and the Semester Big Data Project.

---

## 🚀 One-Line Master Installation Command

To set up a fresh environment with all dependencies for the entire curriculum in a single command, execute:

```powershell
pip install numpy==2.2.6 scipy==1.15.3 pandas==2.3.3 scikit-learn==1.7.2 matplotlib==3.11.2 seaborn==0.13.2 mlxtend==0.23.1 kneed==0.8.5 pyspark==3.5.3 pyarrow==25.0.1 openpyxl==3.1.5 python-docx==1.2.0 joblib==1.6.0
```

---

## 📋 Comprehensive Dependency Manifest

| Library | Version Target | Primary Lab Usage | Core Purpose in Curriculum |
|---|---|---|---|
| **`numpy`** | `2.2.6` | Labs 4, 9, 10, 11, 12 | High-performance array operations, matrix mathematics, random number seeds |
| **`pandas`** | `2.3.3` | Labs 1–5, 9–11 | Tabular DataFrame manipulation, missingness audits, SQL data ingest |
| **`scipy`** | `1.15.3` | Labs 4, 5, 10, 11 | Statistical tests, skewness calculations (`scipy.stats.skew`), spatial distance metrics (`scipy.spatial.distance.cdist`), dendrogram linkages (`scipy.cluster.hierarchy`) |
| **`scikit-learn`** | `1.7.2` | Labs 9, 10, 11, 12 | Decision Trees (CART), CCP pruning, Naive Bayes (`GaussianNB`, `BernoulliNB`, `ComplementNB`), K-Means, DBSCAN, HAC, GMM, Model metrics (`f1_score`, `roc_auc_score`, `silhouette_score`) |
| **`matplotlib`** | `3.11.2` | Labs 3, 4, 6, 9–12 | Diagnostic plots, ROC curves, dendrograms, 2D PCA scatter plots, GA fitness trajectories |
| **`seaborn`** | `0.13.2` | Labs 3, 5, 10, 11 | Statistical heatmaps, correlation matrices, violin plots, pairplots |
| **`mlxtend`** | `0.23.1` | Labs 6, 7 | Frequent pattern mining benchmarks (`mlxtend.frequent_patterns.apriori`, `mlxtend.frequent_patterns.fpgrowth`, `association_rules`) |
| **`kneed`** | `0.8.5` | Lab 11 | Automated knee/elbow point detection (`kneed.KneeLocator`) on $k$-distance curves to set optimal DBSCAN $\varepsilon$ |
| **`pyspark`** | `3.5.3` | Lab 12, Project | Distributed Big Data engine, DataFrame API, Spark MLlib (`FPGrowth`), Structured Streaming |
| **`py4j`** | `0.10.9.7` | Lab 12, Project | Python-to-JVM communication bridge required by PySpark |
| **`pyarrow`** | `25.0.1` | Project | High-velocity Parquet serialization and PySpark vectorization |
| **`openpyxl`** | `3.1.5` | Labs 1–4, Task 2 | Native reading and writing of Excel (`.xlsx`) multi-sheet workbooks |
| **`python-docx`** | `1.2.0` | Labs 5, 8, 10, 11, 12 | Automated Word report document generation |
| **`joblib`** | `1.6.0` | Lab 9 | Model serialization, parallel grid search execution |

---

## 💻 Windows Environment Configuration Guide

### 1. Dedicated Virtual Environment Setup (PowerShell)
Run from the workspace directory:

```powershell
# Create an isolated virtual environment
python -m venv .venv

# Set execution policy if script activation is restricted
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade core packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install all curriculum requirements
pip install -r requirements.txt
```

---

### 2. PySpark & Java Runtime Prerequisites (Windows)

Apache Spark relies on the Java Virtual Machine (JVM). Running PySpark on Windows requires:

1. **Java Development Kit (JDK 11 or 17):**
   - Download and install Eclipse Temurin OpenJDK 11 or 17.
   - Set environment variable:
     ```powershell
     [System.Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Eclipse Adoptium\jdk-17.x.x-hotspot", "User")
     ```

2. **Windows Native Hadoop Binaries (`winutils.exe`):**
   - Windows requires `winutils.exe` and `hadoop.dll` for native file-system operations.
   - Download `hadoop-3.0.0` winutils from trusted repositories and place in `C:\hadoop\bin`.
   - Set environment variables:
     ```powershell
     [System.Environment]::SetEnvironmentVariable("HADOOP_HOME", "C:\hadoop", "User")
     [System.Environment]::SetEnvironmentVariable("PATH", "$env:PATH;C:\hadoop\bin", "User")
     ```

3. **Memory Allocation Flag for Local Spark:**
   - Always allocate adequate driver memory when initiating local Spark sessions:
     ```python
     from pyspark.sql import SparkSession
     spark = SparkSession.builder \
         .appName("BigDataLab") \
         .master("local[*]") \
         .config("spark.driver.memory", "4g") \
         .getOrCreate()
     ```

---

## 🧪 3-Second Environment Verification Script

Create and run this script to ensure all critical libraries and engines are operational before beginning work:

```python
import sys

def verify_environment():
    checks = {
        "NumPy": lambda: __import__('numpy').__version__,
        "Pandas": lambda: __import__('pandas').__version__,
        "SciPy": lambda: __import__('scipy').__version__,
        "Scikit-Learn": lambda: __import__('sklearn').__version__,
        "Matplotlib": lambda: __import__('matplotlib').__version__,
        "Seaborn": lambda: __import__('seaborn').__version__,
        "MLxtend": lambda: __import__('mlxtend').__version__,
        "Kneed": lambda: __import__('kneed').__version__,
        "OpenPyXL": lambda: __import__('openpyxl').__version__,
        "Python-Docx": lambda: __import__('docx').__version__,
    }
    
    print("=" * 60)
    print("ENVIRONMENT HEALTH CHECK")
    print(f"Python Runtime: {sys.version.split()[0]} ({sys.executable})")
    print("=" * 60)
    
    all_passed = True
    for name, check_fn in checks.items():
        try:
            ver = check_fn()
            print(f"  [PASS] {name:15s} -> v{ver}")
        except Exception as e:
            print(f"  [FAIL] {name:15s} -> ERROR: {e}")
            all_passed = False
            
    # Test PySpark separately
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("Test").master("local[1]").getOrCreate()
        spark.stop()
        print(f"  [PASS] {'PySpark':15s} -> JVM Session Active")
    except Exception as e:
        print(f"  [WARN] {'PySpark':15s} -> PySpark/Java not configured ({e})")
        
    print("=" * 60)
    if all_passed:
        print("ALL ESSENTIAL DATA SCIENCE LIBRARIES ARE OPERATIONAL.")
    else:
        print("SOME PACKAGES FAILED. RUN THE MASTER INSTALL COMMAND.")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()
```
