"""
Master Autonomous Execution Script for CNNpred Big Data Practical Hackathon.
Institution: IIITDM Kancheepuram
Course: Analytics and Systems of Big Data Practice — Endsem
"""
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.run_pipeline import run_full_pipeline
from src.build_notebooks import build_all_notebooks


def main():
    print("=" * 80)
    print("MASTER EXECUTION: ANALYTICS & SYSTEMS OF BIG DATA PRACTICE (ENDSEM)")
    print("IIITDM Kancheepuram — Automated Academic Pipeline")
    print("=" * 80)

    # 1. Run Complete Analytical Pipeline
    print("\n[STEP 1/3] Executing Core Analytical Pipeline...")
    run_full_pipeline()

    # 2. Build and Execute Jupyter Notebooks
    print("\n[STEP 2/3] Building and Executing 11 Jupyter Notebooks...")
    build_all_notebooks()

    # 3. Run Pytest Test Suite
    print("\n[STEP 3/3] Running Formal Acceptance Criteria Test Suite (pytest)...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR: Test suite failed!")
        sys.exit(res.returncode)
    else:
        print("SUCCESS: All 26 formal acceptance tests passed!")

    print("\n" + "=" * 80)
    print("MASTER EXECUTION COMPLETED SUCCESSFULLY!")
    print("Outputs available in:")
    print("  - reports/Analytics_BigData_Endsem_Report.pdf (Academic Monograph)")
    print("  - notebooks/ (11 Executed Jupyter Notebooks)")
    print("  - outputs/figures/ (12 Publication-Quality Figures at 300 DPI)")
    print("  - outputs/tables/ (18 Statistical Verification Tables)")
    print("  - outputs/metrics/ (Big Data Benchmarking Metrics)")
    print("=" * 80)


if __name__ == "__main__":
    main()
