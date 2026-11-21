# PROGRESS

**Status (2026-11-21):** Final independent verification complete. Pipeline re-run, 11 notebooks re-executed, 95/95 tests pass, PDF rebuilt (20 chapters, 36 pages).

The previous version of this file contained figures that are no longer valid (279 MFIs, 26 tests, Amdahl f = 0.6718, NB 50.98%/54.06%, Spark 1,897 itemsets). They were produced by buggy transaction code and an inconsistent Spark comparison.

Current results and the list of corrections: see `VERIFICATION_REPORT.md`.

Open limitation: PySpark showed no speed-up over local Python on one machine; this is reported as measured.
