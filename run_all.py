"""
Run the whole ACCAITICS AI pipeline and print the measured results.

    python run_all.py

Regenerates the data, trains both models, evaluates them, runs the test suite,
and writes every result to outputs/. Everything is seeded, so the numbers
reproduce exactly.

To launch the application afterwards:

    streamlit run app.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Step 1 — Generate the accounting data layer", [sys.executable, "-m", "accaitics.generate"]),
    ("Step 2 — Expense categorisation (rules vs ML)", [sys.executable, "-m", "accaitics.categorise"]),
    ("Step 3 — Anomaly detection (rules + Isolation Forest)", [sys.executable, "-m", "accaitics.anomaly"]),
    ("Step 4 — Ask Finance demonstration", [sys.executable, "-m", "accaitics.ask"]),
    ("Step 5 — Test suite", [sys.executable, "-m", "pytest", "tests/", "-q"]),
]


def main() -> int:
    print("=" * 74)
    print("ACCAITICS AI — FULL PIPELINE")
    print("=" * 74)

    failed = []
    for i, (label, cmd) in enumerate(STEPS, start=1):
        print(f"\n[{i}/{len(STEPS)}] {label}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"    done in {dt:.1f}s")
            tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-3:]
            for ln in tail:
                print(f"    | {ln}")
        else:
            print(f"    FAILED after {dt:.1f}s")
            print(r.stdout[-1200:])
            print(r.stderr[-1200:])
            failed.append(label)

    print("\n" + "=" * 74)
    if failed:
        print(f"COMPLETED WITH {len(failed)} FAILURE(S)")
        return 1
    print("ALL STEPS PASSED")
    print("=" * 74)
    print("\nOutputs written to outputs/:")
    for f in sorted((ROOT / "outputs").glob("*")):
        print(f"  {f.name}")
    print("\nLaunch the app:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
