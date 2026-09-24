"""
run_all.py – Execute the full Fake News Detection pipeline.

Runs Phases 2–7 in order:
  Phase 2: EDA
  Phase 3: Cleaning (naive + leak_free)
  Phase 4: Shortcut test
  Phase 5: Training & evaluation
  Phase 6: Explainability
  Phase 7: Word clouds
  Extra:   Style stress test (Reuters style markers removed)
"""

import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

PHASES = [
    ("Phase 2: EDA",              os.path.join(SRC, "eda.py")),
    ("Phase 3: Cleaning",         os.path.join(SRC, "clean.py")),
    ("Phase 4: Shortcut Test",    os.path.join(SRC, "shortcut_test.py")),
    ("Phase 5: Training",         os.path.join(SRC, "train.py")),
    ("Phase 6: Explainability",   os.path.join(SRC, "explain.py")),
    ("Phase 7: Word Clouds",      os.path.join(SRC, "wordclouds.py")),
    ("Extra: Style Stress Test",  os.path.join(SRC, "style_stress_test.py")),
]


def main():
    print("=" * 60)
    print("  FAKE NEWS DETECTION – Full Pipeline")
    print("=" * 60)
    start = time.time()

    for name, script in PHASES:
        print(f"\n{'━' * 60}")
        print(f"  {name}")
        print(f"{'━' * 60}")
        t0 = time.time()
        sys.stdout.flush()  # keep headers in order with the child process output
        result = subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            check=False,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"\n❌ {name} FAILED (exit code {result.returncode})")
            sys.exit(1)
        print(f"\n✅ {name} completed in {elapsed:.1f}s")

    total = time.time() - start
    print(f"\n{'━' * 60}")
    print(f"  Pipeline complete in {total:.1f}s")
    print(f"{'━' * 60}")


if __name__ == "__main__":
    main()
