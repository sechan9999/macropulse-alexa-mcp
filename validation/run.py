"""
validation/run.py — run the pre-registered historical validation (Stages 1-6).

  python -m validation.run fetch            # download every source and pin it in the manifest
  python -m validation.run stage1 ... stage6
  python -m validation.run all [--perm N]   # every stage, then the report

Downloads and the heavy stages run in GitHub Actions (.github/workflows/validation.yml); results land in
validation/results/ and docs/validation/report.md.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", help="fetch, stage1..stage6, report, all")
    ap.add_argument("--perm", type=int, default=1000, help="permutations for the Stage 2 permutation test")
    a = ap.parse_args(argv)
    todo = ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "report"] if "all" in a.what else a.what
    failed = []
    for step in todo:
        t0 = time.time()
        print(f"── {step}", flush=True)
        try:
            if step == "fetch":
                from validation import data
                data.fetch_all()
            elif step == "report":
                from validation import report
                report.write()
            else:
                mod = __import__(f"validation.{step}", fromlist=["run"])
                mod.run(n_perm=a.perm) if step == "stage2" else mod.run()
            print(f"   done in {time.time() - t0:.0f}s", flush=True)
            from validation.results import RESULTS
            (RESULTS / step / "error.json").unlink(missing_ok=True)      # a stale failure from an earlier run
        except Exception:
            traceback.print_exc()
            failed.append(step)
            from validation.results import save_json
            save_json(step, "error", {"traceback": traceback.format_exc()[-4000:]})
    if failed:
        print("FAILED:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
