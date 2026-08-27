"""
Gather results.json from every experiment directory into one table.

Prints markdown ready to paste into the dissertation and writes a CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--out", type=Path, default=Path("outputs/results.csv"))
    args = parser.parse_args()

    rows = []
    for results_file in sorted(args.dir.glob("*/results.json")):
        record = json.loads(results_file.read_text())
        record["experiment"] = results_file.parent.name
        rows.append(record)

    if not rows:
        print(f"No results found under {args.dir}")
        return

    frame = pd.DataFrame(rows)

    front = ["experiment", "model", "noise_rate", "test_macro_f1", "test_f1",
             "test_precision", "test_recall", "ms_per_post", "params_millions"]
    ordered = [c for c in front if c in frame.columns]
    ordered += [c for c in frame.columns if c not in ordered]
    frame = frame[ordered]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(frame.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
