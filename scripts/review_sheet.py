"""
Spreadsheet front end for the agreement study.

`src.agreement label` asks one post at a time, which suits a careful pass but
is slow over a couple of hundred posts and gives no way to go back and revise
an earlier call. This writes the sample to a CSV instead, so the whole sample
can be worked through in a spreadsheet and read back afterwards.

The judgements are identical either way. Only the interface differs, and the
importer writes exactly the records src.agreement.score expects.

The sheet carries verbatim post text and is therefore covered by the same
constraint as the rest of the corpus: keep it local, do not commit it, and do
not put it on cloud storage.

    python -m scripts.review_sheet export \\
        --sample data/processed/review_sample.jsonl \\
        --out data/processed/review_sheet.csv

    # ... fill in the three judgement columns ...

    python -m scripts.review_sheet import \\
        --sheet data/processed/review_sheet.csv \\
        --out data/processed/manual_labels.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

TYPES = {
    "1": "diagnosis", "2": "symptom", "3": "treatment",
    "4": "help_seeking", "5": "recovery",
}
DIRECTNESS = {"e": "explicit", "i": "implicit"}

COLUMNS = ["post_id", "chars", "disclosure", "type", "directness", "text"]


def cmd_export(args):
    posts = [json.loads(line) for line in args.sample.read_text().splitlines()
             if line.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for post in posts:
            # Judgement columns left blank deliberately: a pre-filled default
            # would be answered by inertia rather than read.
            writer.writerow([post["post_id"], len(post["text"]),
                             "", "", "", post["text"]])

    print(f"wrote {len(posts)} posts to {args.out}")
    print()
    print("Fill in three columns per row:")
    print("  disclosure   y or n")
    print("  type         1 diagnosis  2 symptom  3 treatment  "
          "4 help_seeking  5 recovery   (only when y)")
    print("  directness   e explicit  i implicit                "
          "(only when y)")
    print()
    print("Rows left blank are skipped, so a partial pass is fine.")
    print("The sheet holds post text: keep it local and off cloud storage.")


def cmd_import(args):
    rows = list(csv.DictReader(args.sheet.open()))

    records, skipped, problems = [], 0, []
    for n, row in enumerate(rows, start=2):   # 2: row 1 is the header
        post_id = (row.get("post_id") or "").strip()
        answer = (row.get("disclosure") or "").strip().lower()

        if not answer:
            skipped += 1
            continue

        if answer in ("n", "no", "false", "0"):
            records.append({
                "post_id": post_id, "is_disclosure": False,
                "disclosure_type": "none", "directness": "not_applicable",
            })
            continue

        if answer not in ("y", "yes", "true", "1"):
            problems.append(f"row {n}: disclosure is {answer!r}, expected y or n")
            continue

        dtype = (row.get("type") or "").strip()
        direct = (row.get("directness") or "").strip().lower()

        # A y with no type or directness cannot be scored, and guessing one
        # would invent a judgement the reviewer did not make.
        if dtype not in TYPES:
            problems.append(f"row {n}: y needs a type of 1-5, got {dtype!r}")
            continue
        if direct not in DIRECTNESS:
            problems.append(f"row {n}: y needs a directness of e or i, got {direct!r}")
            continue

        records.append({
            "post_id": post_id, "is_disclosure": True,
            "disclosure_type": TYPES[dtype],
            "directness": DIRECTNESS[direct],
        })

    if problems:
        print(f"{len(problems)} rows could not be read:\n", file=sys.stderr)
        for problem in problems[:25]:
            print(f"  {problem}", file=sys.stderr)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more", file=sys.stderr)
        print("\nNothing written. Fix these and rerun.", file=sys.stderr)
        raise SystemExit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    positive = sum(r["is_disclosure"] for r in records)
    print(f"wrote {len(records)} labels to {args.out}")
    print(f"  {positive} disclosure, {len(records) - positive} not "
          f"({positive / len(records):.1%} positive)" if records else "")
    if skipped:
        print(f"  {skipped} rows left blank, not included")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="write the sample to a CSV")
    p.add_argument("--sample", type=Path,
                   default=Path("data/processed/review_sample.jsonl"))
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/review_sheet.csv"))
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="read the filled CSV back")
    p.add_argument("--sheet", type=Path,
                   default=Path("data/processed/review_sheet.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/manual_labels.jsonl"))
    p.set_defaults(func=cmd_import)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
