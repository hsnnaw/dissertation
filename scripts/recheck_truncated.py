"""
Re-review the posts that were labelled on truncated text.

The review interface printed only the first 1,200 characters. A quarter of the
sample runs longer, so those judgements were made on less text than the
annotator was given, and disagreement on them partly measures who saw more
rather than who judged differently. The interface was fixed, but the labels
made before that were not.

Only the affected posts are pulled, with the hidden portion marked, so the
re-review is forty-odd posts rather than two hundred. Where the hidden text
contains a clinical term the post is flagged, since that is the case most
likely to change a judgement.

    python -m scripts.recheck_truncated           # write the worksheet
    python -m scripts.recheck_truncated --apply   # fold decisions back in
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

CUTOFF = 1200

TERMS = re.compile(
    r"\b(depress\w*|anxiet\w*|anxious|bipolar|schizophreni\w*|psychosis|psychotic|"
    r"ptsd|ocd|adhd|autis\w*|bpd|anorexi\w*|bulimi\w*|therapy|therapist|"
    r"psychiatr\w*|diagnos\w*|medicat\w*|\bmeds\b|antidepressant\w*|ssri|"
    r"sertraline|prozac|zoloft|xanax|lithium|adderall|concerta|vyvan\w*|"
    r"cbt|dbt|inpatient|hospitalis\w*|hospitaliz\w*|self.harm|suicidal)\b", re.I)

TYPES = {"1": "diagnosis", "2": "symptom", "3": "treatment",
         "4": "help_seeking", "5": "recovery"}
DIRECT = {"e": "explicit", "i": "implicit"}


def cmd_write(args):
    texts = {json.loads(l)["post_id"]: json.loads(l)["text"]
             for l in args.sample.read_text().splitlines() if l.strip()}
    manual = {json.loads(l)["post_id"]: json.loads(l)
              for l in args.manual.read_text().splitlines() if l.strip()}

    rows = []
    for post_id, text in texts.items():
        if len(text) <= CUTOFF or post_id not in manual:
            continue
        hidden = text[CUTOFF:]
        hidden_terms = sorted({m.group(0).lower() for m in TERMS.finditer(hidden)})
        record = manual[post_id]
        rows.append({
            "post_id": post_id,
            "chars": len(text),
            "hidden_chars": len(hidden),
            # The judgement most likely to change is one where the unseen part
            # names something clinical the reviewer never saw.
            "hidden_terms": ", ".join(hidden_terms) if hidden_terms else "",
            "current_disclosure": "y" if record["is_disclosure"] else "n",
            "current_type": record["disclosure_type"],
            "current_directness": record["directness"],
            "new_disclosure": "",
            "new_type": "",
            "new_directness": "",
            "hidden_text": hidden,
            "full_text": text,
        })

    rows.sort(key=lambda r: (not r["hidden_terms"], -r["hidden_chars"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    flagged = sum(1 for r in rows if r["hidden_terms"])
    print(f"{len(rows)} posts were labelled on truncated text")
    print(f"  {flagged} have a clinical term in the part you never saw")
    print(f"  {sum(r['hidden_chars'] for r in rows):,} characters were hidden "
          f"in total")
    print(f"\nwrote {args.out}")
    print("\nThe flagged ones are listed first. Read the hidden_text column;")
    print("leave the three new_ columns blank to keep the existing judgement,")
    print("or fill them to change it. Then:")
    print("  python -m scripts.recheck_truncated --apply")


def cmd_apply(args):
    rows = list(csv.DictReader(args.out.open()))
    manual = [json.loads(l) for l in args.manual.read_text().splitlines()
              if l.strip()]
    by_id = {r["post_id"]: r for r in manual}

    changed = 0
    for row in rows:
        record = by_id.get(row["post_id"])
        if not record:
            continue
        answer = (row.get("new_disclosure") or "").strip().lower()
        if not answer:
            continue

        if answer == "n":
            new = {"is_disclosure": False, "disclosure_type": "none",
                   "directness": "not_applicable"}
        else:
            dtype = (row.get("new_type") or "").strip()
            direct = (row.get("new_directness") or "").strip().lower()
            if dtype not in TYPES or direct not in DIRECT:
                print(f"  {row['post_id']}: y needs a type of 1-5 and a "
                      f"directness of e or i; skipped")
                continue
            new = {"is_disclosure": True, "disclosure_type": TYPES[dtype],
                   "directness": DIRECT[direct]}

        if any(record[k] != v for k, v in new.items()):
            record.update(new)
            changed += 1

    args.manual.write_text("".join(json.dumps(r) + "\n" for r in manual))
    print(f"changed {changed} of {len(rows)} re-reviewed labels")
    print(f"wrote {args.manual}")
    if changed:
        print("\nRescore with:\n  python -m src.agreement score "
              f"--model <labels> --manual {args.manual} "
              "--out outputs/agreement.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path,
                        default=Path("data/processed/review_sample.jsonl"))
    parser.add_argument("--manual", type=Path,
                        default=Path("data/processed/manual_labels.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/truncated_recheck.csv"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    (cmd_apply if args.apply else cmd_write)(args)


if __name__ == "__main__":
    main()
