"""
Re-review the directness judgements against the codebook's own naming test.

The agreement study left this axis unvalidated. Of 85 agreed disclosures, 56
were reviewer-explicit against model-implicit and 1 the reverse, which is a
definitional split rather than noise: the codebook defines explicit as NAMING
a condition, medication, therapy or diagnosis applied to the author, and the
review treated it as "clearly a disclosure".

Rather than re-reading all 200 posts, this pulls only the ones where the
reviewer said explicit and lists the clinical terms actually present. Where
nothing is named, the naming test decides it. Where something is named, the
question is only whether it attaches to the author rather than to somebody
else, which is quick to see.

    python -m scripts.recheck_directness            # write the worksheet
    python -m scripts.recheck_directness --apply    # fold decisions back in
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Terms that count as naming, if attached to the author. Deliberately wide:
# a term missing here produces a false "nothing named" flag, which costs the
# reviewer a second look, while a term wrongly included costs a mislabel.
TERMS = re.compile(
    r"\b(depress\w*|anxiet\w*|anxious|bipolar|schizophreni\w*|psychosis|psychotic|"
    r"psychopath\w*|sociopath\w*|delusion\w*|hallucinat\w*|paranoi\w*|mania|manic|"
    r"ptsd|ocd|adhd|add\b|autis\w*|aspergers|bpd|borderline|eating disorder|"
    r"anorexi\w*|bulimi\w*|trichotillomania|dissociat\w*|panic attack\w*|"
    r"agoraphobi\w*|insomnia|burnout|"
    r"therapy|therapist|counsell?or|counselling|psychiatr\w*|psycholog\w*|"
    r"diagnos\w*|medicat\w*|\bmeds\b|antidepressant\w*|anti.?anxiety|ssri|snri|"
    r"stimulant\w*|sertraline|fluoxetine|citalopram|prozac|zoloft|lexapro|xanax|"
    r"lithium|adderall|concerta|ritalin|vyvan\w*|wellbutrin|mirtazapine|"
    r"cbt|dbt|inpatient|sectioned|hospitalis\w*|hospitaliz\w*|rehab|"
    r"mental illness|self.harm|suicidal)\b", re.I)

# Words that, immediately before a term, mean it belongs to somebody else.
THIRD_PARTY = re.compile(
    r"\b(my|his|her|their|the)\s+(mum|mother|dad|father|sister|brother|partner|"
    r"husband|wife|girlfriend|boyfriend|friend|son|daughter|parents|family|"
    r"colleague|boss|ex)\b[^.]{0,60}$", re.I)


def clinical_terms(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in TERMS.finditer(text)})


def third_party_context(text: str, term: str) -> bool:
    """Whether every mention of a term sits in somebody else's clause."""
    mentions = [m for m in re.finditer(re.escape(term), text, re.I)]
    if not mentions:
        return False
    return all(THIRD_PARTY.search(text[max(0, m.start() - 80):m.start()])
               for m in mentions)


def cmd_write(args):
    texts = {json.loads(l)["post_id"]: json.loads(l)["text"]
             for l in args.sample.read_text().splitlines() if l.strip()}
    manual = [json.loads(l) for l in args.manual.read_text().splitlines()
              if l.strip()]

    rows = []
    for record in manual:
        if record.get("directness") != "explicit":
            continue
        text = texts.get(record["post_id"], "")
        terms = clinical_terms(text)
        others = [t for t in terms if third_party_context(text, t)]
        mine = [t for t in terms if t not in others]

        if not terms:
            verdict, why = "i", "nothing clinical named anywhere"
        elif not mine:
            verdict, why = "i", f"only named for someone else: {', '.join(others)}"
        else:
            verdict, why = "e", f"names {', '.join(mine)}"

        rows.append({
            "post_id": record["post_id"],
            "current": "e",
            "suggested": verdict,
            "reason": why,
            "agree": "",          # blank means accept the suggestion
            "text": text,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    changes = sum(r["suggested"] == "i" for r in rows)
    print(f"{len(rows)} posts marked explicit")
    print(f"  {changes} where the naming test says implicit")
    print(f"  {len(rows) - changes} where it confirms explicit")
    print(f"\nwrote {args.out}")
    print("\nRead the flagged ones. Leave 'agree' blank to accept the "
          "suggestion,\nor put e or i there to override it. Then:")
    print(f"  python -m scripts.recheck_directness --apply")


def cmd_apply(args):
    rows = list(csv.DictReader(args.out.open()))
    decisions = {}
    for row in rows:
        override = (row.get("agree") or "").strip().lower()
        # An explicit override wins; otherwise the naming test stands.
        decisions[row["post_id"]] = ("explicit" if override == "e"
                                     else "implicit" if override == "i"
                                     else {"e": "explicit", "i": "implicit"}[row["suggested"]])

    manual = [json.loads(l) for l in args.manual.read_text().splitlines()
              if l.strip()]
    changed = 0
    for record in manual:
        new = decisions.get(record["post_id"])
        if new and record.get("directness") != new:
            record["directness"] = new
            changed += 1

    args.manual.write_text("".join(json.dumps(r) + "\n" for r in manual))
    counts = {}
    for r in manual:
        if r["is_disclosure"]:
            counts[r["directness"]] = counts.get(r["directness"], 0) + 1
    print(f"changed {changed} labels in {args.manual}")
    print(f"directness now: {counts}")
    print("\nRescore with:\n  python -m src.agreement score --model <labels> "
          f"--manual {args.manual} --out outputs/agreement.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path,
                        default=Path("data/processed/review_sample.jsonl"))
    parser.add_argument("--manual", type=Path,
                        default=Path("data/processed/manual_labels.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/directness_recheck.csv"))
    parser.add_argument("--apply", action="store_true",
                        help="Fold the worksheet's decisions back into the labels")
    args = parser.parse_args()
    (cmd_apply if args.apply else cmd_write)(args)


if __name__ == "__main__":
    main()
