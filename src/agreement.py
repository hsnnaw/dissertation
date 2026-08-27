"""
Human agreement study.

Two commands:

    sample   draw a stratified sample from the annotated corpus for manual review
    label    label that sample manually, blind to the model's judgement
    score    compute agreement between the manual and model labels

The point of this is to establish whether the model-generated labels are
trustworthy. Without it, every downstream result rests on an unverified
assumption about label quality.

Blindness is enforced rather than assumed: the sample file written by `sample`
contains no model labels at all, and the manual labels are only joined back to
the model output at scoring time. Reviewing your own agreement while labelling
would defeat the purpose entirely.

Stratification is by the model's confidence and its disclosure judgement.
Sampling uniformly at random would fill the sample with easy negatives, since
most posts are not disclosures, and tell you almost nothing about the cases
where the model is uncertain.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.prompts import DIRECTNESS, DISCLOSURE_TYPES


# --------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------

def stratified_sample(
    records: list[dict],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """
    Draw n records stratified by model judgement and confidence band.

    Four strata: positive/negative crossed with confident/uncertain, split at
    0.8. Uncertain cases are deliberately over-sampled relative to their share
    of the corpus, because they carry most of the information about where the
    codebook is failing.
    """
    rng = random.Random(seed)
    strata = defaultdict(list)

    for record in records:
        if "error" in record:
            continue
        band = "confident" if record.get("confidence", 0) >= 0.8 else "uncertain"
        strata[(record["is_disclosure"], band)].append(record)

    # Weights over-represent the uncertain strata on purpose.
    weights = {
        (True, "uncertain"): 0.35,
        (False, "uncertain"): 0.30,
        (True, "confident"): 0.20,
        (False, "confident"): 0.15,
    }

    sample = []
    for key, weight in weights.items():
        pool = strata.get(key, [])
        take = min(len(pool), round(n * weight))
        sample.extend(rng.sample(pool, take))

    # Top up from the largest remaining pool if rounding left us short.
    if len(sample) < n:
        chosen = {r["post_id"] for r in sample}
        remainder = [r for r in records
                     if "error" not in r and r["post_id"] not in chosen]
        rng.shuffle(remainder)
        sample.extend(remainder[:n - len(sample)])

    rng.shuffle(sample)
    return sample[:n]


def cmd_sample(args):
    records = [json.loads(line) for line in args.input.read_text().splitlines()
               if line.strip()]
    sample = stratified_sample(records, args.n, args.seed)

    # The written file carries the text and nothing else. No model label, no
    # confidence, no disclosure type: the reviewer must not see them.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in sample:
            handle.write(json.dumps({
                "post_id": record["post_id"],
                "text": record["text"],
            }) + "\n")

    print(f"wrote {len(sample)} posts to {args.output}")
    print("This file contains no model labels. Label it with:")
    print(f"  python -m src.agreement label --sample {args.output} "
          f"--output data/processed/manual_labels.jsonl")


# --------------------------------------------------------------------------
# label
# --------------------------------------------------------------------------

TYPE_KEYS = {
    "1": "diagnosis", "2": "symptom", "3": "treatment",
    "4": "help_seeking", "5": "recovery",
}
DIRECT_KEYS = {"e": "explicit", "i": "implicit"}


def cmd_label(args):
    posts = [json.loads(line) for line in args.sample.read_text().splitlines()
             if line.strip()]

    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["post_id"])

    pending = [p for p in posts if p["post_id"] not in done]
    print(f"{len(posts)} posts, {len(done)} labelled, {len(pending)} remaining")
    print("Enter y/n for disclosure. Then type and directness if yes.")
    print("Type: 1 diagnosis  2 symptom  3 treatment  4 help_seeking  5 recovery")
    print("Directness: e explicit  i implicit")
    print("Enter s to skip, q to save and quit.\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as handle:
        for i, post in enumerate(pending, 1):
            print("=" * 74)
            print(f"[{i}/{len(pending)}]")
            print(post["text"][:1200])
            print("-" * 74)

            answer = input("disclosure? (y/n/s/q) ").strip().lower()
            if answer == "q":
                print("saved, exiting")
                break
            if answer == "s":
                continue

            if answer == "y":
                dtype = ""
                while dtype not in TYPE_KEYS:
                    dtype = input("type (1-5) ").strip()
                directness = ""
                while directness not in DIRECT_KEYS:
                    directness = input("directness (e/i) ").strip().lower()
                record = {
                    "post_id": post["post_id"],
                    "is_disclosure": True,
                    "disclosure_type": TYPE_KEYS[dtype],
                    "directness": DIRECT_KEYS[directness],
                }
            else:
                record = {
                    "post_id": post["post_id"],
                    "is_disclosure": False,
                    "disclosure_type": "none",
                    "directness": "not_applicable",
                }

            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print()


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------

def cohens_kappa(a: list, b: list) -> float:
    """
    Cohen's kappa for two annotators over the same items.

    Kappa rather than raw agreement because raw agreement is inflated by the
    class imbalance here: two annotators who both say 'no disclosure' most of
    the time will agree often by chance alone.
    """
    if not a:
        return 0.0
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n

    count_a, count_b = Counter(a), Counter(b)
    expected = sum(
        (count_a[label] / n) * (count_b[label] / n)
        for label in set(count_a) | set(count_b)
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def interpret(kappa: float) -> str:
    """Landis and Koch (1977) bands. Conventional, and worth citing as such."""
    if kappa < 0.0:
        return "poor"
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"


def cmd_score(args):
    model = {}
    for line in args.model.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if "error" not in record:
                model[record["post_id"]] = record

    manual = {}
    for line in args.manual.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            manual[record["post_id"]] = record

    shared = sorted(set(model) & set(manual))
    if not shared:
        print("No overlapping post ids between the two files.")
        return

    human_disc = [manual[i]["is_disclosure"] for i in shared]
    model_disc = [model[i]["is_disclosure"] for i in shared]

    kappa = cohens_kappa(human_disc, model_disc)
    raw = sum(a == b for a, b in zip(human_disc, model_disc)) / len(shared)

    print(f"n = {len(shared)}")
    print(f"raw agreement  {raw:.3f}")
    print(f"Cohen's kappa  {kappa:.3f}  ({interpret(kappa)})")

    # Disagreement direction matters. Model-positive-human-negative is an
    # over-flagging failure; the reverse is a missed disclosure. They have
    # different consequences for a protective tool.
    model_only = sum(m and not h for h, m in zip(human_disc, model_disc))
    human_only = sum(h and not m for h, m in zip(human_disc, model_disc))
    print(f"\nmodel positive, reviewer negative: {model_only}")
    print(f"reviewer positive, model negative: {human_only}")

    # Where both agree a disclosure is present, how well do type and
    # directness line up?
    both = [i for i in shared
            if manual[i]["is_disclosure"] and model[i]["is_disclosure"]]
    if both:
        type_kappa = cohens_kappa(
            [manual[i]["disclosure_type"] for i in both],
            [model[i]["disclosure_type"] for i in both],
        )
        direct_kappa = cohens_kappa(
            [manual[i]["directness"] for i in both],
            [model[i]["directness"] for i in both],
        )
        print(f"\non {len(both)} agreed disclosures:")
        print(f"  type kappa       {type_kappa:.3f}  ({interpret(type_kappa)})")
        print(f"  directness kappa {direct_kappa:.3f}  ({interpret(direct_kappa)})")

    # Agreement by the model's own confidence tells you whether the confidence
    # score is meaningful, which matters if it is used for anything downstream.
    print("\nagreement by model confidence:")
    bands = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]
    for low, high in bands:
        ids = [i for i in shared if low <= model[i].get("confidence", 0) < high]
        if not ids:
            continue
        hits = sum(manual[i]["is_disclosure"] == model[i]["is_disclosure"]
                   for i in ids)
        print(f"  {low:.2f}-{high:.2f}  n={len(ids):4d}  agreement {hits / len(ids):.3f}")

    # Which disclosure types the model most often gets wrong, per the reviewer.
    print("\ndisagreements by reviewer's disclosure type:")
    wrong = Counter(
        manual[i]["disclosure_type"] for i in shared
        if manual[i]["is_disclosure"] != model[i]["is_disclosure"]
    )
    total = Counter(manual[i]["disclosure_type"] for i in shared)
    for dtype in sorted(total):
        if total[dtype]:
            print(f"  {dtype:14} {wrong[dtype]:3d}/{total[dtype]:3d}  "
                  f"({wrong[dtype] / total[dtype]:.1%})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "n": len(shared),
            "raw_agreement": round(raw, 4),
            "kappa": round(kappa, 4),
            "interpretation": interpret(kappa),
            "model_only_positive": model_only,
            "human_only_positive": human_only,
        }, indent=2))
        print(f"\nwrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="draw a stratified sample for review")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("data/processed/review_sample.jsonl"))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("label", help="label the sample manually")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("data/processed/manual_labels.jsonl"))
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("score", help="compute agreement")
    p.add_argument("--model", type=Path, required=True,
                   help="annotation output from src.annotate")
    p.add_argument("--manual", type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=Path("outputs/agreement.json"))
    p.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()