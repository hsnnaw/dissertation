"""
Classifier performance disaggregated by directness, without the annotator's
directness labels.

Human review found those labels unreliable: on 85 posts both sides agreed were
disclosures, a deterministic application of the codebook's naming criterion
gives 50 explicit, the human reviewer 60, and the annotator 14. The annotator
answers "implicit" almost regardless of whether a condition is named, so
slicing by its labels measures that failure rather than the classifier's
behaviour on implicit disclosure.

The criterion itself is mechanical, so it is applied directly here: a post is
explicit when it names a condition, medication, therapy or diagnosis in a
first-person context. Deterministic, inspectable, and reproducible from the
text alone, which the annotator's labels are not.

Both slicings are reported so the difference between them is visible rather
than asserted.

    python -m scripts.directness_breakdown \\
        --splits data/processed/splits_random \\
        --predictions outputs/experiments/gen_seen/test_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Conditions, medications, therapies and diagnostic language. Kept wide: a
# missing term wrongly demotes a post to implicit, which understates the
# explicit slice, so breadth errs in the safer direction.
TERMS = re.compile(
    r"\b(depress\w*|anxiet\w*|anxious|bipolar|schizophreni\w*|psychosis|psychotic|"
    r"psychopath\w*|delusion\w*|hallucinat\w*|paranoi\w*|mania|manic|"
    r"ptsd|ocd|adhd|add\b|autis\w*|aspergers|bpd|borderline|eating disorder|"
    r"anorexi\w*|bulimi\w*|trichotillomania|dissociat\w*|panic attack\w*|"
    r"agoraphobi\w*|insomnia|"
    r"therapy|therapist|counsell?or|counselling|psychiatr\w*|psycholog\w*|"
    r"diagnos\w*|medicat\w*|\bmeds\b|antidepressant\w*|anti.?anxiety|ssri|snri|"
    r"sertraline|fluoxetine|citalopram|prozac|zoloft|lexapro|xanax|lithium|"
    r"adderall|concerta|ritalin|vyvan\w*|wellbutrin|mirtazapine|"
    r"cbt|dbt|inpatient|sectioned|hospitalis\w*|hospitaliz\w*|rehab|"
    r"mental illness|self.harm)\b", re.I)

# First person within the clause before the term. "my depression" counts;
# "my brother's depression" does not, because the subject changed.
FIRST_PERSON = re.compile(r"\b(i|my|me|i'm|im|ive|i've|myself)\b[^.]{0,40}$", re.I)


def names_own_condition(text: str) -> bool:
    return any(FIRST_PERSON.search(text[max(0, m.start() - 60):m.start()])
               for m in TERMS.finditer(text))


def score(rows: list[dict], negatives: list[dict]) -> dict:
    """
    Score one slice of positives against the whole test set's negatives.

    A disclosure detector is judged on separating disclosures from
    non-disclosures, so the negatives must be shared. Scoring a slice against
    only its own members would make precision depend on the slice's size
    rather than on the model.
    """
    pool = rows + negatives
    tp = sum(r["label"] == 1 and r["pred"] == 1 for r in pool)
    fp = sum(r["label"] == 0 and r["pred"] == 1 for r in pool)
    fn = sum(r["label"] == 1 and r["pred"] == 0 for r in pool)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n_positive": len(rows), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path,
                        default=Path("data/processed/splits_random"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("outputs/experiments/gen_seen/test_predictions.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/directness_breakdown.json"))
    args = parser.parse_args()

    texts = {json.loads(l)["post_id"]: json.loads(l)["text"]
             for l in (args.splits / "test.jsonl").read_text().splitlines()
             if l.strip()}
    preds = [json.loads(l) for l in args.predictions.read_text().splitlines()
             if l.strip()]

    missing = sum(p["post_id"] not in texts for p in preds)
    if missing:
        print(f"warning: {missing} predictions have no matching post text")

    for p in preds:
        p["names_own"] = names_own_condition(texts.get(p["post_id"], ""))

    negatives = [p for p in preds if p["label"] == 0]
    positives = [p for p in preds if p["label"] == 1]
    print(f"{len(preds)} test predictions: {len(positives)} positive, "
          f"{len(negatives)} negative\n")

    result = {"n_test": len(preds), "n_positive": len(positives)}

    print("By the codebook's naming criterion, applied deterministically")
    print(f"  {'slice':<14}{'n':>6}{'F1':>9}{'precision':>11}{'recall':>9}")
    result["by_naming"] = {}
    for name, rows in (("explicit", [p for p in positives if p["names_own"]]),
                       ("implicit", [p for p in positives if not p["names_own"]])):
        s = score(rows, negatives)
        result["by_naming"][name] = s
        print(f"  {name:<14}{s['n_positive']:>6}{s['f1']:>9.4f}"
              f"{s['precision']:>11.4f}{s['recall']:>9.4f}")

    print("\nBy the annotator's directness labels, for comparison")
    print(f"  {'slice':<14}{'n':>6}{'F1':>9}{'precision':>11}{'recall':>9}")
    result["by_annotator"] = {}
    for name in ("explicit", "implicit"):
        rows = [p for p in positives if p.get("directness") == name]
        if not rows:
            continue
        s = score(rows, negatives)
        result["by_annotator"][name] = s
        print(f"  {name:<14}{s['n_positive']:>6}{s['f1']:>9.4f}"
              f"{s['precision']:>11.4f}{s['recall']:>9.4f}")

    # How far apart the two slicings are, which is the evidence that the
    # annotator's labels are not usable for this.
    agree = sum((p.get("directness") == "explicit") == p["names_own"]
                for p in positives)
    result["annotator_agreement_with_naming"] = round(agree / len(positives), 4)
    print(f"\nThe annotator's labels match the naming criterion on "
          f"{agree}/{len(positives)} positives ({agree/len(positives):.0%}).")
    print("That is why the disaggregation above uses the criterion directly.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
