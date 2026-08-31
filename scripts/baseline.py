"""
Lexical baseline: TF-IDF over word and character n-grams, logistic regression.

The project's argument is that disclosure often has to be inferred rather than
read off the surface, and that a lexical method therefore fails precisely on
the implicit cases. That claim is asserted throughout the codebook and the
analysis but never tested, which leaves the RoBERTa result without a reference
point: 0.92 F1 means little until something simpler has been tried on the same
splits.

This trains the simplest thing that could work and reports it overall and split
by the annotator's directness judgement. The gap between explicit and implicit
is the number the argument rests on.

Writes results.json in the same shape as src.train, so scripts.collect_results
picks it up alongside the transformer runs.

    python -m scripts.baseline --splits data/processed/splits_random \\
                               --output outputs/experiments/baseline_tfidf
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def main():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.pipeline import make_pipeline, make_union

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path,
                        default=Path("data/processed/splits_random"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/experiments/baseline_tfidf"))
    parser.add_argument("--label-key", default="is_disclosure")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = read_jsonl(args.splits / "train.jsonl")
    test = read_jsonl(args.splits / "test.jsonl")
    print(f"train={len(train)} test={len(test)}")

    y_train = np.array([int(bool(r[args.label_key])) for r in train])
    y_test = np.array([int(bool(r[args.label_key])) for r in test])
    print(f"  train {y_train.mean():.1%} positive, test {y_test.mean():.1%}")

    # Word n-grams catch the vocabulary a keyword approach would use. Character
    # n-grams are included because they are more robust to the misspelling and
    # informal register of the corpus, so the baseline is a fair opponent
    # rather than a strawman built to lose.
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000,
                        sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        max_features=50_000, sublinear_tf=True),
    )
    model = make_pipeline(
        features,
        # Balanced weights to match the class weighting used in src.train, so
        # the comparison is about representation rather than loss weighting.
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=args.seed),
    )

    started = time.perf_counter()
    model.fit([r["text"] for r in train], y_train)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probs = model.predict_proba([r["text"] for r in test])[:, 1]
    inference_seconds = time.perf_counter() - started
    preds = (probs >= 0.5).astype(int)

    result = {
        "model": "tfidf-logreg",
        "noise_rate": 0.0,
        "n_train": len(train),
        "n_test": len(test),
        "train_seconds": round(train_seconds, 1),
        "inference_seconds": round(inference_seconds, 3),
        "ms_per_post": round(inference_seconds * 1000 / len(test), 3),
        "n_features": len(model[0].get_feature_names_out()),
        "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "test_precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "test_macro_f1": round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4),
    }
    if len(set(y_test)) > 1:
        result["test_auc"] = round(float(roc_auc_score(y_test, probs)), 4)

    # The point of the exercise. If the baseline holds up on explicit
    # disclosures and falls away on implicit ones, the claim that motivates the
    # whole approach has evidence behind it. If it does not, that is worth
    # knowing before the claim is made in writing.
    print("\nby the annotator's directness judgement")
    print(f"{'slice':<18}{'n':>6}{'F1':>9}{'precision':>11}{'recall':>9}")
    by_directness = {}
    for slice_name in ("explicit", "implicit"):
        idx = [i for i, r in enumerate(test)
               if r.get("directness") == slice_name]
        if len(idx) < 10:
            continue
        # Scored against the whole test set's negatives: a disclosure detector
        # is judged on separating this slice from non-disclosures, not on
        # ranking within the slice.
        keep = idx + [i for i, r in enumerate(test) if not r[args.label_key]]
        yk, pk = y_test[keep], preds[keep]
        by_directness[slice_name] = {
            "n_positive": len(idx),
            "f1": round(float(f1_score(yk, pk, zero_division=0)), 4),
            "precision": round(float(precision_score(yk, pk, zero_division=0)), 4),
            "recall": round(float(recall_score(yk, pk, zero_division=0)), 4),
        }
        d = by_directness[slice_name]
        print(f"{slice_name:<18}{len(idx):>6}{d['f1']:>9.4f}"
              f"{d['precision']:>11.4f}{d['recall']:>9.4f}")
    result["by_directness"] = by_directness

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2))

    with (args.output / "test_predictions.jsonl").open("w") as handle:
        for record, prob, pred in zip(test, probs, preds):
            handle.write(json.dumps({
                "post_id": record.get("post_id"),
                "source": record.get("source"),
                "directness": record.get("directness"),
                "disclosure_type": record.get("disclosure_type"),
                "annotator_confidence": record.get("confidence"),
                "label": int(bool(record[args.label_key])),
                "pred": int(pred),
                "prob": round(float(prob), 4),
            }) + "\n")

    print("\n" + json.dumps({k: v for k, v in result.items()
                             if k != "by_directness"}, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
