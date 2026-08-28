"""
RoBERTa fine-tuning for self-disclosure detection.

Supports the experiments directly rather than needing separate scripts:

  --noise-rate    flips a proportion of training labels, for the label noise
                  robustness study
  --model         swaps the encoder, for the size-versus-cost comparison
  --class-weights applies inverse-frequency weighting, for class imbalance

Validation and test labels are never corrupted by --noise-rate. Noise models a
flawed annotator, and a flawed annotator does not change the ground truth you
are measuring against.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def inject_noise(
    records: list[dict],
    rate: float,
    label_key: str = "is_disclosure",
    seed: int = 42,
) -> tuple[list[dict], int]:
    """
    Flip a proportion of labels at random.

    Flipping is symmetric: positives and negatives are equally likely to be
    corrupted. Real annotator error is usually asymmetric, so this is a
    simplification worth stating in the write-up rather than glossing over.
    """
    if rate <= 0:
        return records, 0

    rng = random.Random(seed)
    corrupted = []
    flipped = 0
    for record in records:
        record = dict(record)
        if rng.random() < rate:
            record[label_key] = not record[label_key]
            flipped += 1
        corrupted.append(record)
    return corrupted, flipped


def compute_class_weights(records: list[dict], label_key: str) -> list[float]:
    """Inverse-frequency weights, normalised to mean 1 so the loss scale holds."""
    positive = sum(bool(r[label_key]) for r in records)
    negative = len(records) - positive
    if positive == 0 or negative == 0:
        return [1.0, 1.0]
    total = positive + negative
    weights = [total / (2 * negative), total / (2 * positive)]
    return weights


def build_metrics(threshold: float = 0.5):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    def compute(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits[:, 1] + logits[:, 0]))
        preds = (probs >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }
        # AUC is undefined on a single-class split, which happens with small
        # held-out subreddits.
        if len(set(labels)) > 1:
            metrics["auc"] = roc_auc_score(labels, probs)
        return metrics

    return compute


def train(
    splits_dir: Path,
    output_dir: Path,
    model_name: str = "roberta-base",
    label_key: str = "is_disclosure",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    noise_rate: float = 0.0,
    class_weights: bool = True,
    seed: int = 42,
) -> dict:
    import torch
    from torch import nn
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    set_seed(seed)

    train_records = read_jsonl(splits_dir / "train.jsonl")
    val_records = read_jsonl(splits_dir / "val.jsonl")
    test_records = read_jsonl(splits_dir / "test.jsonl")

    print(f"train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    # A split with only one class produces meaningless metrics: accuracy goes
    # to 1.0 while precision, recall and F1 all go to 0, which looks like a
    # result but is not one. Surface it loudly rather than letting it into the
    # results table.
    for name, records in [("train", train_records), ("val", val_records),
                          ("test", test_records)]:
        positives = sum(bool(r[label_key]) for r in records)
        rate = positives / len(records) if records else 0
        print(f"  {name}: {positives}/{len(records)} positive ({rate:.1%})")
        if records and (positives == 0 or positives == len(records)):
            print(f"\n  WARNING: {name} split contains a single class. "
                  f"Metrics from this run are not interpretable.")

    if noise_rate > 0:
        train_records, flipped = inject_noise(train_records, noise_rate,
                                              label_key, seed)
        print(f"injected {noise_rate:.0%} noise: {flipped} training labels flipped")

    weights = compute_class_weights(train_records, label_key) if class_weights \
        else [1.0, 1.0]
    print(f"class weights: {[round(w, 3) for w in weights]}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_dataset(records: list[dict]) -> Dataset:
        return Dataset.from_dict({
            "text": [r["text"] for r in records],
            "labels": [int(bool(r[label_key])) for r in records],
        })

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length,
                         padding="max_length")

    datasets = {
        name: to_dataset(records).map(tokenize, batched=True,
                                      remove_columns=["text"])
        for name, records in [("train", train_records), ("val", val_records),
                              ("test", test_records)]
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    weight_tensor = torch.tensor(weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        """Cost-sensitive learning: penalise minority-class errors more."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(
                weight=weight_tensor.to(outputs.logits.device)
            )(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    output_dir.mkdir(parents=True, exist_ok=True)

    # transformers 5.x removed warmup_ratio, so the step count is computed here
    # instead. warmup_steps exists in both 4.x and 5.x, which keeps this script
    # portable across whichever version the GPU box happens to have.
    import math
    steps_per_epoch = math.ceil(len(train_records) / batch_size)
    warmup_steps = max(1, int(steps_per_epoch * epochs * 0.06))

    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        seed=seed,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        compute_metrics=build_metrics(),
    )

    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started

    # Test is evaluated once, at the end. Selecting on it would inflate the
    # numbers and invalidate the comparison across experiment conditions.
    inference_started = time.perf_counter()
    test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    inference_seconds = time.perf_counter() - inference_started

    # Per-example predictions are written out so that performance can be broken
    # down afterwards by directness, disclosure type, and source community.
    # Aggregate metrics can conceal poor performance on implicit disclosures,
    # which are the cases this project is most interested in, so the breakdown
    # is not optional.
    predictions = trainer.predict(datasets["test"])
    logits = predictions.predictions
    probs = 1 / (1 + np.exp(-logits[:, 1] + logits[:, 0]))
    preds = (probs >= 0.5).astype(int)

    with (output_dir / "test_predictions.jsonl").open("w") as handle:
        for record, prob, pred in zip(test_records, probs, preds):
            handle.write(json.dumps({
                "post_id": record.get("post_id"),
                "source": record.get("source"),
                "directness": record.get("directness"),
                "disclosure_type": record.get("disclosure_type"),
                "annotator_confidence": record.get("confidence"),
                "label": int(bool(record[label_key])),
                "pred": int(pred),
                "prob": round(float(prob), 4),
            }) + "\n")

    result = {
        "model": model_name,
        "noise_rate": noise_rate,
        "class_weights": weights,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "train_seconds": round(train_seconds, 1),
        "inference_seconds": round(inference_seconds, 2),
        "ms_per_post": round(inference_seconds * 1000 / len(test_records), 2),
        "params_millions": round(
            sum(p.numel() for p in model.parameters()) / 1e6, 1
        ),
        **{k: round(float(v), 4) for k, v in test_metrics.items()
           if isinstance(v, (int, float))},
    }

    (output_dir / "results.json").write_text(json.dumps(result, indent=2))
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    print("\n" + json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path,
                        default=Path("data/processed/splits"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--label-key", default="is_disclosure")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--noise-rate", type=float, default=0.0,
                        help="Proportion of TRAINING labels to flip (0 to 1)")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        splits_dir=args.splits,
        output_dir=args.output,
        model_name=args.model,
        label_key=args.label_key,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        noise_rate=args.noise_rate,
        class_weights=not args.no_class_weights,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()