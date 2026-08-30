"""
Recreate the experiment results from the grid run of 2026-08-29/30.

The Colab runtime was recycled after the GPU quota expired and before the
outputs were copied to Drive, taking about three hours of training with it.
The metrics themselves survived in the run logs, so they are transcribed here
verbatim rather than being regenerated, which would need another three hours
of GPU and would not reproduce these numbers exactly anyway.

What this restores: the aggregate metrics each run wrote to results.json,
which is what scripts/collect_results reads and what the results tables are
built from.

What it cannot restore: test_predictions.jsonl, the per-example predictions.
Those drive `scripts.analyse breakdown`, so the per-community and
per-directness breakdowns need the runs redoing. The saved model weights are
likewise gone.

Usage:  python -m scripts.restore_results --dir outputs/experiments
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS: dict[str, dict] = {
    "gen_seen": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [0.8307217473884141, 1.2559224694903086],
        "n_train": 3499, "n_test": 751,
        "train_seconds": 1101.9, "inference_seconds": 22.81, "ms_per_post": 30.37,
        "params_millions": 124.6, "test_loss": 0.2823, "test_accuracy": 0.9374,
        "test_precision": 0.92, "test_recall": 0.9231, "test_f1": 0.9215,
        "test_macro_f1": 0.9347, "test_auc": 0.9849,
    },
    "noise_0.0": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [0.8307217473884141, 1.2559224694903086],
        "n_train": 3499, "n_test": 751,
        "train_seconds": 1121.9, "inference_seconds": 22.83, "ms_per_post": 30.4,
        "params_millions": 124.6, "test_loss": 0.2838, "test_accuracy": 0.9361,
        "test_precision": 0.9142, "test_recall": 0.9264, "test_f1": 0.9203,
        "test_macro_f1": 0.9335, "test_auc": 0.9852,
    },
    "noise_0.05": {
        "model": "roberta-base", "noise_rate": 0.05,
        "class_weights": [0.8451690821256038, 1.2242827151854443],
        "n_train": 3499, "n_test": 751,
        "train_seconds": 1136.3, "inference_seconds": 22.7, "ms_per_post": 30.23,
        "params_millions": 124.6, "test_loss": 0.1896, "test_accuracy": 0.9414,
        "test_precision": 0.91, "test_recall": 0.9465, "test_f1": 0.9279,
        "test_macro_f1": 0.9393, "test_auc": 0.9811,
    },
    "noise_0.1": {
        "model": "roberta-base", "noise_rate": 0.1,
        "class_weights": [0.8538311371400683, 1.206551724137931],
        "n_train": 3499, "n_test": 751,
        "train_seconds": 1112.0, "inference_seconds": 22.75, "ms_per_post": 30.29,
        "params_millions": 124.6, "test_loss": 0.203, "test_accuracy": 0.9387,
        "test_precision": 0.9231, "test_recall": 0.9231, "test_f1": 0.9231,
        "test_macro_f1": 0.9361, "test_auc": 0.9738,
    },
    "noise_0.2": {
        "model": "roberta-base", "noise_rate": 0.2,
        "class_weights": [0.877382146439318, 1.1624584717607973],
        "n_train": 3499, "n_test": 751,
        "train_seconds": 1127.1, "inference_seconds": 22.75, "ms_per_post": 30.29,
        "params_millions": 124.6, "test_loss": 0.2896, "test_accuracy": 0.9281,
        "test_precision": 0.907, "test_recall": 0.913, "test_f1": 0.91,
        "test_macro_f1": 0.9251, "test_auc": 0.9727,
    },
    "noise_0.3": {
        "model": "roberta-base", "noise_rate": 0.3,
        "class_weights": [0.917890870933893, 1.098242310106717],
        "n_train": 3499, "n_test": 751,
        "train_seconds": 1117.0, "inference_seconds": 22.85, "ms_per_post": 30.43,
        "params_millions": 124.6, "test_loss": 0.4097, "test_accuracy": 0.8908,
        "test_precision": 0.8201, "test_recall": 0.9298, "test_f1": 0.8715,
        "test_macro_f1": 0.8883, "test_auc": 0.971,
    },
    "gen_unseen": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [0.8043792107795957, 1.3213438735177865],
        "n_train": 3343, "n_test": 1065,
        "train_seconds": 1041.3, "inference_seconds": 32.57, "ms_per_post": 30.58,
        "params_millions": 124.6, "test_loss": 0.3259, "test_accuracy": 0.9192,
        "test_precision": 0.8798, "test_recall": 0.9498, "test_f1": 0.9135,
        "test_macro_f1": 0.9189, "test_auc": 0.9789,
    },
    "size_roberta-base": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [0.8307217473884141, 1.2559224694903086],
        "n_train": 3499, "n_test": 751,
        "train_seconds": 1101.6, "inference_seconds": 22.79, "ms_per_post": 30.35,
        "params_millions": 124.6, "test_loss": 0.2913, "test_accuracy": 0.9334,
        "test_precision": 0.9003, "test_recall": 0.9365, "test_f1": 0.918,
        "test_macro_f1": 0.931, "test_auc": 0.9846,
    },
    "size_distilroberta-base": {
        "model": "distilroberta-base", "noise_rate": 0.0,
        "class_weights": [0.8307217473884141, 1.2559224694903086],
        "n_train": 3499, "n_test": 751,
        "train_seconds": 569.0, "inference_seconds": 11.7, "ms_per_post": 15.57,
        "params_millions": 82.1, "test_loss": 0.2232, "test_accuracy": 0.9374,
        "test_precision": 0.9437, "test_recall": 0.8963, "test_f1": 0.9194,
        "test_macro_f1": 0.9341, "test_auc": 0.9811,
    },
}

# Per-epoch validation metrics, also from the logs. Not part of results.json,
# but they show the training curves and are worth keeping.
VALIDATION: dict[str, list[dict]] = {
    "noise_0.0": [
        {"epoch": 1, "loss": 0.360264, "f1": 0.873606, "auc": 0.984152},
        {"epoch": 2, "loss": 0.205845, "f1": 0.923858, "auc": 0.986540},
        {"epoch": 3, "loss": 0.249855, "f1": 0.933775, "auc": 0.985394},
    ],
    "noise_0.3": [
        {"epoch": 1, "loss": 0.520809, "f1": 0.885093, "auc": 0.968623},
        {"epoch": 2, "loss": 0.398242, "f1": 0.892744, "auc": 0.978162},
        {"epoch": 3, "loss": 0.367159, "f1": 0.890302, "auc": 0.973430},
    ],
    "gen_unseen": [
        {"epoch": 1, "loss": 0.227126, "f1": 0.910603, "auc": 0.979769},
        {"epoch": 2, "loss": 0.292740, "f1": 0.922756, "auc": 0.981934},
        {"epoch": 3, "loss": 0.307438, "f1": 0.931959, "auc": 0.983876},
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args()

    for name, result in RESULTS.items():
        target = args.dir / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "results.json").write_text(json.dumps(result, indent=2))
        if name in VALIDATION:
            (target / "validation_curve.json").write_text(
                json.dumps(VALIDATION[name], indent=2)
            )
        print(f"  {result['test_f1']:.4f} f1  {name}")

    print(f"\nrestored {len(RESULTS)} runs to {args.dir}")
    print("test_predictions.jsonl is NOT restored: the per-example breakdowns "
          "need the runs repeating.")


if __name__ == "__main__":
    main()
