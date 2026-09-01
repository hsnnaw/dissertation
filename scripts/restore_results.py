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
    "baseline_random": {
        "model": "tfidf-logreg", "noise_rate": 0.0,
        "n_train": 3499, "n_test": 751,
        "train_seconds": 6.6, "inference_seconds": 1.031, "ms_per_post": 1.372,
        "n_features": 100000, "test_loss": None, "test_accuracy": 0.9028,
        "test_precision": 0.8717, "test_recall": 0.8863, "test_f1": 0.8789,
        "test_macro_f1": 0.8989, "test_auc": 0.9617,
        "by_directness": {
            "explicit": {"n_positive": 58, "f1": 0.6434,
                         "precision": 0.5412, "recall": 0.7931},
            "implicit": {"n_positive": 241, "f1": 0.8778,
                         "precision": 0.8488, "recall": 0.9087},
        },
    },
    "baseline_subreddit": {
        "model": "tfidf-logreg", "noise_rate": 0.0,
        "n_train": 3343, "n_test": 1065,
        "train_seconds": 7.8, "inference_seconds": 1.241, "ms_per_post": 1.165,
        "n_features": 100000, "test_loss": None, "test_accuracy": 0.8789,
        "test_precision": 0.8402, "test_recall": 0.9017, "test_f1": 0.8698,
        "test_macro_f1": 0.8783, "test_auc": 0.9465,
        "by_directness": {
            "explicit": {"n_positive": 67, "f1": 0.532,
                         "precision": 0.3971, "recall": 0.806},
            "implicit": {"n_positive": 411, "f1": 0.8667,
                         "precision": 0.8214, "recall": 0.9173},
        },
    },
    "gen_seen": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [0.8307217473884141, 1.2559224694903086],
        "n_train": 3499, "n_test": 751, "seed": 42,
        "train_seconds": 1115.8, "inference_seconds": 22.67, "ms_per_post": 30.18,
        "params_millions": 124.6, "test_loss": 0.2883, "test_accuracy": 0.9334,
        "test_precision": 0.9055, "test_recall": 0.9298, "test_f1": 0.9175,
        "test_macro_f1": 0.9308, "test_auc": 0.9847,
    },
    "gen_seen_seed43": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [0.8307217473884141, 1.2559224694903086],
        "n_train": 3499, "n_test": 751, "seed": 43,
        "train_seconds": 1130.0, "inference_seconds": 23.17, "ms_per_post": 30.85,
        "params_millions": 124.6, "test_loss": 0.2842, "test_accuracy": 0.9467,
        "test_precision": 0.942, "test_recall": 0.9231, "test_f1": 0.9324,
        "test_macro_f1": 0.9442, "test_auc": 0.9836,
    },
    "gen_seen_seed44": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [0.8307217473884141, 1.2559224694903086],
        "n_train": 3499, "n_test": 751, "seed": 44,
        "train_seconds": 1133.9, "inference_seconds": 23.22, "ms_per_post": 30.91,
        "params_millions": 124.6, "test_loss": 0.1946, "test_accuracy": 0.9294,
        "test_precision": 0.9046, "test_recall": 0.9197, "test_f1": 0.9121,
        "test_macro_f1": 0.9266, "test_auc": 0.9861,
    },
    "no_weights": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [1.0, 1.0],
        "n_train": 3499, "n_test": 751, "seed": 42,
        "train_seconds": 1139.5, "inference_seconds": 23.51, "ms_per_post": 31.3,
        "params_millions": 124.6, "test_loss": 0.2756, "test_accuracy": 0.9334,
        "test_precision": 0.9192, "test_recall": 0.913, "test_f1": 0.9161,
        "test_macro_f1": 0.9305, "test_auc": 0.9841,
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
        "n_train": 3343, "n_test": 1065, "seed": 42,
        "train_seconds": 1089.1, "inference_seconds": 33.66, "ms_per_post": 31.6,
        "params_millions": 124.6, "test_loss": 0.3539, "test_accuracy": 0.9164,
        "test_precision": 0.8663, "test_recall": 0.9623, "test_f1": 0.9118,
        "test_macro_f1": 0.9162, "test_auc": 0.9802,
    },
    "gen_unseen_seed43": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [0.8043792107795957, 1.3213438735177865],
        "n_train": 3343, "n_test": 1065, "seed": 43,
        "train_seconds": 1085.5, "inference_seconds": 32.66, "ms_per_post": 30.67,
        "params_millions": 124.6, "test_loss": 0.3565, "test_accuracy": 0.9127,
        "test_precision": 0.8723, "test_recall": 0.9435, "test_f1": 0.9065,
        "test_macro_f1": 0.9123, "test_auc": 0.9761,
    },
    "gen_unseen_seed44": {
        "model": "roberta-base", "noise_rate": 0.0,
        "class_weights": [0.8043792107795957, 1.3213438735177865],
        "n_train": 3343, "n_test": 1065, "seed": 44,
        "train_seconds": 1088.6, "inference_seconds": 33.91, "ms_per_post": 31.84,
        "params_millions": 124.6, "test_loss": 0.3314, "test_accuracy": 0.9211,
        "test_precision": 0.8803, "test_recall": 0.954, "test_f1": 0.9157,
        "test_macro_f1": 0.9208, "test_auc": 0.9793,
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
