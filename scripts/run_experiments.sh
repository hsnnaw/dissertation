#!/usr/bin/env bash
# Runs the full experiment grid. Assumes annotation is complete and splits exist.
#
# On a single GPU expect roughly 20 minutes per RoBERTa-base run, so the whole
# grid is a few hours. Run it overnight rather than interactively.

set -euo pipefail

SPLITS_RANDOM="data/processed/splits_random"
SPLITS_SUBREDDIT="data/processed/splits_subreddit"
OUT="outputs/experiments"

mkdir -p "$OUT"

echo "=== Experiment 1: label noise robustness ==="
# Does downstream performance degrade gracefully as label quality drops, or
# does it collapse? Directly addresses the reliability concern about LLM labels.
for rate in 0.0 0.05 0.10 0.20 0.30; do
    echo "--- noise ${rate} ---"
    python -m src.train \
        --splits "$SPLITS_RANDOM" \
        --output "$OUT/noise_${rate}" \
        --noise-rate "$rate"
done

echo "=== Experiment 2: cross-subreddit generalisation ==="
# The gap between these two is the headline: did it learn disclosure, or
# community writing style?
python -m src.train --splits "$SPLITS_RANDOM"    --output "$OUT/gen_seen"
python -m src.train --splits "$SPLITS_SUBREDDIT" --output "$OUT/gen_unseen"

echo "=== Experiment 3: model size against inference cost ==="
for model in roberta-base distilroberta-base; do
    name="${model//\//_}"
    python -m src.train \
        --splits "$SPLITS_RANDOM" \
        --output "$OUT/size_${name}" \
        --model "$model"
done

echo "=== Experiment 4: annotation strategy comparison ==="
# Requires annotating the same posts under each strategy first, then splitting
# each separately. Which prompt produces labels that train the best classifier?
for strategy in zero_shot few_shot cot few_shot_cot; do
    if [ -d "data/processed/splits_${strategy}" ]; then
        python -m src.train \
            --splits "data/processed/splits_${strategy}" \
            --output "$OUT/strategy_${strategy}"
    else
        echo "  skipping ${strategy}: splits not built"
    fi
done

echo "=== Ablation: class weighting ==="
python -m src.train --splits "$SPLITS_RANDOM" --output "$OUT/no_weights" --no-class-weights

echo
echo "Done. Collect results with:"
echo "  python scripts/collect_results.py --dir $OUT"
