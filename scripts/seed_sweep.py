"""
Repeat the headline runs across seeds and report mean and spread.

Every result so far comes from a single run at seed 42, which leaves the
central comparison undefended: the generalisation gap is 0.008 F1, and one
run per condition cannot say whether that is a real difference or the
variance of the training procedure. Three seeds per condition gives a
standard deviation to quote it against.

Only the two conditions the argument rests on are repeated. Repeating the
noise sweep would be fifteen runs for a curve whose shape is already clear at
one seed, and the compute is better spent elsewhere.

Also regenerates test_predictions.jsonl, which a reclaimed Colab runtime took
and which the per-community breakdown needs.

    python -m scripts.seed_sweep --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CONDITIONS = {
    "gen_seen": Path("data/processed/splits_random"),
    "gen_unseen": Path("data/processed/splits_subreddit"),
}
METRICS = ("test_f1", "test_macro_f1", "test_precision", "test_recall", "test_auc")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--outdir", type=Path,
                        default=Path("outputs/experiments"))
    parser.add_argument("--summary", type=Path,
                        default=Path("outputs/seed_sweep.json"))
    args = parser.parse_args()

    from src.train import train

    results: dict[str, dict[int, dict]] = {}
    for name, splits in CONDITIONS.items():
        if not (splits / "train.jsonl").exists():
            print(f"skipping {name}: {splits} not built")
            continue
        results[name] = {}
        for seed in args.seeds:
            # Seed 42 keeps the original directory name so the existing
            # results and figures stay valid; the others get suffixed.
            out = args.outdir / (name if seed == 42 else f"{name}_seed{seed}")
            print(f"\n{'=' * 66}\n{name}  seed {seed}  ->  {out}\n{'=' * 66}")
            results[name][seed] = train(splits_dir=splits, output_dir=out,
                                        seed=seed)

    summary = {}
    print(f"\n\n{'=' * 66}\nSummary across seeds\n{'=' * 66}")
    for name, by_seed in results.items():
        summary[name] = {"seeds": sorted(by_seed)}
        print(f"\n{name}  (n = {len(by_seed)})")
        print(f"  {'metric':<18}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
        for metric in METRICS:
            values = [r[metric] for r in by_seed.values() if metric in r]
            if not values:
                continue
            # Sample standard deviation needs at least two runs; a single seed
            # has no spread to report and saying 0.0 would overstate it.
            sd = statistics.stdev(values) if len(values) > 1 else None
            summary[name][metric] = {
                "mean": round(statistics.mean(values), 4),
                "sd": round(sd, 4) if sd is not None else None,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "values": [round(v, 4) for v in values],
            }
            m = summary[name][metric]
            sd_s = f"{m['sd']:>9.4f}" if m["sd"] is not None else f"{'—':>9}"
            print(f"  {metric:<18}{m['mean']:>9.4f}{sd_s}"
                  f"{m['min']:>9.4f}{m['max']:>9.4f}")

    # The comparison the dissertation turns on, stated against its own spread.
    if {"gen_seen", "gen_unseen"} <= set(summary):
        for metric in ("test_f1", "test_macro_f1", "test_auc"):
            seen, unseen = summary["gen_seen"].get(metric), summary["gen_unseen"].get(metric)
            if not seen or not unseen:
                continue
            gap = seen["mean"] - unseen["mean"]
            sds = [s for s in (seen["sd"], unseen["sd"]) if s is not None]
            pooled = (sum(s ** 2 for s in sds) / len(sds)) ** 0.5 if sds else None
            summary.setdefault("generalisation_gap", {})[metric] = {
                "gap": round(gap, 4),
                "pooled_sd": round(pooled, 4) if pooled else None,
                "gap_in_sds": round(gap / pooled, 2) if pooled else None,
            }
        print(f"\n{'=' * 66}\nGeneralisation gap against its own variance\n{'=' * 66}")
        for metric, g in summary["generalisation_gap"].items():
            n_sd = f"{g['gap_in_sds']:.2f} sd" if g["gap_in_sds"] is not None else "—"
            print(f"  {metric:<18}gap {g['gap']:+.4f}   pooled sd "
                  f"{g['pooled_sd']}   {n_sd}")
        print("\n  A gap smaller than about two pooled standard deviations "
              "should be\n  reported as within run-to-run variance, not as an "
              "effect.")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
