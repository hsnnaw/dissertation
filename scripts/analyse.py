"""
Analysis and figures from experiment outputs.

Reads the per-example predictions written by src.train and produces the
breakdowns and plots that Chapter 4 needs:

    breakdown   performance by directness, disclosure type, and community
    noise       degradation curve across injected label noise rates
    cost        classification quality against measured inference cost
    gap         seen against unseen communities

    python -m scripts.analyse breakdown --dir outputs/experiments/gen_seen
    python -m scripts.analyse noise --dir outputs/experiments
    python -m scripts.analyse cost --dir outputs/experiments
    python -m scripts.analyse gap --dir outputs/experiments

The breakdown is the important one. Aggregate F1 can look healthy while the
model fails on implicit disclosures, which are precisely the cases a lexical
baseline would also miss and therefore the cases that justify the approach at
all. Reporting only the aggregate would conceal that.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def prf(rows: list[dict]) -> dict:
    """Precision, recall, F1 for a group of predictions."""
    tp = sum(r["label"] == 1 and r["pred"] == 1 for r in rows)
    fp = sum(r["label"] == 0 and r["pred"] == 1 for r in rows)
    fn = sum(r["label"] == 1 and r["pred"] == 0 for r in rows)
    tn = sum(r["label"] == 0 and r["pred"] == 0 for r in rows)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": len(rows),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round((tp + tn) / len(rows), 3) if rows else 0.0,
    }


def group_by(rows: list[dict], key: str) -> dict[str, dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key) or "unknown"].append(row)
    return {name: prf(group) for name, group in sorted(groups.items())}


def print_table(title: str, table: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(f"{'group':<22}{'n':>7}{'P':>8}{'R':>8}{'F1':>8}{'acc':>8}")
    for name, m in table.items():
        print(f"{name:<22}{m['n']:>7}{m['precision']:>8.3f}"
              f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['accuracy']:>8.3f}")


def cmd_breakdown(args):
    path = args.dir / "test_predictions.jsonl"
    if not path.exists():
        print(f"No predictions at {path}. Run src.train first.")
        return

    rows = load_predictions(path)
    print(f"{len(rows)} test predictions from {args.dir.name}")
    print(f"overall: {prf(rows)}")

    # Directness is only meaningful on the positive class. A true negative has
    # directness 'not_applicable' by construction, so including those rows
    # would just measure performance on negatives twice.
    positives = [r for r in rows if r["label"] == 1]

    print_table("by directness (positives only)",
                group_by(positives, "directness"))
    print_table("by disclosure type (positives only)",
                group_by(positives, "disclosure_type"))
    print_table("by source community", group_by(rows, "source"))

    # Whether the annotator's confidence predicts classifier difficulty. If it
    # does, low-confidence labels are a candidate for exclusion or reweighting.
    banded = []
    for row in rows:
        conf = row.get("annotator_confidence")
        if conf is None:
            continue
        band = ("<0.6" if conf < 0.6 else
                "0.6-0.8" if conf < 0.8 else
                "0.8-0.95" if conf < 0.95 else ">=0.95")
        banded.append({**row, "band": band})
    if banded:
        print_table("by annotator confidence", group_by(banded, "band"))

    if args.plot:
        table = group_by(positives, "directness")
        if table:
            fig, ax = plt.subplots(figsize=(6, 4))
            names = list(table)
            ax.bar(names, [table[n]["recall"] for n in names],
                   color="#4a6fa5", edgecolor="black", linewidth=0.6)
            ax.set_ylabel("Recall")
            ax.set_ylim(0, 1)
            ax.set_title("Recall by directness of disclosure")
            for i, n in enumerate(names):
                ax.text(i, table[n]["recall"] + 0.02,
                        f"{table[n]['recall']:.2f}\nn={table[n]['n']}",
                        ha="center", fontsize=9)
            fig.tight_layout()
            out = args.out / "recall_by_directness.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=200)
            print(f"\nwrote {out}")


def collect_results(directory: Path) -> list[dict]:
    results = []
    for path in sorted(directory.glob("*/results.json")):
        record = json.loads(path.read_text())
        record["experiment"] = path.parent.name
        results.append(record)
    return results


def cmd_noise(args):
    results = [r for r in collect_results(args.dir)
               if r["experiment"].startswith("noise")]
    if not results:
        print(f"No noise experiments under {args.dir}")
        return

    results.sort(key=lambda r: r["noise_rate"])
    rates = [r["noise_rate"] for r in results]
    f1s = [r.get("test_macro_f1", 0) for r in results]

    print(f"\n{'noise':>8}{'macro F1':>11}{'change':>10}")
    baseline = f1s[0] if f1s else 0
    for rate, f1 in zip(rates, f1s):
        delta = f1 - baseline
        print(f"{rate:>8.0%}{f1:>11.3f}{delta:>+10.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r * 100 for r in rates], f1s, marker="o", color="#c0392b",
            linewidth=2, markersize=7)
    # Reference line at the clean-label baseline makes the degradation legible
    # as a distance rather than requiring the reader to compare two points.
    ax.axhline(baseline, linestyle="--", color="grey", linewidth=1,
               label="clean labels")
    ax.set_xlabel("Injected label noise (%)")
    ax.set_ylabel("Macro F1 on clean test set")
    ax.set_title("Classifier robustness to annotation error")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = args.out / "noise_robustness.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}")


def cmd_cost(args):
    results = [r for r in collect_results(args.dir)
               if r["experiment"].startswith("size")]
    if not results:
        print(f"No size experiments under {args.dir}")
        return

    print(f"\n{'model':<24}{'params M':>10}{'ms/post':>10}{'macro F1':>11}")
    for r in results:
        print(f"{r['model']:<24}{r.get('params_millions', 0):>10.1f}"
              f"{r.get('ms_per_post', 0):>10.2f}{r.get('test_macro_f1', 0):>11.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in results:
        ax.scatter(r.get("ms_per_post", 0), r.get("test_macro_f1", 0),
                   s=120, edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(r["model"].split("/")[-1],
                    (r.get("ms_per_post", 0), r.get("test_macro_f1", 0)),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    if args.llm_ms:
        ax.axvline(args.llm_ms, linestyle="--", color="#c0392b", linewidth=1.5,
                   label=f"LLM annotator ({args.llm_ms:.0f} ms/post)")
        ax.legend(loc="lower right")

    ax.set_xscale("log")
    ax.set_xlabel("Inference cost (ms per post, log scale)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Classification quality against inference cost")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    out = args.out / "quality_vs_cost.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}")


def cmd_gap(args):
    seen = args.dir / "gen_seen" / "results.json"
    unseen = args.dir / "gen_unseen" / "results.json"
    if not (seen.exists() and unseen.exists()):
        print("Need both gen_seen and gen_unseen results.")
        return

    a = json.loads(seen.read_text())
    b = json.loads(unseen.read_text())

    print(f"\n{'metric':<16}{'seen':>10}{'unseen':>10}{'gap':>10}")
    for metric in ["test_macro_f1", "test_f1", "test_precision", "test_recall"]:
        if metric in a and metric in b:
            print(f"{metric.replace('test_', ''):<16}{a[metric]:>10.3f}"
                  f"{b[metric]:>10.3f}{b[metric] - a[metric]:>+10.3f}")

    metrics = ["test_precision", "test_recall", "test_macro_f1"]
    labels = ["Precision", "Recall", "Macro F1"]
    x = range(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], [a.get(m, 0) for m in metrics],
           width, label="Seen communities", color="#4a6fa5",
           edgecolor="black", linewidth=0.6)
    ax.bar([i + width / 2 for i in x], [b.get(m, 0) for m in metrics],
           width, label="Held-out communities", color="#c98b5e",
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Generalisation to unseen communities")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    out = args.out / "generalisation_gap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("breakdown")
    p.add_argument("--dir", type=Path, required=True,
                   help="A single experiment directory")
    p.add_argument("--out", type=Path, default=Path("outputs/figures"))
    p.add_argument("--plot", action="store_true")
    p.set_defaults(func=cmd_breakdown)

    p = sub.add_parser("noise")
    p.add_argument("--dir", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--out", type=Path, default=Path("outputs/figures"))
    p.set_defaults(func=cmd_noise)

    p = sub.add_parser("cost")
    p.add_argument("--dir", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--out", type=Path, default=Path("outputs/figures"))
    p.add_argument("--llm-ms", type=float, default=None,
                   help="Measured LLM annotation latency in ms, for reference")
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("gap")
    p.add_argument("--dir", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--out", type=Path, default=Path("outputs/figures"))
    p.set_defaults(func=cmd_gap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
