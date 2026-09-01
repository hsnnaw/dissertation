"""
Does the headline result depend on the two communities that dominate the corpus?

Sampling was proportional to community size, so r/legaladvice and
r/personalfinance together make up about a third of the corpus, and both are
near-zero disclosure. A reader can reasonably ask whether performance is
carried by those easy negatives.

Rebalancing would mean resampling and reannotating. This answers the question
without that: recompute performance with those communities removed, and again
with every community weighted equally rather than by size. If the numbers hold,
the imbalance is a description of the corpus rather than a confound.

    python -m scripts.sensitivity \\
        --predictions outputs/experiments/gen_seen/test_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DOMINANT = ("legaladvice", "personalfinance")


def prf(rows: list[dict]) -> dict:
    tp = sum(r["label"] == 1 and r["pred"] == 1 for r in rows)
    fp = sum(r["label"] == 0 and r["pred"] == 1 for r in rows)
    fn = sum(r["label"] == 1 and r["pred"] == 0 for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(rows),
            "positive_rate": round(sum(x["label"] for x in rows) / len(rows), 4)
            if rows else 0.0,
            "precision": round(p, 4), "recall": round(r_, 4),
            "f1": round(2 * p * r_ / (p + r_), 4) if p + r_ else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=Path("outputs/experiments/gen_seen/test_predictions.jsonl"))
    parser.add_argument("--exclude", nargs="*", default=list(DOMINANT))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/sensitivity.json"))
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.predictions.read_text().splitlines()
            if l.strip()]
    by_community = defaultdict(list)
    for r in rows:
        by_community[r.get("source", "?")].append(r)

    result = {"all": prf(rows)}
    print(f"{'condition':<34}{'n':>6}{'pos rate':>10}{'F1':>9}"
          f"{'precision':>11}{'recall':>9}")
    a = result["all"]
    print(f"{'all communities':<34}{a['n']:>6}{a['positive_rate']:>10.1%}"
          f"{a['f1']:>9.4f}{a['precision']:>11.4f}{a['recall']:>9.4f}")

    kept = [r for r in rows if r.get("source") not in set(args.exclude)]
    result["excluding_dominant"] = prf(kept)
    result["excluded"] = list(args.exclude)
    e = result["excluding_dominant"]
    label = f"excluding {', '.join(args.exclude)}"
    print(f"{label:<34}{e['n']:>6}{e['positive_rate']:>10.1%}"
          f"{e['f1']:>9.4f}{e['precision']:>11.4f}{e['recall']:>9.4f}")

    # Macro average over communities: every community counts once, whatever
    # its size, which is the comparison the proportional sample cannot make.
    per = {c: prf(rs) for c, rs in by_community.items() if len(rs) >= 10}
    if per:
        macro = {k: round(sum(v[k] for v in per.values()) / len(per), 4)
                 for k in ("f1", "precision", "recall")}
        result["macro_over_communities"] = {"n_communities": len(per), **macro}
        print(f"{'macro avg over communities':<34}{len(per):>6}{'—':>10}"
              f"{macro['f1']:>9.4f}{macro['precision']:>11.4f}"
              f"{macro['recall']:>9.4f}")

    result["per_community"] = per
    print(f"\nper community (n >= 10)")
    print(f"  {'community':<18}{'n':>6}{'pos rate':>10}{'F1':>9}")
    for c, v in sorted(per.items(), key=lambda kv: -kv[1]["positive_rate"]):
        print(f"  {c:<18}{v['n']:>6}{v['positive_rate']:>10.1%}{v['f1']:>9.4f}")

    drop = result["all"]["f1"] - result["excluding_dominant"]["f1"]
    print(f"\nRemoving {' and '.join(args.exclude)} moves F1 by {drop:+.4f}.")
    print("Against a run-to-run standard deviation of about 0.011, a move")
    print("smaller than that is not evidence the result depends on them.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
