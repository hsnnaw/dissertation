"""
Evaluate annotation prompt strategies against the synthetic benchmark.

Reports overall accuracy, a per-slice breakdown, separate figures for clear and
contested cases, directness accuracy on true positives only, and the specific
cases each strategy gets wrong.

    python -m scripts.evaluate_prompts --strategy few_shot
    python -m scripts.evaluate_prompts --compare
    python -m scripts.evaluate_prompts --compare --repeats 3

Why the per-slice breakdown matters more than the headline number: a strategy
that scores 0.85 by handling every explicit case and failing every implicit one
is useless for this project, and a single aggregate figure hides that
completely.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_cases import CASES, summary      # noqa: E402
from src.llm import get_backend                          # noqa: E402
from src.parsing import Label, parse_response            # noqa: E402
from src.prompts import SYSTEM_PROMPT, STRATEGIES, build  # noqa: E402


def evaluate(strategy: str, backend, verbose: bool = False) -> dict:
    per_slice = defaultdict(lambda: {"n": 0, "correct": 0})
    errors = []

    n = len(CASES)
    correct = correct_clear = correct_contested = 0
    n_clear = n_contested = 0
    directness_n = directness_correct = 0
    type_n = type_correct = 0
    failures = 0
    latencies = []

    # Confusion counts for the disclosure judgement.
    tp = fp = tn = fn = 0

    for case in CASES:
        generation = backend.generate(SYSTEM_PROMPT, build(strategy, case["text"]))
        parsed = parse_response(generation.text)
        latencies.append(generation.latency_s)

        slice_name = case["slice"]
        per_slice[slice_name]["n"] += 1

        if case["contested"]:
            n_contested += 1
        else:
            n_clear += 1

        if not isinstance(parsed, Label):
            failures += 1
            errors.append({
                "text": case["text"][:70],
                "slice": slice_name,
                "contested": case["contested"],
                "expected": case["disclosure"],
                "got": f"PARSE FAILURE: {parsed.reason}",
                "note": case["note"],
            })
            # A parse failure counts as wrong for the disclosure judgement, but
            # is not folded into the confusion matrix, which describes decisions
            # the model actually made.
            continue

        got = parsed.is_disclosure
        expected = case["disclosure"]
        ok = got == expected

        if ok:
            correct += 1
            per_slice[slice_name]["correct"] += 1
            if case["contested"]:
                correct_contested += 1
            else:
                correct_clear += 1
        else:
            errors.append({
                "text": case["text"][:70],
                "slice": slice_name,
                "contested": case["contested"],
                "expected": f"{expected}/{case['dtype']}/{case['directness']}",
                "got": f"{got}/{parsed.disclosure_type}/{parsed.directness}",
                "note": case["note"],
            })

        if expected and got:
            tp += 1
        elif not expected and got:
            fp += 1
        elif not expected and not got:
            tn += 1
        else:
            fn += 1

        # Directness and type are only meaningful where both agree a disclosure
        # is present. Scoring them on true negatives would inflate both, since
        # not_applicable/none is the trivially correct answer there.
        if expected and got:
            directness_n += 1
            directness_correct += parsed.directness == case["directness"]
            type_n += 1
            type_correct += parsed.disclosure_type == case["dtype"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result = {
        "strategy": strategy,
        "disclosure_acc": round(correct / n, 3),
        "clear_acc": round(correct_clear / n_clear, 3) if n_clear else None,
        "contested_acc": round(correct_contested / n_contested, 3) if n_contested else None,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "directness_acc": round(directness_correct / directness_n, 3) if directness_n else None,
        "type_acc": round(type_correct / type_n, 3) if type_n else None,
        "parse_failures": failures,
        "mean_latency_s": round(sum(latencies) / n, 2),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_slice": {
            name: round(v["correct"] / v["n"], 3)
            for name, v in sorted(per_slice.items())
        },
        "errors": errors,
    }

    if verbose:
        print(f"\n{'=' * 78}\n{strategy}\n{'=' * 78}")
        print(f"disclosure {result['disclosure_acc']}  "
              f"clear {result['clear_acc']}  contested {result['contested_acc']}  "
              f"P {result['precision']}  R {result['recall']}  F1 {result['f1']}")
        print(f"directness {result['directness_acc']}  type {result['type_acc']}  "
              f"failures {failures}  latency {result['mean_latency_s']}s")
        print(f"confusion {result['confusion']}")

        print("\nper slice:")
        for name, acc in result["per_slice"].items():
            bar = "#" * int(acc * 20)
            print(f"  {name:22} {acc:5.2f}  {bar}")

        if errors:
            print(f"\nerrors ({len(errors)}):")
            for err in errors:
                flag = "contested" if err["contested"] else "CLEAR   "
                print(f"  [{flag}] {err['slice']}")
                print(f"    {err['text']}...")
                print(f"    expected {err['expected']}  got {err['got']}")
                print(f"    ({err['note']})")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="few_shot", choices=sorted(STRATEGIES))
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--compare", action="store_true",
                        help="Run every strategy and print a comparison.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run each strategy N times. Temperature is 0, so "
                             "variation across repeats indicates nondeterminism "
                             "in the backend rather than sampling.")
    parser.add_argument("--out", type=Path, default=Path("outputs/prompt_eval.json"))
    args = parser.parse_args()

    kwargs = {"model": args.model} if args.backend == "ollama" else {"model_id": args.model}
    backend = get_backend(args.backend, **kwargs)

    print(json.dumps(summary(), indent=2))

    strategies = sorted(STRATEGIES) if args.compare else [args.strategy]
    all_results = []

    for strategy in strategies:
        for run in range(args.repeats):
            label = strategy if args.repeats == 1 else f"{strategy} (run {run + 1})"
            print(f"\nrunning {label}...")
            result = evaluate(strategy, backend, verbose=True)
            result["run"] = run + 1
            all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'=' * 78}\nCOMPARISON\n{'=' * 78}")
        header = (f"{'strategy':<15}{'run':>4}{'overall':>9}{'clear':>8}"
                  f"{'contest':>9}{'F1':>7}{'direct':>8}{'fail':>6}{'sec':>7}")
        print(header)
        for r in all_results:
            print(f"{r['strategy']:<15}{r['run']:>4}{r['disclosure_acc']:>9}"
                  f"{r['clear_acc']:>8}{r['contested_acc']:>9}{r['f1']:>7}"
                  f"{str(r['directness_acc']):>8}{r['parse_failures']:>6}"
                  f"{r['mean_latency_s']:>7}")

        print("\nper-slice accuracy:")
        slices = sorted(all_results[0]["per_slice"])
        print(f"{'slice':<22}" + "".join(f"{r['strategy'][:11]:>13}" for r in all_results))
        for name in slices:
            row = "".join(f"{r['per_slice'].get(name, 0):>13.2f}" for r in all_results)
            print(f"{name:<22}{row}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"benchmark": summary(), "results": all_results}, indent=2
    ))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
