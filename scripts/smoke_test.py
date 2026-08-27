"""
Prompt smoke test.

Runs the annotation pipeline against a small set of synthetic posts with known
labels. Use this to iterate on the prompt before touching the real dataset, and
to compare the four strategies on cases you already know the answer to.

The posts here are written for this test. None are real, so nothing sensitive is
committed to the repository.

    python -m scripts.smoke_test --strategy few_shot
    python -m scripts.smoke_test --strategy zero_shot --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import get_backend                       # noqa: E402
from src.parsing import Label, parse_response         # noqa: E402
from src.prompts import SYSTEM_PROMPT, STRATEGIES, build  # noqa: E402


# (text, expected_is_disclosure, expected_directness, note)
# The hard cases are the point. Anything can get the explicit ones right.
CASES = [
    (
        "Diagnosed with GAD last month. Still processing it honestly.",
        True, "explicit", "plain diagnosis, should be easy",
    ),
    (
        "Upped my sertraline to 100mg and the first fortnight was rough but "
        "it seems to be settling now.",
        True, "explicit", "medication named",
    ),
    (
        "I keep cancelling on people. Third time this month. They have stopped "
        "asking now which is probably fair enough.",
        True, "implicit", "withdrawal described, nothing named",
    ),
    (
        "Everyone says it gets easier and I keep waiting for the bit where it "
        "does.",
        True, "implicit", "hard: implied distress, very indirect",
    ),
    (
        "My dad was sectioned in the nineties and we still don't really talk "
        "about it.",
        False, "not_applicable", "third party, common false positive",
    ),
    (
        "Reminder that the crisis line is open 24/7 if anyone needs it tonight.",
        False, "not_applicable", "supportive, not disclosure",
    ),
    (
        "Does CBT actually work for anyone here? Considering it but sceptical.",
        True, "explicit", "help_seeking, borderline",
    ),
    (
        "New study out showing waiting lists have doubled since 2019.",
        False, "not_applicable", "news, no personal content",
    ),
    (
        "Been off work six weeks. GP signed me off again yesterday.",
        True, "implicit", "hard: absence without a named condition",
    ),
    (
        "Anyone got recommendations for a decent noise cancelling headset for "
        "the office?",
        False, "not_applicable", "unrelated, control case",
    ),
]


def evaluate(strategy: str, backend, verbose: bool = True) -> dict:
    correct_disclosure = 0
    correct_directness = 0
    failures = 0
    rows = []

    for text, expected_disc, expected_direct, note in CASES:
        generation = backend.generate(SYSTEM_PROMPT, build(strategy, text))
        parsed = parse_response(generation.text)

        if isinstance(parsed, Label):
            disc_ok = parsed.is_disclosure == expected_disc
            direct_ok = parsed.directness == expected_direct
            correct_disclosure += disc_ok
            correct_directness += direct_ok
            rows.append((text, parsed, disc_ok, direct_ok, note,
                         generation.latency_s))
        else:
            failures += 1
            rows.append((text, parsed, False, False, note, generation.latency_s))

    n = len(CASES)
    summary = {
        "strategy": strategy,
        "disclosure_acc": round(correct_disclosure / n, 3),
        "directness_acc": round(correct_directness / n, 3),
        "parse_failures": failures,
        "mean_latency_s": round(
            sum(r[5] for r in rows) / n, 3
        ),
    }

    if verbose:
        print(f"\n{'=' * 78}\nSTRATEGY: {strategy}\n{'=' * 78}")
        for text, parsed, disc_ok, direct_ok, note, latency in rows:
            mark = "PASS" if disc_ok and direct_ok else ("PART" if disc_ok else "FAIL")
            print(f"\n[{mark}] {text[:66]}...")
            print(f"       expected: {note}")
            if isinstance(parsed, Label):
                print(f"       got: disclosure={parsed.is_disclosure} "
                      f"type={parsed.disclosure_type} "
                      f"directness={parsed.directness} "
                      f"conf={parsed.confidence}")
            else:
                print(f"       PARSE FAILURE: {parsed.reason}")
        print(f"\n{summary}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="few_shot", choices=sorted(STRATEGIES))
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--compare", action="store_true",
                        help="Run all four strategies and print a comparison.")
    args = parser.parse_args()

    kwargs = {"model": args.model} if args.backend == "ollama" else {"model_id": args.model}
    backend = get_backend(args.backend, **kwargs)

    if args.compare:
        results = [evaluate(s, backend) for s in sorted(STRATEGIES)]
        print(f"\n{'=' * 78}\nCOMPARISON\n{'=' * 78}")
        print(f"{'strategy':<16}{'disclosure':>12}{'directness':>12}"
              f"{'failures':>10}{'latency':>10}")
        for r in results:
            print(f"{r['strategy']:<16}{r['disclosure_acc']:>12}"
                  f"{r['directness_acc']:>12}{r['parse_failures']:>10}"
                  f"{r['mean_latency_s']:>10}")
    else:
        evaluate(args.strategy, backend)


if __name__ == "__main__":
    main()
