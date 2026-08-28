"""
Dataset splitting.

Two split modes, and the choice between them is itself part of the research
design rather than a detail.

  random    - stratified by label, subreddits mixed across splits. Measures how
              well the model does on posts from communities it has seen.

  subreddit - whole subreddits held out. Measures whether the model learned
              disclosure or merely learned the house style of particular
              communities. This is the harder and more honest test, and the
              gap between the two is the cross-corpus generalisation result.

Splitting happens on the annotated file, after labels exist, because
stratification needs them.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_labelled(
    path: Path,
    drop_failures: bool = True,
    posts_path: Path | None = None,
) -> list[dict]:
    """
    Read annotation output. Parse failures carry an 'error' key instead of a
    label and are dropped from training by default, but their count is reported
    because the failure rate is a result in its own right.

    The annotation output deliberately does not carry post text, so training
    needs it joined back from the posts file by post_id. Pass posts_path to do
    that. Records whose text cannot be found are dropped and counted, since a
    record with no text cannot be trained on.
    """
    records, failures = [], 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "error" in record:
                failures += 1
                if drop_failures:
                    continue
            records.append(record)

    if failures:
        rate = failures / (len(records) + failures)
        print(f"  {failures} parse failures ({rate:.1%}) "
              f"{'dropped' if drop_failures else 'kept'}")

    if posts_path is not None:
        text_by_id = {}
        with Path(posts_path).open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                post = json.loads(line)
                text_by_id[str(post["post_id"])] = post["text"]

        joined, missing = [], 0
        for record in records:
            text = text_by_id.get(str(record.get("post_id", "")))
            if text is None:
                missing += 1
                continue
            joined.append({**record, "text": text})

        if missing:
            print(f"  {missing} records dropped, no matching post text")
        records = joined

    return records


def stratified_split(
    records: list[dict],
    label_key: str = "is_disclosure",
    train: float = 0.7,
    val: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Random split, stratified so each split holds the same class balance."""
    rng = random.Random(seed)
    by_label: dict = {}
    for record in records:
        by_label.setdefault(record[label_key], []).append(record)

    splits = {"train": [], "val": [], "test": []}
    for group in by_label.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train)
        n_val = int(n * (train + val))
        splits["train"] += shuffled[:n_train]
        splits["val"] += shuffled[n_train:n_val]
        splits["test"] += shuffled[n_val:]

    for split in splits.values():
        rng.shuffle(split)
    return splits


def subreddit_split(
    records: list[dict],
    held_out: list[str] | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Hold out whole subreddits for test. If none are named, the smallest
    communities making up roughly 20% of the data are used, which keeps the
    training set large while still testing on genuinely unseen communities.
    """
    rng = random.Random(seed)
    sizes = Counter(r.get("source", "") for r in records)

    if held_out is None:
        target = len(records) * 0.2
        held_out, running = [], 0
        for name, size in sorted(sizes.items(), key=lambda kv: kv[1]):
            if running >= target:
                break
            held_out.append(name)
            running += size

    held = set(held_out)
    print(f"  holding out: {sorted(held)}")

    test = [r for r in records if r.get("source", "") in held]
    remainder = [r for r in records if r.get("source", "") not in held]
    rng.shuffle(remainder)

    cut = int(len(remainder) * (1 - val_fraction))
    return {"train": remainder[:cut], "val": remainder[cut:], "test": test}


def summarise(splits: dict[str, list[dict]], label_key: str) -> None:
    print(f"\n{'split':<8}{'n':>8}{'positive':>11}{'rate':>8}{'subreddits':>12}")
    for name, records in splits.items():
        if not records:
            continue
        positive = sum(bool(r.get(label_key)) for r in records)
        subs = len({r.get("source", "") for r in records})
        print(f"{name:<8}{len(records):>8}{positive:>11}"
              f"{positive / len(records):>8.1%}{subs:>12}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Annotated JSONL from src.annotate")
    parser.add_argument("--posts", type=Path, required=True,
                        help="Posts JSONL the annotations came from. Training "
                             "needs the text, which the annotations do not carry.")
    parser.add_argument("--outdir", type=Path,
                        default=Path("data/processed/splits"))
    parser.add_argument("--mode", choices=["random", "subreddit"],
                        default="random")
    parser.add_argument("--held-out", nargs="*", default=None,
                        help="Subreddits to hold out (subreddit mode only)")
    parser.add_argument("--label-key", default="is_disclosure")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"reading {args.input}")
    records = load_labelled(args.input, posts_path=args.posts)
    print(f"  {len(records)} labelled records")

    if args.mode == "random":
        splits = stratified_split(records, args.label_key, seed=args.seed)
    else:
        splits = subreddit_split(records, args.held_out, seed=args.seed)

    summarise(splits, args.label_key)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, group in splits.items():
        path = args.outdir / f"{name}.jsonl"
        with path.open("w") as handle:
            for record in group:
                handle.write(json.dumps(record) + "\n")
    print(f"\nwrote splits to {args.outdir}")


if __name__ == "__main__":
    main()
