"""
Annotation runner.

Reads posts, labels them with a local model, writes JSONL. Resumable, because
annotating tens of thousands of posts on one GPU takes hours and losing a run to
a dropped connection or an OOM at hour three is avoidable.

Parse failures are written to the output with an error field rather than being
dropped. The failure rate is itself a result worth reporting: it tells you how
reliably the model follows the codebook, and it differs by prompt strategy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from .llm import get_backend
from .parsing import Label, ParseFailure, parse_response
from .prompts import SYSTEM_PROMPT, build


def load_done_ids(path: Path) -> set[str]:
    """Post ids already present in the output, so a rerun resumes."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["post_id"])
            except (json.JSONDecodeError, KeyError):
                # A partial final line from an interrupted run. Skip it; the
                # post gets re-annotated, which is harmless.
                continue
    return done


def load_posts(path: Path, text_field: str, id_field: str) -> list[dict]:
    posts = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            posts.append({"post_id": str(record[id_field]),
                          "text": record[text_field],
                          "source": record.get("subreddit", "")})
    return posts


def run(
    input_path: Path,
    output_path: Path,
    strategy: str = "few_shot",
    backend_kind: str = "ollama",
    model: str | None = None,
    batch_size: int = 8,
    limit: int | None = None,
    text_field: str = "text",
    id_field: str = "post_id",
) -> dict:
    backend_kwargs = {"model": model} if model else {}
    if backend_kind == "transformers" and model:
        backend_kwargs = {"model_id": model}
    backend = get_backend(backend_kind, **backend_kwargs)

    posts = load_posts(input_path, text_field, id_field)
    if limit:
        posts = posts[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(output_path)
    pending = [p for p in posts if p["post_id"] not in done]

    print(f"{len(posts)} posts, {len(done)} already annotated, {len(pending)} to do")
    print(f"backend={backend.name} strategy={strategy}")

    stats = {"ok": 0, "failed": 0, "total_latency_s": 0.0}

    with output_path.open("a") as out:
        for start in tqdm(range(0, len(pending), batch_size), desc="annotating"):
            chunk = pending[start:start + batch_size]
            prompts = [build(strategy, p["text"]) for p in chunk]
            generations = backend.generate_batch(SYSTEM_PROMPT, prompts)

            for post, generation in zip(chunk, generations):
                parsed = parse_response(generation.text)
                record = {
                    "post_id": post["post_id"],
                    "source": post["source"],
                    "strategy": strategy,
                    "model": backend.name,
                    "latency_s": round(generation.latency_s, 4),
                    "output_tokens": generation.output_tokens,
                }
                if isinstance(parsed, Label):
                    record.update(parsed.to_dict())
                    stats["ok"] += 1
                else:
                    record.update(parsed.to_dict())
                    stats["failed"] += 1

                stats["total_latency_s"] += generation.latency_s
                out.write(json.dumps(record) + "\n")
            out.flush()

    total = stats["ok"] + stats["failed"]
    if total:
        stats["failure_rate"] = round(stats["failed"] / total, 4)
        stats["mean_latency_s"] = round(stats["total_latency_s"] / total, 4)
    print(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL of posts")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSONL of labels (appended to, resumable)")
    parser.add_argument("--strategy", default="few_shot",
                        choices=["zero_shot", "few_shot", "cot", "few_shot_cot"])
    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "transformers"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Annotate only the first N posts. Use for smoke tests.")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="post_id")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        strategy=args.strategy,
        backend_kind=args.backend,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        text_field=args.text_field,
        id_field=args.id_field,
    )


if __name__ == "__main__":
    main()
