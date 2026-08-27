"""
Dataset loading and normalisation.

Takes the Low et al. (2020) Reddit Mental Health Dataset, which ships as one CSV
per subreddit per time window, and normalises it to a single JSONL of posts
ready for annotation.

Anonymisation happens here, at the boundary. Author fields are dropped and never
written, and identifier-shaped strings are scrubbed from post text. Downstream
code never sees an author, so it cannot accidentally aggregate by one.

Post ids are hashed rather than carried through. The original Reddit id would let
anyone reconstruct the source post from a published dataset, which defeats the
anonymisation commitment made in the ethics checklist.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


# Fields the Low et al. CSVs carry that must not survive normalisation.
DROP_FIELDS = {"author", "author_fullname", "permalink", "url", "id", "name"}

# Identifier-shaped patterns scrubbed from post text before anything downstream
# sees it. Deliberately blunt: over-scrubbing costs a little signal, and
# under-scrubbing costs anonymity.
SCRUBBERS = [
    (re.compile(r"\bu/[A-Za-z0-9_-]+"), "[USER]"),
    (re.compile(r"\b/u/[A-Za-z0-9_-]+"), "[USER]"),
    (re.compile(r"https?://\S+"), "[URL]"),
    (re.compile(r"\bwww\.\S+"), "[URL]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    # UK and international phone shapes
    (re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\(?\d{3,5}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}\b"),
     "[PHONE]"),
]

# Markdown quote lines. A post that is mostly quoted text is someone else's
# disclosure, not the author's, and including it inflates the positive class
# with mislabelled examples.
QUOTE_LINE = re.compile(r"^\s*>", re.MULTILINE)


def scrub(text: str) -> str:
    """Remove identifier-shaped strings from post text."""
    for pattern, replacement in SCRUBBERS:
        text = pattern.sub(replacement, text)
    return text


def hash_id(raw: str, salt: str = "comm070") -> str:
    """
    Stable pseudonymous id. Deterministic so reruns line up, but not reversible
    to the original Reddit post.
    """
    return hashlib.sha256(f"{salt}:{raw}".encode()).hexdigest()[:16]


def quote_ratio(text: str) -> float:
    """Fraction of non-empty lines that are markdown quotes."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return sum(bool(QUOTE_LINE.match(line)) for line in lines) / len(lines)


def is_usable(
    text: str,
    min_chars: int = 120,
    max_chars: int = 6000,
    max_quote_ratio: float = 0.5,
) -> tuple[bool, str]:
    """
    Whether a post is worth annotating. Returns (keep, reason_if_dropped).

    Length bounds matter for two different reasons. Very short posts rarely
    contain enough context to judge disclosure, so they mostly produce
    low-confidence noise. Very long posts blow up annotation latency and get
    truncated by RoBERTa's 512-token window anyway.
    """
    if not text or not text.strip():
        return False, "empty"

    stripped = text.strip()
    lowered = stripped.lower()

    if lowered in {"[deleted]", "[removed]"}:
        return False, "deleted"
    if len(stripped) < min_chars:
        return False, "too_short"
    if len(stripped) > max_chars:
        return False, "too_long"
    if quote_ratio(stripped) > max_quote_ratio:
        return False, "mostly_quoted"

    return True, ""


def load_csv_directory(
    directory: Path,
    text_columns: tuple[str, ...] = ("post", "selftext", "body", "text"),
    title_column: str = "title",
) -> pd.DataFrame:
    """
    Read every CSV in a directory into one frame.

    The Low et al. files name the text column differently across releases, so
    this probes a list of candidates rather than assuming one.
    """
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {directory}")

    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)

        text_col = next((c for c in text_columns if c in frame.columns), None)
        if text_col is None:
            print(f"  skipping {path.name}: no recognised text column "
                  f"(has {list(frame.columns)[:6]}...)")
            continue

        # Subreddit comes from the filename, which is how the dataset is
        # organised. Strip any trailing date window.
        subreddit = re.sub(r"_(pre|post)?_?\d{4}.*$", "", path.stem)

        body = frame[text_col].fillna("").astype(str)

        # Removal markers must be checked on the body alone. Prepending the
        # title first would turn "[deleted]" into "Some title\n\n[deleted]",
        # which no exact match will catch, and the post would survive into the
        # annotation set as a title with no content.
        body = body.mask(
            body.str.strip().str.lower().isin(["[deleted]", "[removed]"]),
            "",
        )

        if title_column in frame.columns:
            title = frame[title_column].fillna("").astype(str)
            combined = (title + "\n\n" + body).str.strip()
            # Title-only rows are what remains after a body is blanked above.
            combined = combined.mask(body.str.strip() == "", "")
        else:
            combined = body

        frames.append(pd.DataFrame({
            "raw_id": frame.get("id", pd.Series(range(len(frame)))).astype(str),
            "text": combined,
            "subreddit": subreddit,
            "source_file": path.name,
        }))

    if not frames:
        raise ValueError(f"No usable CSVs in {directory}")

    return pd.concat(frames, ignore_index=True)


def normalise(
    frame: pd.DataFrame,
    min_chars: int = 120,
    max_chars: int = 6000,
) -> tuple[pd.DataFrame, dict]:
    """Scrub, filter, deduplicate. Returns the clean frame and drop counts."""
    counts = {"input": len(frame)}

    frame = frame.copy()
    frame["text"] = frame["text"].map(scrub)

    reasons = frame["text"].map(
        lambda t: is_usable(t, min_chars, max_chars)[1]
    )
    for reason in ["empty", "deleted", "too_short", "too_long", "mostly_quoted"]:
        counts[f"dropped_{reason}"] = int((reasons == reason).sum())

    frame = frame[reasons == ""].copy()

    # Exact-duplicate text appears across subreddits (crossposts) and within
    # them (reposts). Keeping both would leak between train and test splits.
    before = len(frame)
    frame["text_hash"] = frame["text"].map(
        lambda t: hashlib.sha256(t.strip().encode()).hexdigest()
    )
    frame = frame.drop_duplicates(subset="text_hash", keep="first")
    counts["dropped_duplicate"] = before - len(frame)

    frame["post_id"] = frame["raw_id"].map(hash_id)
    frame = frame.drop(columns=["raw_id", "text_hash"])

    counts["output"] = len(frame)
    return frame.reset_index(drop=True), counts


def write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare(
    input_dir: Path,
    output_path: Path,
    min_chars: int = 120,
    max_chars: int = 6000,
    sample: int | None = None,
    seed: int = 42,
) -> dict:
    """Full pipeline: CSV directory to annotation-ready JSONL."""
    print(f"reading CSVs from {input_dir}")
    frame = load_csv_directory(input_dir)
    print(f"  {len(frame)} rows across {frame['subreddit'].nunique()} subreddits")

    frame, counts = normalise(frame, min_chars, max_chars)

    if sample and sample < len(frame):
        # Stratify by subreddit so a sample keeps the community mix. A plain
        # random sample would over-represent whichever subreddit is largest.
        frame = (
            frame.groupby("subreddit", group_keys=False)
            .apply(lambda g: g.sample(
                max(1, round(sample * len(g) / counts["output"])),
                random_state=seed,
            ))
            .reset_index(drop=True)
        )
        counts["sampled"] = len(frame)

    write_jsonl(frame, output_path)
    print(f"\nwrote {len(frame)} posts to {output_path}")
    print(json.dumps(counts, indent=2))
    return counts


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of dataset CSVs")
    parser.add_argument("--output", type=Path,
                        default=Path("data/interim/posts.jsonl"))
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--sample", type=int, default=None,
                        help="Stratified sample of N posts. Omit to keep all.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare(
        input_dir=args.input_dir,
        output_path=args.output,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        sample=args.sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
