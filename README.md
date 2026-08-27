# Mental Health Self-Disclosure Detection

MSc Data Science dissertation (COMM070), University of Surrey.

Distilling a lightweight classifier for mental health self-disclosure from a
locally-run LLM annotator, and studying when that distillation works.

## Approach

1. Take an existing published Reddit dataset from a repository (no scraping).
2. Annotate it for self-disclosure using a locally-hosted Llama model.
3. Fine-tune RoBERTa on those labels.
4. Run experiments on when and why the distillation holds up.

Everything runs locally. No post text is sent to any external API, which is the
condition the ethics route depends on.

## Ethics

Uses only secondary data from a public repository. Not scraped from social media
feeds. Covered by the Surrey Secondary Data Checklist rather than full Ethics RM
review, subject to supervisor sign-off.

Constraints observed throughout:

- No verbatim post content in the dissertation, any output, or this repository.
  Reddit text is reverse-searchable, so a real quote de-anonymises the author
  regardless of what else has been stripped.
- No aggregation of posts by author.
- No attempt to identify any individual.
- Raw data is gitignored and never committed.
- All prompt exemplars and test cases are synthetic, written for this project.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For local annotation via Ollama (easiest, good for prompt iteration):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
```

For the full annotation run on a GPU, use the transformers backend instead.
An 8B model in 4-bit fits on a single 16GB card.

## Usage

Iterate on the prompt against known cases before touching real data:

```bash
python -m scripts.smoke_test --strategy few_shot
python -m scripts.smoke_test --compare        # all four strategies
```

Annotate:

```bash
python -m src.annotate \
    --input data/interim/posts.jsonl \
    --output data/processed/labels_few_shot.jsonl \
    --strategy few_shot \
    --backend transformers \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 16
```

Resumable. Rerun the same command after an interruption and it picks up where it
stopped.

## Label schema

| Field | Values |
|---|---|
| `is_disclosure` | true / false |
| `disclosure_type` | diagnosis, symptom, treatment, help_seeking, recovery, none |
| `directness` | explicit, implicit, not_applicable |
| `confidence` | 0.0 to 1.0 |

`directness` is separate from `disclosure_type` deliberately. A post can be a
clear disclosure that is nonetheless implicit, and those are precisely the cases
where a lexical baseline fails. Keeping the axis separate makes that analysis
possible.

Third-party mentions are not disclosures. "My brother has bipolar" reveals
nothing sensitive about the author, and conflating the two is a common source of
label noise.

## Layout

```
src/
  prompts.py    label taxonomy, codebook, four prompt strategies
  parsing.py    response parsing, schema validation, no silent defaults
  llm.py        Ollama and transformers backends behind one interface
  annotate.py   resumable annotation runner
scripts/
  smoke_test.py prompt evaluation against synthetic cases
data/
  raw/          downloaded dataset (gitignored)
  interim/      normalised posts (gitignored)
  processed/    labels (gitignored)
```

## Planned experiments

- **Label noise robustness.** Inject known error rates into training labels and
  measure downstream degradation. Establishes how much label quality actually
  matters, rather than assuming it.
- **Cross-corpus generalisation.** Train on one subreddit or dataset, evaluate on
  held-out ones. Tests whether the model learned disclosure or learned a corpus.
- **Quality against cost.** RoBERTa-base versus distilled variants, plotted
  against measured inference cost. Makes the deployment argument empirical.
- **Annotation strategy comparison.** Which of the four prompt strategies
  produces labels that yield the best downstream classifier. Not obvious in
  advance.
- **Agreement study.** Self-conducted labelling of a stratified sample, reported
  as agreement with the LLM labels, identifying which disclosure types are least
  reliably labelled.
