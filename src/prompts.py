"""
Annotation prompts and label taxonomy for mental health self-disclosure detection.

The taxonomy separates three orthogonal judgements:

  1. is_disclosure   - does the author disclose their OWN mental health experience?
  2. disclosure_type - what kind of disclosure is it?
  3. directness      - is it stated explicitly, or implied?

Keeping directness separate is what makes the "indirect disclosure" analysis
possible later. A post can be a clear disclosure that is nonetheless implicit
("I have not left the flat in three weeks and cannot see the point"), and those
are exactly the cases where a lexical baseline is expected to fail.

Third-party mentions are deliberately NOT disclosures. "My brother has bipolar"
tells us nothing sensitive about the author, and conflating the two is a common
source of label noise.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Label space
# --------------------------------------------------------------------------

DISCLOSURE_TYPES = [
    "diagnosis",       # names a condition they have / were diagnosed with
    "symptom",         # describes their own symptoms or mental state over time
    "treatment",       # therapy, medication, hospitalisation, support services
    "help_seeking",    # asking for advice or support for their own situation
    "recovery",        # describes their own progress, relapse, or coping
    "none",            # no self-disclosure of the author's mental health
]

DIRECTNESS = [
    "explicit",        # stated outright, would be caught by keyword matching
    "implicit",        # conveyed through description, narrative, or implication
    "not_applicable",  # only when is_disclosure is false
]

LABEL_SCHEMA = {
    "is_disclosure": "boolean",
    "disclosure_type": DISCLOSURE_TYPES,
    "directness": DIRECTNESS,
    "confidence": "float between 0 and 1",
}


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert annotator working on a research dataset about self-disclosure \
in online mental health communities. You label posts according to a fixed \
codebook. You are careful, consistent, and conservative: when a post is \
genuinely ambiguous you say so through the confidence score rather than \
guessing decisively.

You never editorialise, never give advice, and never comment on the wellbeing of \
the author. You output only the requested JSON object and nothing else.\
"""


# --------------------------------------------------------------------------
# Codebook (shared by all prompt variants)
# --------------------------------------------------------------------------

CODEBOOK = """\
DEFINITION
A mental health self-disclosure is a passage in which the AUTHOR reveals \
something about THEIR OWN mental health experience: a condition, symptoms, \
treatment, help-seeking, or recovery.

NOT a self-disclosure:
- Discussing someone else's mental health ("my sister has anxiety")
- General or abstract discussion of mental health as a topic
- News, statistics, research, or advocacy with no personal content
- Quoting or reacting to someone else's disclosure without adding their own

FIELDS

is_disclosure (true / false)
  true if the author reveals their own mental health experience.

disclosure_type (one of: diagnosis, symptom, treatment, help_seeking, recovery, none)
  diagnosis     names a condition they have or were diagnosed with
  symptom       describes their own mental state, symptoms, or experience
  treatment     therapy, medication, hospitalisation, or support services they use
  help_seeking  asks for advice or support for their own situation, including
                asking about a treatment they are personally considering
  recovery      describes their own progress, relapse, setback, or coping
  none          use only when is_disclosure is false
  If several apply, choose the one carrying the most sensitive information,
  using this precedence: diagnosis > treatment > symptom > recovery > help_seeking.

directness (one of: explicit, implicit, not_applicable)
  explicit        the post NAMES a condition, medication, therapy type, or
                  diagnosis applied to the author (e.g. "diagnosed with GAD",
                  "upped my sertraline", "I am depressed")
  implicit        the disclosure is conveyed through description, narrative, or
                  consequence without naming what it is about (e.g. "I have not
                  left the flat in three weeks and I cannot see the point in any
                  of it", "signed off work again")
  not_applicable  use only when is_disclosure is false

confidence (number between 0 and 1)
  Your confidence in the is_disclosure judgement. Use values below 0.7 for
  genuinely ambiguous cases. Do not default to 1.0.

EDGE CASES
- Asking about a treatment, therapy, or medication while personally
  considering it IS help_seeking. The author reveals something about their
  own situation by asking. A purely abstract question with no personal
  stake is not.
- Mark explicit only when the post NAMES a condition, medication, therapy
  type, or diagnosis applied to the author. Describing a consequence
  (time off work, a GP visit, cancelled plans) without naming what it is
  about is implicit.
- If the post does not make clear that the topic is mental health rather
  than physical health, set confidence below 0.6.

OUTPUT
Return a single JSON object with exactly these four keys and no other text:
{"is_disclosure": <bool>, "disclosure_type": "<string>", "directness": "<string>", "confidence": <float>}\
"""


# --------------------------------------------------------------------------
# Few-shot exemplars
#
# These are SYNTHETIC. They were written for this codebook and are not taken
# from any real post. This matters: putting real user text in a prompt that is
# stored in the repository would defeat the anonymisation guarantees made in the
# ethics checklist.
# --------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    (
        "Finally got the referral through after eight months on the list. "
        "Starting CBT next week and honestly I have no idea what to expect.",
        {"is_disclosure": True, "disclosure_type": "treatment",
         "directness": "explicit", "confidence": 0.95},
    ),
    (
        "Does anyone else find that the days just blur into each other? "
        "I keep meaning to reply to people and then a week has gone by.",
        {"is_disclosure": True, "disclosure_type": "symptom",
         "directness": "implicit", "confidence": 0.72},
    ),
    (
        "My flatmate has been really struggling since her diagnosis and I want "
        "to support her properly but I don't know how.",
        {"is_disclosure": False, "disclosure_type": "none",
         "directness": "not_applicable", "confidence": 0.9},
    ),
    (
        "Interesting piece in the Guardian today about waiting times for adult "
        "ADHD assessments across different trusts.",
        {"is_disclosure": False, "disclosure_type": "none",
         "directness": "not_applicable", "confidence": 0.96},
    ),
    (
        "Two years since my last episode. Still take the meds, still see my "
        "psychiatrist twice a year, but I feel like myself again.",
        {"is_disclosure": True, "disclosure_type": "treatment",
         "directness": "explicit", "confidence": 0.94},
    ),
    (
        "Has anyone here tried EMDR? My GP mentioned it and I am torn about "
        "whether to go ahead.",
        {"is_disclosure": True, "disclosure_type": "help_seeking",
         "directness": "explicit", "confidence": 0.8},
    ),
]


# --------------------------------------------------------------------------
# Prompt builders
#
# Four strategies. Comparing which produces the best DOWNSTREAM classifier is
# one of the experiments, so they share a codebook and differ only in framing.
# --------------------------------------------------------------------------

import json


def _format_examples() -> str:
    blocks = []
    for text, label in FEW_SHOT_EXAMPLES:
        blocks.append(f"POST:\n{text}\n\nLABEL:\n{json.dumps(label)}")
    return "\n\n---\n\n".join(blocks)


def build_zero_shot(post: str) -> str:
    """Codebook only. The cheapest strategy and the baseline for comparison."""
    return f"{CODEBOOK}\n\nPOST:\n{post}\n\nLABEL:"


def build_few_shot(post: str) -> str:
    """Codebook plus worked examples spanning the label space."""
    return (
        f"{CODEBOOK}\n\n"
        f"WORKED EXAMPLES\n\n{_format_examples()}\n\n---\n\n"
        f"POST:\n{post}\n\nLABEL:"
    )


def build_chain_of_thought(post: str) -> str:
    """
    Asks for reasoning before the label. Returns two blocks; the parser takes
    the JSON only. Slower per post, and on smaller models the reasoning step
    can degrade accuracy rather than improve it.
    """
    return (
        f"{CODEBOOK}\n\n"
        f"POST:\n{post}\n\n"
        "First, in no more than three sentences, work through: whose mental "
        "health is being described, whether anything is revealed about the "
        "author themselves, and whether it is stated plainly or implied.\n\n"
        "Then output the JSON object on a new line, after the marker LABEL:"
    )


def build_few_shot_cot(post: str) -> str:
    """Both. The most expensive strategy per post."""
    return (
        f"{CODEBOOK}\n\n"
        f"WORKED EXAMPLES\n\n{_format_examples()}\n\n---\n\n"
        f"POST:\n{post}\n\n"
        "First, in no more than three sentences, work through: whose mental "
        "health is being described, whether anything is revealed about the "
        "author themselves, and whether it is stated plainly or implied.\n\n"
        "Then output the JSON object on a new line, after the marker LABEL:"
    )


STRATEGIES = {
    "zero_shot": build_zero_shot,
    "few_shot": build_few_shot,
    "cot": build_chain_of_thought,
    "few_shot_cot": build_few_shot_cot,
}


def build(strategy: str, post: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Options: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy](post)