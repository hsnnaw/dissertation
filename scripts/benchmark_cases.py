"""
Synthetic evaluation benchmark for annotation prompt strategies.

Fifty cases with reference labels, spanning the label space and weighted toward
the boundaries where disagreement actually lives. Every case is written for this
benchmark. None are real posts, so this file is safe to commit.

IMPORTANT METHODOLOGICAL NOTE
-----------------------------
The reference labels here are the researcher's judgements, not ground truth. The
codebook was revised once already after initial testing, which means there is a
risk of tuning the codebook until the model agrees with it rather than until it
is correct. Two mitigations apply:

  1. The reference labels should be reviewed and, where the researcher
     disagrees, changed BEFORE running any strategy against them. A benchmark
     whose labels are adjusted after seeing model output measures nothing.

  2. Roughly a fifth of the cases are marked contested=True. These are cases
     where a reasonable annotator could defend either label. They are scored
     separately, because a strategy that does well on clear cases and badly on
     contested ones is a different proposition from one that fails across the
     board.

Fields per case:
    text        the post
    disclosure  reference is_disclosure
    dtype       reference disclosure_type
    directness  reference directness
    slice       category, for per-slice reporting
    contested   whether a reasonable annotator could disagree
    note        why the case is here
"""

from __future__ import annotations

CASES = [
    # ---------------------------------------------------------------- explicit
    # Named condition, medication, or therapy. A lexical baseline should get
    # these, so they establish the floor rather than discriminating.
    dict(text="Got the ADHD diagnosis confirmed on Tuesday after two years on the waiting list. Relieved and furious in roughly equal measure.",
         disclosure=True, dtype="diagnosis", directness="explicit",
         slice="explicit_diagnosis", contested=False,
         note="plain diagnosis"),
    dict(text="I was told it was probably bipolar II rather than recurrent depression. Not sure how I feel about that yet.",
         disclosure=True, dtype="diagnosis", directness="explicit",
         slice="explicit_diagnosis", contested=False,
         note="diagnosis with hedging language"),
    dict(text="Twenty six and just found out I have generalised anxiety disorder. Apparently this is not how everyone experiences a Tuesday.",
         disclosure=True, dtype="diagnosis", directness="explicit",
         slice="explicit_diagnosis", contested=False,
         note="diagnosis with humour, tests tone robustness"),
    dict(text="They have changed my diagnosis three times in four years. Currently it says EUPD on the letter which I do not accept.",
         disclosure=True, dtype="diagnosis", directness="explicit",
         slice="explicit_diagnosis", contested=False,
         note="diagnosis disputed by author, still a disclosure"),

    dict(text="Started on 50mg sertraline a fortnight ago. The nausea is grim but I am told it settles.",
         disclosure=True, dtype="treatment", directness="explicit",
         slice="explicit_treatment", contested=False,
         note="named medication"),
    dict(text="Third session of CBT tomorrow and I still have not done the thought record homework.",
         disclosure=True, dtype="treatment", directness="explicit",
         slice="explicit_treatment", contested=False,
         note="named therapy"),
    dict(text="Came off the mirtazapine in March against advice and I am now fairly sure that was a mistake.",
         disclosure=True, dtype="treatment", directness="explicit",
         slice="explicit_treatment", contested=False,
         note="medication cessation"),
    dict(text="Two weeks as a voluntary inpatient. Would not describe it as restful but I am glad I went.",
         disclosure=True, dtype="treatment", directness="explicit",
         slice="explicit_treatment", contested=False,
         note="hospitalisation, no condition named"),

    dict(text="Has anyone tried EMDR? My GP raised it and I cannot decide whether it sounds credible or not.",
         disclosure=True, dtype="help_seeking", directness="explicit",
         slice="explicit_help_seeking", contested=False,
         note="asking about treatment personally offered"),
    dict(text="Looking for a therapist in the Guildford area who does evening appointments. Any recommendations appreciated.",
         disclosure=True, dtype="help_seeking", directness="explicit",
         slice="explicit_help_seeking", contested=False,
         note="seeking treatment, implies own need"),
    dict(text="Is it normal for propranolol to make you feel this flat? Second week on it.",
         disclosure=True, dtype="treatment", directness="explicit",
         slice="explicit_help_seeking", contested=True,
         note="question form but discloses medication; type could be help_seeking"),

    # ---------------------------------------------------------------- implicit
    # The cases that matter. Disclosure is present but nothing is named.
    dict(text="I keep cancelling on people. Third time this month. They have stopped asking now, which is probably fair enough.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_symptom", contested=False,
         note="withdrawal, nothing named"),
    dict(text="Everyone says it gets easier and I keep waiting for the bit where it does.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_symptom", contested=False,
         note="very indirect, implied duration"),
    dict(text="Woke at four again. That is the ninth night running. I have given up pretending I will get back to sleep.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_symptom", contested=False,
         note="sleep disturbance, sustained"),
    dict(text="The washing up has been there since Sunday. It is not that I do not want to. I just cannot make myself start.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_symptom", contested=False,
         note="avolition described behaviourally"),
    dict(text="I have started taking the long way round so I do not walk past the office. It adds twenty minutes.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_symptom", contested=True,
         note="avoidance behaviour, could be non-clinical"),
    dict(text="Everything is fine. Genuinely. I do not know why I am typing this at 3am.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_symptom", contested=True,
         note="denial with contradicting context"),

    dict(text="Been off work six weeks. The GP signed me off again yesterday for another month.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_consequence", contested=True,
         note="absence with no named cause, could be physical"),
    dict(text="My sister has taken the kids for the weekend so I can, in her words, sort myself out.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_consequence", contested=True,
         note="disclosure via someone else's action"),
    dict(text="Failed the module resit. Third time. I did not even open the paper, I just sat there.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_consequence", contested=False,
         note="functional impairment through consequence"),
    dict(text="Handed my notice in without another job lined up. Everyone thinks I have lost it. Maybe.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_consequence", contested=True,
         note="impulsive action, disclosure arguable"),
    dict(text="I have eaten the same thing for dinner every night for two weeks because deciding is beyond me right now.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="implicit_consequence", contested=False,
         note="decision fatigue described"),

    # ---------------------------------------------------------------- recovery
    dict(text="Two years since the last episode. Still on the meds, still see someone twice a year, but I feel like myself.",
         disclosure=True, dtype="treatment", directness="explicit",
         slice="recovery", contested=True,
         note="recovery narrative; precedence rule puts treatment above recovery"),
    dict(text="Managed the supermarket on my own today. Small thing. Would have been unthinkable in January.",
         disclosure=True, dtype="recovery", directness="implicit",
         slice="recovery", contested=False,
         note="progress without naming condition"),
    dict(text="Relapsed last week after eight months. Trying not to treat it as the whole thing collapsing.",
         disclosure=True, dtype="recovery", directness="implicit",
         slice="recovery", contested=False,
         note="setback, nothing named"),
    dict(text="One year today. Not going to make a speech about it but it felt worth marking somewhere.",
         disclosure=True, dtype="recovery", directness="implicit",
         slice="recovery", contested=True,
         note="milestone, subject entirely unstated"),

    # ------------------------------------------------------------ third party
    # The most common false positive source. Mental health content is present
    # but it is not the author's.
    dict(text="My dad was sectioned in the nineties and we still do not really talk about it.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="third_party", contested=False,
         note="family member, historical"),
    dict(text="My flatmate has been struggling since her diagnosis and I want to support her properly but I do not know how.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="third_party", contested=False,
         note="third party, author seeking advice for them"),
    dict(text="My partner takes lithium and the blood tests are relentless. Does anyone else find the monitoring exhausting?",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="third_party", contested=True,
         note="third party medication, author reports own exhaustion"),
    dict(text="Watching my brother go through this has been the hardest year of my life and I am not coping with it well.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="third_party", contested=True,
         note="starts third party, pivots to own state"),
    dict(text="A friend asked me to post this for her because she does not want to make an account.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="third_party", contested=False,
         note="proxy posting, disclosure belongs to another"),
    dict(text="Everyone in my family has anxiety. It is basically the house style at this point.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="third_party", contested=True,
         note="family framing that includes the author"),

    # ------------------------------------------------------- news / advocacy
    dict(text="New figures out today showing adult assessment waiting lists have doubled since 2019.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="news", contested=False,
         note="statistics, no personal content"),
    dict(text="Good long read in the Guardian on how the referral system actually works. Worth twenty minutes.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="news", contested=False,
         note="article share"),
    dict(text="Petition to fund more community mental health teams has hit 100k signatures. Link in comments.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="news", contested=False,
         note="advocacy"),
    dict(text="Interesting study suggesting exercise matches medication for mild depression. Methodology looks shaky to me though.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="news", contested=False,
         note="research discussion with opinion, still not personal"),

    # ---------------------------------------------------------- supportive
    dict(text="Reminder that the crisis line is open 24/7 if anyone needs it tonight.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="supportive", contested=False,
         note="community support post"),
    dict(text="Whoever posted earlier about the waiting list, I hope you are alright. This sub is here.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="supportive", contested=False,
         note="responding to another's disclosure"),
    dict(text="Just want to say I read every post here even when I do not comment. You are all doing better than you think.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="supportive", contested=True,
         note="lurker admission, arguably implies membership"),

    # ------------------------------------------------------------- quoting
    dict(text="Someone said to me last week that I seemed lighter. I have been turning that over ever since.",
         disclosure=True, dtype="recovery", directness="implicit",
         slice="quoting", contested=True,
         note="quoted observation about the author"),
    dict(text="My therapist says I catastrophise. My response was to spend the evening imagining all the ways that could go wrong.",
         disclosure=True, dtype="treatment", directness="explicit",
         slice="quoting", contested=False,
         note="quotes clinician, discloses own treatment"),
    dict(text="Quoting from the pinned post: recovery is not linear. Reposting because someone needs to read it today.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="quoting", contested=False,
         note="quoted material, no personal content"),

    # ------------------------------------------------- physical ambiguity
    # Codebook says confidence below 0.6 when mental vs physical is unclear.
    dict(text="Exhausted constantly. Sleeping ten hours and waking up as tired as when I went to bed.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="physical_ambiguity", contested=True,
         note="fatigue, could be thyroid, anaemia, anything"),
    dict(text="Lost about a stone without trying. Should probably get that looked at.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="physical_ambiguity", contested=True,
         note="weight change with no mental health framing"),
    dict(text="The headaches are back and the GP wants more tests before referring me anywhere.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="physical_ambiguity", contested=True,
         note="physical symptom, medical but not mental health"),
    dict(text="Chest tightness most mornings. Cardiology cleared me so now they are saying it might be stress.",
         disclosure=True, dtype="symptom", directness="implicit",
         slice="physical_ambiguity", contested=True,
         note="physical presentation reattributed to mental health"),

    # ------------------------------------------------------------- control
    dict(text="Anyone got recommendations for a decent noise cancelling headset for an open plan office?",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="control", contested=False,
         note="unrelated"),
    dict(text="Third time this month the 07:42 has been cancelled. Considering taking up cycling out of pure spite.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="control", contested=False,
         note="frustration, non-clinical, tests over-triggering"),
    dict(text="Made sourdough for the first time and it came out like a frisbee. Where did I go wrong.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="control", contested=False,
         note="unrelated with mild self-criticism"),
    dict(text="I am so depressed that Arsenal dropped points again. Genuinely ruined my weekend.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="control", contested=True,
         note="colloquial 'depressed', key lexical false positive"),
    dict(text="This weather is doing my head in. Fourth grey week in a row.",
         disclosure=False, dtype="none", directness="not_applicable",
         slice="control", contested=True,
         note="idiomatic distress language, non-clinical"),
]


def summary() -> dict:
    """Composition of the benchmark, for reporting in the dissertation."""
    from collections import Counter
    return {
        "total": len(CASES),
        "positive": sum(c["disclosure"] for c in CASES),
        "contested": sum(c["contested"] for c in CASES),
        "by_slice": dict(Counter(c["slice"] for c in CASES)),
        "by_directness": dict(Counter(c["directness"] for c in CASES)),
        "by_type": dict(Counter(c["dtype"] for c in CASES)),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
