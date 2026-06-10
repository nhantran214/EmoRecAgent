"""Versioned prompt templates for all LLM stages (U3).

Templates are plain strings with `{placeholders}` so they stay testable without
importing LangChain prompt objects. Bump the version comment when wording changes
so paper runs can cite the prompt revision.
"""

from __future__ import annotations

# v1 — PASTEL-style extract-then-judge decomposition

ABSA_EXTRACT_V1 = """\
You are an aspect-based sentiment analysis (ABSA) extractor for product reviews.

Review text:
\"\"\"{review_text}\"\"\"

Extract every (aspect, opinion, sentiment) triple where:
- aspect: a concrete product attribute (e.g. scent, texture, packaging)
- opinion: the exact opinion phrase from the text
- sentiment: positive | negative | neutral

Think step by step, then return ONLY the structured triple list.
"""

ABSA_JUDGE_V1 = """\
You are an ABSA validator. Given a review and candidate triples, keep only triples
that are explicitly supported by the review text. Drop hallucinated aspects or
wrong polarities. Assign confidence in [0, 1] reflecting how clearly the text supports each triple.

Review text:
\"\"\"{review_text}\"\"\"

Candidate triples (JSON):
{candidates_json}

Return the validated triple list.
"""

REASONING_COT_V1 = """\
You are a recommendation reasoning agent. Explain why the top candidate items match
the user's aspect preferences. Ground every claim in the provided preference weights
and item aspect scores.

User top aspects (weight): {top_aspects}
Candidate breakdowns: {candidate_summary}

Provide a concise chain-of-thought, then a ranked justification.
"""

REFLECTION_JUDGE_V1 = """\
You are a reflection agent auditing a recommendation list.

User constraints:
{constraints}

Recommended items with scores:
{recommendations}

Check: budget fit (if price known), alignment with recent complaints, and whether
top aspects are covered. Return approved=true only if all hard constraints pass.
"""

EXPLANATION_V1 = """\
Write a short, evidence-grounded explanation for why this item was recommended.

User cared about: {user_aspects}
Item aspect scores (0-1): {item_aspects}
Score contributions: {contributions}

Cite only aspects present in the data. Do not invent numeric product specs.
"""


def format_prompt(template: str, **kwargs: str) -> str:
    return template.format(**kwargs)
