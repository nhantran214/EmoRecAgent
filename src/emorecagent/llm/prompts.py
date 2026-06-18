"""Versioned prompt templates for all LLM stages.

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

Extract every distinct (aspect, opinion, sentiment) triple where:
- aspect: a concrete product attribute (e.g. scent, texture, packaging)
- opinion: the exact opinion phrase from the text
- sentiment: positive | negative | neutral (required on every triple)

Rules:
- No duplicate triples; at most 15 triples per review
- Every triple MUST include aspect, opinion, and sentiment

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

ABSA_AGENT_VALIDATE_FAST_V1 = """\
You are an ABSA validator for high-confidence classical model candidates.

Candidate triples (JSON) were produced by a DeBERTa/PyABSA tool. Each triple must have:
- aspect: lowercase product attribute
- opinion: exact opinion phrase grounded in the review (you must infer from aspect context if empty)
- sentiment: positive | negative | neutral
- confidence: [0, 1]

Candidate triples:
{candidates_json}

Rules:
- Drop unsupported aspects; fix wrong polarities
- Fill missing opinion phrases when aspect is supported
- Set needs_repair=true if important aspects are likely missing (list hints in missing_aspect_hints)
- At most 15 triples

Return HybridAbsaVerdict (triples, needs_repair, missing_aspect_hints).
"""

ABSA_AGENT_VALIDATE_V1 = """\
You are an ABSA validator orchestrating classical model output.

Review text:
\"\"\"{review_text}\"\"\"

Candidate triples (JSON):
{candidates_json}

Validate candidates against the review. Fill grounded opinion phrases. Drop hallucinations.
Set needs_repair=true if hidden aspects remain (list hints). Max 15 triples.
Return HybridAbsaVerdict.
"""

ABSA_AGENT_REPAIR_V1 = """\
You are an ABSA repair agent. The classical tool missed or mislabeled some aspects.

Review text:
\"\"\"{review_text}\"\"\"

Tool candidates (JSON):
{candidates_json}

Missing aspect hints:
{missing_hints}

Return final validated TripleSet grounded in the review only. Max 15 triples.
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
