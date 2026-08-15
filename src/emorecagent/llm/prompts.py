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

PREFERENCE_MANIFESTO_V1 = """\
You distill a user's real-time shopping preference from their review history and
aspect-weight trajectory. Write a short preference statement (2-4 sentences) capturing:
- salient product aspects they care about now
- how tastes shifted over time (if signals disagree)
- what they are likely seeking next

Ground every claim in the provided ABSA triples and profiling weights. Do not invent
products or prices.

ABSA triples (aspect, sentiment, item):
{absa_summary}

Profiling weights at query time (aspect: weight):
{weights_summary}

Recent review snippets:
{review_snippets}

Return JSON: {{"preference_statement": "<T_u text>"}}
One compact JSON line only.
"""

STAGE2_RERANK_V1 = """\
You are a Stage 2 recommendation reranker. Reorder ONLY the candidate items
using the user's preference summary, their reviewed items, cross-user purchase
patterns, and Stage 1 scores. Do not invent product facts.

User preference summary (T_u):
{T_u}

Prefix-reviewed items: {reviewed_items}

Cross-user hints (same-category co-purchases after reviewing an anchor):
{lookup_hints}

Candidate cards (item_id | Stage 1 score):
{candidate_cards}

Return JSON matching ReasoningRankingVerdict with ranked_item_ids: a list of item_id
strings containing every candidate exactly once, best-first.

Output format: one compact JSON line only, e.g.
{{"ranked_item_ids":["id1","id2","id3"]}}
No pretty-printing, no newlines inside the JSON, no extra keys.
"""

# Paper §III.F evidence pack (Fig. 4): includes alignment confidence c_u.
STAGE2_RERANK_PAPER_V1 = """\
You are the Guarded Reranking Agent. Reorder ONLY the candidate shortlist using
the evidence pack below. Do not invent product facts. The sequential ranking
agent already scored the full catalog; you may only permute this bounded pool.

Head-preservation rules (critical):
- Treat items with small π¹_rank (especially 1–10) as a strong prior. Keep them
  near the top of your list unless T_u clearly contradicts their title/cats.
- Do NOT bury π¹_rank 1–5 below position ~15 without strong title/T_u evidence.
- Prefer mild reordering inside the Stage-1 head over aggressive reshuffles.
- Promote a mid/tail item (worse π¹_rank) into the top only when its title/cats
  clearly match T_u better than the current head items.
- If unsure, stay close to the Stage-1 order (π¹_rank ascending).

User preference manifesto (T_u):
{T_u}

Alignment confidence c_u (how strongly T_u agrees with the sequential state s_u;
higher → more trust in language preferences): {c_u}

Prefix-reviewed / anchor items (with titles when available):
{reviewed_items}

Cross-user co-occurrence hints:
{lookup_hints}

Candidate cards (item_id | score | name | cats | π¹_rank):
{candidate_cards}

Return JSON matching ReasoningRankingVerdict with ranked_item_ids: a list of item_id
strings containing every candidate exactly once, best-first.

Output format: one compact JSON line only, e.g.
{{"ranked_item_ids":["id1","id2","id3"]}}
No pretty-printing, no newlines inside the JSON, no extra keys.
"""

# Listwise ranking of a φ-window (ltr_llm). Full permutation — not swap/abstain.
STAGE2_RERANK_V6_PHI = """\
You are the Stage-2 ranking agent. Reorder the candidate list for the user's
next purchase. This is a full ranking job: do NOT abstain, do NOT return [].

Signals on each card:
- φ= complementary potential (higher = more likely the held-out next item than
  Stage-1 rank suggests)
- π¹_rank= Stage-1 catalog rank
- co= cross-user co-purchase with the user's history (0-1)
- name / cats / rev= product text (titles, categories, review snippets)

Ranking rules:
1. Primary: match T_u (product type, ingredients, use-case, brand, avoid).
2. Secondary: high φ is a useful prior — keep strong-φ items near the top unless
   they contradict T_u or appear in avoid=.
3. You MAY move a mid/tail candidate into the top-10 when title/cats/rev match
   T_u clearly better than current head items.
4. You MAY demote a high-φ head item that mismatches T_u (wrong type, in avoid=).
5. Do not invent product facts. Use only the cards.
6. Return EVERY candidate id exactly once, best-first.

User preference manifesto (T_u):
{T_u}

Alignment confidence c_u: {c_u}

Prefix-reviewed / anchor items:
{reviewed_items}

Cross-user co-occurrence hints:
{lookup_hints}

Candidate cards (item_id | S= | φ= | co= | name | cats | π¹_rank | rev=):
{candidate_cards}

Return JSON matching ReasoningRankingVerdict with ranked_item_ids: a list of item_id
strings containing every candidate exactly once, best-first.

Output format: one compact JSON line only, e.g.
{{"ranked_item_ids":["id1","id2","id3"]}}
No pretty-printing, no newlines inside the JSON, no extra keys.
"""

# Listwise ranking of a φ-window (ltr_llm). φ is primary; T_u / co are secondary.
STAGE2_RERANK_V7_PHI = """\
You are the Stage-2 ranking agent. Reorder the candidate list for the user's
next purchase. This is a full ranking job: do NOT abstain, do NOT return [].

Signal weights (higher = more important; your permutation is a secondary nudge):
- φ PRIMARY (weight {w_phi}): complementary potential. High φ means the item is
  more likely the held-out next purchase than Stage-1 rank suggests. Keep the
  φ order unless a secondary signal is clearly stronger.
- T_u / ABSA SECONDARY (weight {w_tu}): title/cats/rev match to the manifesto.
- co-purchase SECONDARY (weight {w_co}): cross-user co-occurrence with history.
- Your listwise order is mixed back with weight {w_llm}; it cannot fully override φ.

Card fields: φ=, π¹_rank=, co=, name / cats / rev=.

Ranking rules:
1. Primary: preserve high-φ items near the top.
2. Secondary: use T_u (product type, ingredients, use-case, avoid=) and co= to
   make SMALL adjustments, especially outside the top-20.
3. Top-20 (hr@20) — reason carefully, one swap at a time:
   - Do NOT demote a high-φ item out of the top-20 unless T_u clearly contradicts
     (wrong product type or listed in avoid=).
   - Promote a mid/tail item into the top-20 ONLY when title/cats/rev match T_u
     substantially better than the displaced head item AND φ is not much weaker.
4. Outside top-20 you may reorder more freely to surface T_u / co matches.
5. Do not invent product facts. Use only the cards.
6. Return EVERY candidate id exactly once, best-first.

User preference manifesto (T_u):
{T_u}

Alignment confidence c_u: {c_u}

Prefix-reviewed / anchor items:
{reviewed_items}

Cross-user co-occurrence hints:
{lookup_hints}

Candidate cards (item_id | S= | φ= | co= | name | cats | π¹_rank | rev=):
{candidate_cards}

Return JSON matching ReasoningRankingVerdict with ranked_item_ids: a list of item_id
strings containing every candidate exactly once, best-first.

Output format: one compact JSON line only, e.g.
{{"ranked_item_ids":["id1","id2","id3"]}}
No pretty-printing, no newlines inside the JSON, no extra keys.
"""

# Top-K promote: easier for 7B than full listwise permute of 40 IDs.
# Used by ``top_k_promote`` (may demote Stage-1 head) and ``promote_preserve``
# (A1: structural head freeze — see STAGE2_RERANK_PROMOTE_PRESERVE_V1).
STAGE2_RERANK_PROMOTE_V1 = """\
You are a preference-aware product reranker. From the candidate shortlist, pick
the {promote_k} items that BEST match the user's preference manifesto T_u based
on product titles, categories, and review snippets when present. Do not invent
product facts.

Rules:
- Prefer title/category/snippet match to T_u over Stage-1 score or π¹_rank.
- If T_u mentions brands, ingredients, product types, or use-cases, promote
  candidates whose titles or review snippets reflect those.
- Return ONLY item_id strings from the candidate list (no titles in the JSON).
- Return at most {promote_k} ids, best-first. You may return fewer if unsure.

User preference manifesto (T_u):
{T_u}

Alignment confidence c_u: {c_u}

Recently reviewed / anchor products:
{reviewed_items}

Cross-user co-occurrence hints:
{lookup_hints}

Candidate cards (item_id | score | name | cats | π¹_rank | rev=…):
{candidate_cards}

Return JSON: {{"ranked_item_ids":["id1","id2",...]}} with up to {promote_k}
best item_ids, best-first. One compact JSON line only.
"""

# A1 insert-after-head (can shift ranks protect_n+1.. out of top-10).
STAGE2_RERANK_PROMOTE_PRESERVE_V1 = """\
You are the Guarded Reranking Agent (promote-only). Pick up to {promote_k}
candidates to INSERT just below the frozen Stage-1 head (π¹ ranks 1–{protect_n}
stay fixed in code). Do not invent product facts.

Rules:
- Prefer items with π¹_rank > {protect_n} whose title/cats/review snippet clearly
  match T_u better than unprotected head items.
- Do NOT nominate items already in π¹ ranks 1–{protect_n} (they cannot move).
- If unsure, return fewer ids (or none).
- Return ONLY item_id strings from the candidate list.

User preference manifesto (T_u):
{T_u}

Alignment confidence c_u: {c_u}

Recently reviewed / anchor products:
{reviewed_items}

Cross-user co-occurrence hints:
{lookup_hints}

Candidate cards (item_id | score | name | cats | π¹_rank | rev=…):
{candidate_cards}

Return JSON: {{"ranked_item_ids":["id1","id2",...]}} with up to {promote_k}
best item_ids, best-first. One compact JSON line only.
"""

# Promote-swap: replace only the last k slots of top-(protect_n+k); no shift.
STAGE2_RERANK_PROMOTE_SWAP_V1 = """\
You are the Guarded Reranking Agent (promote-swap). Pick up to {promote_k}
candidates to SWAP into the tail of the Stage-1 top-{head_n} (π¹ ranks
1–{protect_n} stay frozen; only ranks {protect_n_plus_1}–{head_n} may be
replaced). Do not invent product facts.

Rules:
- Only nominate items with π¹_rank > {head_n} (outside the current top-{head_n}).
- Promote only when title/cats/review snippet clearly match T_u better than the
  weak tail of the head.
- If unsure, return fewer ids (or none) — skipping is better than a bad swap.
- Return ONLY item_id strings from the candidate list.

User preference manifesto (T_u):
{T_u}

Alignment confidence c_u: {c_u}

Recently reviewed / anchor products:
{reviewed_items}

Cross-user co-occurrence hints:
{lookup_hints}

Candidate cards (item_id | score | name | cats | π¹_rank | rev=…):
{candidate_cards}

Return JSON: {{"ranked_item_ids":["id1","id2",...]}} with up to {promote_k}
best item_ids, best-first. One compact JSON line only.
"""

# Call 1 of reason-then-pick: extract structured prefs from T_u.
STAGE2_EXTRACT_PREFS_V1 = """\
Extract shopping preference facts from the user manifesto T_u. Do not invent
brands or ingredients that are not clearly implied by the text. Empty lists
are fine when T_u is vague.

Rules for each string value:
- At most 4 items per list, at most 4 words per item.
- ASCII letters/digits/hyphen/space only — no quotation marks, no commas inside
  an item, no apostrophes (write LOreal not L'Oreal; write doesnt not doesn't).

User preference manifesto (T_u):
{T_u}

Return ONE JSON object with keys brands, product_types, ingredients, use_cases,
avoid, keywords (each a list of strings).
"""

# Call 1 (deep): richer prefs + explicit decision rule for swap.
STAGE2_EXTRACT_PREFS_V2 = """\
You are a careful preference analyst. Extract shopping preference facts from T_u.
Do not invent brands or ingredients that are not clearly implied. Empty lists are
fine when T_u is vague.

Think about what would justify replacing a mediocre Stage-1 head item with a
better match from the shortlist.

Rules for each string list value:
- At most 4 items per list, at most 4 words per item.
- ASCII letters/digits/hyphen/space only — no quotation marks, no commas inside
  an item, no apostrophes (write LOreal not L'Oreal; write doesnt not doesnt).

User preference manifesto (T_u):
{T_u}

Return ONE JSON object with keys:
- must_have (list): hard requirements implied by T_u
- nice_to_have (list): soft preferences
- avoid (list): dealbreakers / dislikes
- brands, product_types, ingredients, keywords (lists)
- decision_rule (string): one short ASCII sentence stating when a candidate
  should beat a displacee (e.g. matches must_have product type and ingredient)
"""

# Call 2 of reason-then-pick: contrastive swap against the displacee slot(s).
STAGE2_REASON_PICK_V1 = """\
You are the Guarded Reranking Agent (reason-then-pick). Decide whether to SWAP
any eligible candidate into the Stage-1 head tail. Do not invent product facts.

Preference facts extracted from T_u:
{preference_facts}

Original T_u:
{T_u}

Alignment confidence c_u: {c_u}

Displacee card(s) currently occupying the replaceable head slot(s) — a swap
removes one of these from the top:
{displacee_cards}

Eligible candidate cards (outside the frozen head; only these may be swapped in):
{candidate_cards}

Rules:
- Swap ONLY if a candidate clearly matches preference facts / T_u better than
  the displacee (title, categories, or review snippet evidence).
- Prefer brands / product_types / ingredients / use_cases matches; respect avoid.
- If unsure or evidence is weak, return an empty ranked_item_ids list.
- Return at most {promote_k} item_id strings from the eligible candidates.

Return JSON: {{"ranked_item_ids":["id1",...],"rationale":"one short sentence"}}
"""

# Call 2 (deep): mandatory contrastive scorecard before pick.
STAGE2_REASON_PICK_V2 = """\
You are the Guarded Reranking Agent (deep scorecard). Decide whether to SWAP any
eligible candidate into the Stage-1 replaceable head slot. Do not invent facts.
Never copy or repeat the candidate card lines in your answer — JSON only.

Preference facts / decision rule from T_u:
{preference_facts}

Original T_u:
{T_u}

Alignment confidence c_u: {c_u}

Displacee card(s) currently in the replaceable slot(s) — a swap removes one:
{displacee_cards}

Eligible candidate cards (only these ids may be swapped in):
{candidate_cards}

Eligible ids: {eligible_ids}

Reasoning procedure (do this carefully before choosing):
1. Score the displacee fit to preference facts on a 0-5 scale; cite evidence
   from its title, categories, or rev= snippet only.
2. Score ONLY candidates that could plausibly beat the displacee (typically a
   handful). Omit clear non-matches — do NOT emit a row for every eligible id.
3. Set beats_displacee true only if clearly better on must_have / decision_rule.
4. A swap is allowed ONLY when fit(candidate) >= fit(displacee) + 2 AND
   beats_displacee is true AND evidence is concrete (not generic praise).
5. If nobody meets the threshold, return ranked_item_ids as [].
6. Return at most {promote_k} swap ids, best-first.

Evidence: at most 6 ASCII words; no quotes, commas, or apostrophes.

Return ONE SINGLE-LINE JSON object (no markdown, no pretty-print, no newlines):
{{"displacee":{{"id":"...","fit":0,"evidence":"..."}},"candidates":[{{"id":"...","fit":0,"beats_displacee":false,"evidence":"..."}}],"ranked_item_ids":[],"rationale":"one short ASCII sentence"}}
"""

# Call 2 (deep + overlap focus): scorecard forced to use ov= / must_have evidence.
STAGE2_REASON_PICK_V3 = """\
You are the Guarded Reranking Agent (overlap-grounded scorecard). Decide whether
to SWAP eligible candidates into the Stage-1 replaceable head slots. Do not invent
facts. Never copy candidate card lines — JSON only.

DEFAULT ACTION: abstain (ranked_item_ids=[]). Swap only when evidence is strong.

Preference facts / decision rule from T_u:
{preference_facts}

Original T_u:
{T_u}

Alignment confidence c_u: {c_u}

Displacee card(s) in replaceable slot(s) — ov=cand_hits/disp_hits on each card:
{displacee_cards}

Eligible candidate cards (already lexical-filtered; only these ids may swap in).
Each card ends with ov=cand_hits/disp_hits — higher ov is stronger lexical fit:
{candidate_cards}

Eligible ids: {eligible_ids}

Reasoning procedure (follow in order):
1. Read must_have / product_types / ingredients / decision_rule from preference facts.
2. For the displacee: note its ov left number and whether title/rev matches must_have.
3. Consider at most the 3 eligible cands with the highest ov left numbers.
4. A candidate may enter ranked_item_ids ONLY if ALL hold:
   (a) ov left number is STRICTLY greater than the displacee ov left number,
   (b) fit(candidate) >= fit(displacee) + 2,
   (c) beats_displacee is true,
   (d) evidence cites a concrete must_have / product_type / ingredient token that
       appears in that candidate title, cats, or rev= (not generic praise).
5. If none qualify, ranked_item_ids MUST be [].
6. Return at most {promote_k} swap ids, best-first (prefer higher ov, then fit).

Evidence: at most 6 ASCII words; no quotes, commas, or apostrophes.

Return ONE SINGLE-LINE JSON object (no markdown, no pretty-print, no newlines):
{{"displacee":{{"id":"...","fit":0,"evidence":"..."}},"candidates":[{{"id":"...","fit":0,"beats_displacee":false,"evidence":"..."}}],"ranked_item_ids":[],"rationale":"one short ASCII sentence"}}
"""

# Call 2 (override): quality-first — ov=/T_u lexical is advisory; LLM may contradict it.
STAGE2_REASON_PICK_V4_OVERRIDE = """\
You are the Guarded Reranking Agent (quality-first override). Your goal is to
maximize hit@10/hit@20 and ranking quality for the held-out next item. Do not invent
facts. Never copy candidate card lines — JSON only.

Lexical ov=cand_hits/disp_hits on cards is ADVISORY only. If title/cats/rev evidence
clearly matches must_have / decision_rule, you MAY promote a candidate even when ov
is low or does not beat the displacee. Prefer improving top-20 quality over obeying
overlap heuristics.

Preference facts / decision rule from T_u:
{preference_facts}

Original T_u:
{T_u}

Alignment confidence c_u: {c_u}

Displacee card(s) in replaceable slot(s):
{displacee_cards}

Eligible candidate cards (full Stage-1 pool outside the replaceable head — scan
ALL of them; π¹_rank may range into the deep tail, e.g. 11–300):
{candidate_cards}

Eligible ids: {eligible_ids}

Reasoning procedure:
1. Extract must_have / product_types / ingredients / decision_rule.
2. Score displacee fit 0-5 from title/cats/rev only.
3. Scan the FULL eligible list for must_have / decision_rule matches in title/cats/rev.
   Do not stop at π¹ 11–20 — a strong match at rank 50–300 may still be the best swap.
4. Promote a cand when fit(cand) >= fit(displacee)+2 AND beats_displacee true AND
   evidence cites a concrete preference token present on that cand card.
5. If lexical ov conflicts with strong card evidence, TRUST the card evidence.
6. If nobody is clearly better, ranked_item_ids=[].
7. At most {promote_k} ids, best-first (prefer stronger evidence, then better π¹_rank).

Evidence: at most 6 ASCII words; no quotes, commas, or apostrophes.

Return ONE SINGLE-LINE JSON object (no markdown, no pretty-print, no newlines):
{{"displacee":{{"id":"...","fit":0,"evidence":"..."}},"candidates":[{{"id":"...","fit":0,"beats_displacee":false,"evidence":"..."}}],"ranked_item_ids":[],"rationale":"one short ASCII sentence"}}
"""

# Call 2 (φ-grounded): complementary potential + ABSA + cross-user co-purchase.
STAGE2_REASON_PICK_V5_PHI = """\
You are the Guarded Reranking Agent (phi-grounded scorecard). Swap items in the
replaceable head when a focus candidate has higher complementary potential and
preference evidence. Do not invent facts. Never copy candidate card lines — JSON only.

phi is a listwise potential score on the Stage-1 pool: higher phi means the item is
more likely the held-out next purchase than Stage-1 rank suggests. co= is cross-user
co-purchase with the user's history (0-1). T_u / preference facts come from ABSA
on the user's reviews.

DEFAULT ACTION: abstain (ranked_item_ids=[]). Keep the current head when evidence
is weak.

Preference facts / decision rule (ABSA-derived) from T_u:
{preference_facts}

Original T_u:
{T_u}

Alignment confidence c_u: {c_u}

Displacee card(s) in the replaceable head. Low phi here is a demotion candidate:
{displacee_cards}

Eligible focus cards (top-phi from the Stage-1 pool outside the head). High phi
here is a promotion candidate:
{candidate_cards}

Eligible ids: {eligible_ids}

Reasoning procedure:
1. Read must_have / product_types / ingredients / decision_rule from ABSA facts.
2. Score displacee fit 0-5 from title/cats/rev and whether phi is weak vs the focus set.
3. Scan focus cards. A good swap has:
   (a) phi(cand) > phi(displacee) — primary potential signal,
   (b) title/cats/rev or co= supports T_u / must_have at least as well as the displacee,
   (c) fit(cand) >= fit(displacee)+2 AND beats_displacee true,
   (d) evidence cites a concrete token from the cand card (name, cats, rev, or co).
4. Do not promote a high-phi item that contradicts avoid= or clearly mismatches T_u.
5. Do not demote a high-phi displacee that already matches T_u well.
6. If nobody is clearly better, ranked_item_ids=[].
7. At most {promote_k} ids, best-first (higher phi, then stronger ABSA/co evidence).

Evidence: at most 6 ASCII words; no quotes, commas, or apostrophes.

Return ONE SINGLE-LINE JSON object (no markdown, no pretty-print, no newlines):
{{"displacee":{{"id":"...","fit":0,"evidence":"..."}},"candidates":[{{"id":"...","fit":0,"beats_displacee":false,"evidence":"..."}}],"ranked_item_ids":[],"rationale":"one short ASCII sentence"}}
"""

RANKING_PROMOTE_JSON_SUFFIX = (
    '\n\nOutput ONE compact JSON line: {"ranked_item_ids":["id1","id2",...]} '
    "with up to the requested top-K candidate ids, best-first. "
    "No newlines inside JSON."
)

REASONING_RANK_V1 = """\
You are a recommendation ranking agent. Order the candidate items to maximize relevance
for the user's salient aspect preferences. Use ONLY the provided scores and aspect
drivers — do not invent product facts.

User top aspects (weight): {top_aspects}

Candidate cards (item_id, total score, top aspect drivers, base CF score):
{candidate_cards}

Return JSON matching ReasoningRankingVerdict with ranked_item_ids: a list of item_id
strings containing every candidate exactly once, best-first.

Output format: one compact JSON line only, e.g.
{{"ranked_item_ids":["id1","id2","id3"]}}
No pretty-printing, no newlines inside the JSON, no extra keys.
"""

RANKING_JSON_SUFFIX = (
    '\n\nOutput ONE compact JSON line: {"ranked_item_ids":["id1","id2",...]} '
    "with every candidate exactly once, best-first. No newlines inside JSON."
)

BATCH_RANKING_JSON_SUFFIX = (
    '\n\nOutput ONE compact JSON line: {"rows":[{"row_id":"...","ranked_item_ids":[...]},...]} '
    "No newlines inside JSON."
)

REASONING_RANK_BATCH_V1 = """\
You are a recommendation ranking agent. For each task below, order that task's
candidate items to maximize relevance for the user's salient aspect preferences.
Use ONLY the provided scores and aspect drivers — do not invent product facts.

Each task is independent. Return JSON matching BatchReasoningRankingVerdict:
rows: a list of {{row_id, ranked_item_ids}} where ranked_item_ids contains every
candidate for that task exactly once, best-first.

Output format: one compact JSON line only, e.g.
{{"rows":[{{"row_id":"r1","ranked_item_ids":["a","b"]}}]}}
No pretty-printing, no newlines inside the JSON.

{task_blocks}
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
