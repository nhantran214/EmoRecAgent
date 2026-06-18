# ABSA gold labeling

## Workflow

1. Sample candidates: `make sample-gold` → `absa_gold_candidates.jsonl`
2. Label `(aspect, sentiment)` triples per review (use canonical aspects from `src/emorecagent/absa/normalize.py` `SYNONYM_MAP`)
3. Save finalized labels to `absa_gold.jsonl`
4. Run ABSA on gold review ids: `make absa`
5. Evaluate: `make absa-quality`

## JSONL schema

```json
{
  "review_id": "...",
  "triples": [
    {"aspect": "scent", "opinion": "...", "sentiment": "positive", "confidence": 1.0}
  ]
}
```

## QC

- Pilot: 50 reviews (`--pilot 50`)
- Dual annotation: 100 reviews, report Jaccard via `eval_absa_quality.py --dual-annotation v1.jsonl v2.jsonl`
