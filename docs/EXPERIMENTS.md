# Experiment protocol

This document describes how to reproduce the paper tables for **EmoRecAgent** on Amazon Reviews 2023 / `Beauty_and_Personal_Care`.

## Dataset

- **Source**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) category `Beauty_and_Personal_Care` (reviews + meta JSONL).
- **Filtering**: iterative **5-core** (`data.k_core: 5`) — repeatedly drop users with fewer than 5 reviews and items with fewer than 5 reviews until stable. Every surviving user has reviewed ≥ 5 distinct products; every surviving product has ≥ 5 distinct reviewers. This guarantees enough interaction density for User Profiling and agent reasoning. After optional agentic sampling (`max_users` / `max_items`), 5-core is re-applied so the invariant still holds.
- **Dedup**: one row per `(user_id, parent_asin)`, keep earliest timestamp.
- **Split**: **chronological ratio split (never random)** — per user, interactions sorted by time:
  - **80%** earliest → **train** (profile learning / CF fit)
  - **10%** next → **validation** (hyperparameter tuning: α, λ)
  - **10%** latest → **test** (final evaluation)
  Users without enough history for all three partitions are train-only (`data.min_history`).
- **Temporal cutoff**: ABSA aggregation and CF training use **train only**; valid is for tuning; test is held out until final metrics. No future interaction may appear in train.

## Agentic subset

Users in the agentic evaluation must have enough history for temporal preference to matter:

- `data.min_history` prior reviews (default 5)
- `data.min_distinct_aspects` distinct aspects in ABSA signals (default 2)
- Optional caps: `data.max_users`, `data.max_items` for LLM-bound paths

All methods are compared on the **identical** test-user set.

## Scoring model

\[
S(u,i) = \alpha \cdot S_{\text{base}}(u,i) + (1-\alpha) \sum_a w_u(a,t) \cdot \hat{E}_i(a)
\]

- \(S_{\text{base}}\): truncated SVD or ItemKNN on train interactions, min-max normalized over candidates.
- \(w_u(a,t)\): time-decayed salience from past ABSA signals (λ = `scoring.lambda_decay` per day).
- \(\hat{E}_i(a)\): helpfulness-capped mean polarity rescaled to \([0,1]\).

Default hyperparameters: `alpha=0.5`, `lambda_decay=0.01` (tune on valid only).

## Ranking metrics

**All reported metrics use internationally standard definitions** (see evaluation-metrics plan R-M0).

Per test row (one held-out interaction), candidates = all items not in train history (held-out included). Relevant set = `{held_out_item}`.

| Metric | Standard formula |
|--------|------------------|
| **HR@K** | \(\mathbb{1}[\exists\, i \le K : \text{rank}_i \in \mathcal{R}]\) |
| **Recall@K** | \(\|\mathcal{R} \cap \text{top-}K\| / \|\mathcal{R}\|\) |
| **NDCG@K** | \(\text{DCG@K}/\text{IDCG@K}\), \(\text{DCG@K}=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}\) |
| **MRR@K** | \(1/\text{rank of first relevant in top-}K\) (else 0) |
| **AvgHR@1,3,5** | \(\frac{1}{3}(\text{HR@1}+\text{HR@3}+\text{HR@5})\) |

Default K ∈ {5, 10, 20} (`eval.k_values`). AvgHR uses `eval.hr_avg_k: [1, 3, 5]`.

**Aggregation**

- **Row-mean** (`means`): each test interaction weighted equally.
- **User-mean** (`means_per_user`): average per user first, then across users — **primary for paper** when `n_test_rows >> n_test_users`.

**Protocols**

- `full_catalog` (default): rank all unseen items.
- `sampled_negatives`: `--n-negatives N` — held-out + N shuffled negatives.
- `cumulative_history`: `--cumulative-history` — prior test items for the same user enter seen history (leakage-safe).

**Significance**: `scripts/compare_results.py --a results/svd.json --b results/emorec.json --metric ndcg@10`

## ABSA quality metrics

Gold file: `data/labeled/absa_gold.jsonl` (500–1000 reviews).

| Metric | Definition |
|--------|------------|
| Micro-F1 | Global TP/FP/FN over `(aspect, sentiment)` keys |
| Macro-F1 (review) | Unweighted mean of per-review F1 |
| Macro-F1 (aspect) | Mean per-aspect F1 (aspects with gold support ≥ 5) |
| Coverage | Fraction of gold reviews present in ABSA cache |

Run: `make absa-quality` → `results/absa_quality.json`

### Hybrid ABSA (efficiency variant)

Offline ABSA supports two backends (`absa.backend` in `configs/default.yaml`):

| Backend | Role | Typical LLM calls / review |
|--------|------|----------------------------|
| `llm_only` | PASTEL-style extract→judge (methodology baseline) | 2 |
| `hybrid` | PyABSA ATEPC tool + LLM validate/repair | 1–2 |

- **Ship default:** `llm_only` until the quality gate passes.
- **Install hybrid:** `pip install -e ".[absa-ml]"`; `make warmup-absa` prefetches the PyABSA checkpoint.
- **Cache invalidation:** `absa.pipeline_version` + sidecar `absa_cache.cache_manifest.json`; mismatch → `make clean-absa`.
- **Latency benchmark:** `make absa-benchmark` → `results/absa_latency.json` (p50/p95/mean, `repair_rate`, `speedup_ratio`). Uses isolated temp caches under `/tmp` — does not modify production `absa_cache.sqlite`.
- **Quality gate (R11):** `make absa-quality-compare` → `results/absa_quality_comparison.json`. Flip default to `hybrid` only if macro review F1 drop vs `llm_only` is ≤ `absa.quality_gate_max_f1_drop` (default 0.02).

Paper positioning: hybrid ABSA is the **offline tool-augmented extraction stage**; the four-agent LangGraph path still loads precomputed cache. Report quality row (`llm_only` vs `hybrid`) before latency claims.

## Methods

| Method | Description |
|--------|-------------|
| `popularity` | global item frequency |
| `itemknn` | cosine ItemKNN |
| `svd` / `base_cf` | truncated SVD matrix factorization |
| `sequential` | first-order Markov over item sequences |
| `aspect_aware` | static aspect weights × \(\hat{E}_i(a)\) (EFM-style) |
| `emorecagent` | dynamic weights + blended \(S(u,i)\); ablation toggles control reflection |

## Factorial ablations

Configs under `configs/ablations/`:

| Config | reflection | dynamic_weights | aspect_term |
|--------|------------|-----------------|-------------|
| `full.yaml` | ✓ | ✓ | ✓ |
| `no_reflection.yaml` | ✗ | ✓ | ✓ |
| `no_dynamic_weights.yaml` | ✓ | ✗ | ✓ |
| `base_cf.yaml` | ✓ | ✓ | ✗ (α→1) |

Run all:

```bash
make ablations
```

## Claim-specific evaluation

Implemented in `src/emorecagent/eval/shift_eval.py`:

1. **Shift subpopulation** — users with a new salient-aspect complaint; report dynamic vs static lift.
2. **Counterfactual probe** — inject synthetic complaint; verify ranking shifts toward matching aspects.

## Explanation faithfulness

`src/emorecagent/eval/faithfulness.py`:

- ERASER-style perturbation (zero cited aspect contribution → rank should drop)
- Unfaithful control (shuffled aspects) should score lower
- Evidence coverage / sentiment agreement as secondary descriptors

## Significance

Paired bootstrap over per-user metric vectors (`eval.n_bootstrap`, default 1000). Report mean delta + p-value for ablation vs full system.

## Reproducibility

- Seed: `experiment.seed` (default 42); `emorecagent.utils.seeding.set_global_seed`
- Resolved config + dataset manifest hash written beside results via `RunLogger`
- Pin dependencies: `requirements.txt` / `pyproject.toml`

## Example commands

```bash
# Baseline
PYTHONPATH=src python3 scripts/run_experiment.py \
  --config configs/default.yaml --method svd \
  --split data/processed/Beauty_and_Personal_Care \
  --out results/svd.json

# Full system
PYTHONPATH=src python3 scripts/run_experiment.py \
  --config configs/ablations/full.yaml --method emorecagent \
  --split data/processed/Beauty_and_Personal_Care \
  --out results/full.json
```
