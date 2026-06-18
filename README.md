# EmoRecAgent

Multi-agent affective recommendation framework for research (LangGraph + local LLM + Neo4j KG).

**Core contributions**

1. **Dynamic Preference Shifting** — time-decayed aspect weights \(w_u(a,t)\) blended with collaborative filtering in \(S(u,i)\).
2. **Rationalized Explanations** — evidence-grounded justifications tied to score drivers, with faithfulness checks.

## Requirements

- Python 3.11+ (conda env **ERA** for full EmoRecAgent dev)
- Agent4Rec baseline: Python 3.9 (**A4R-baseline**) — see `baseline/Agent4Rec/py39/README.md`
- [Ollama](https://ollama.com/) with `qwen2.5:7b` (agents / experiment) and optionally `qwen2.5:3b` for ABSA (`LLM_MODEL` / `LLM_MODEL_SMALL` in `.env`)
- Docker (Neo4j via `docker compose`)
- Amazon Reviews 2023 — `Beauty_and_Personal_Care` (see Data)

## Quick start

```bash
conda activate ERA     # or your venv
cp .env.example .env   # fill NEO4J_URI, OLLAMA_HOST, etc.
make install-dev       # installs package + pytest (dev extras)
make test              # 130+ unit tests; most need no GPU/Neo4j
```

If `pytest` is missing: `pip install -e ".[dev]"` or `pip install pytest>=8`.

### Neo4j

```bash
make neo4j             # starts Neo4j on host ports 7475 (HTTP) / 7688 (Bolt)
make load-kg           # train interactions + ABSA cache → Neo4j (after make data + make absa)
```

`make load-kg -- --fresh` clears the graph before reload.

### Data pipeline

```bash
# Download raw JSONL (if not present)
python3 scripts/download_amazon_reviews.py --category Beauty_and_Personal_Care

# k-core filter (default 10: users & items each >= 10 distinct partners) + chronological 80/10/10 split
make data            # logs → logs/data_YYYYMMDD_HHMMSS.log
```

Full build logs each pipeline stage (config, counts, timing, output paths) to `logs/` and stdout.

### ABSA cache (required for agentic methods; also for ABSA quality eval)

`make data` exports **train-scoped** targets → `data/processed/.../absa_targets.jsonl` (only reviews used by the KG, not the full 24M-review category).

Default backend is **`llm_only`** (extract→judge, two Ollama calls per review). A **`hybrid`** backend (PyABSA classical tool + LLM validate/repair) is available after optional ML install and a quality gate (see below).

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b   # optional: ABSA only (LLM_MODEL_SMALL in .env)
make data              # also writes absa_targets.jsonl
make absa              # scoped targets; writes results/absa_summary.json + absa_preview.html
make absa-preview      # refresh summary/preview from cache without re-running extraction

# Hybrid backend (optional — GPU recommended):
pip install -e ".[absa-ml]"   # pins transformers<4.30, update-checker<1 for PyABSA
make warmup-absa              # prefetch PyABSA checkpoint (~800MB download once)
# Set absa.backend: hybrid in configs/default.yaml after quality gate passes

# Dev smoke (~5 min), then open charts:
# python3 scripts/run_absa.py --config configs/default.yaml --max-reviews 100
# xdg-open results/absa_preview.html
# cat results/absa_summary.json   # sentiment + aspects series for plotting

# Reset ABSA cache + manifest + error log after backend/version change:
# make clean-absa
```

**Benchmark & quality gate** (compare `llm_only` vs `hybrid` before flipping default):

```bash
make absa-benchmark           # → results/absa_latency.json (latency, repair_rate, speedup)
make absa-quality-compare     # → results/absa_quality_comparison.json (macro F1 delta; gate ≤2pt)
```

### Ranking experiments

Canonical pipeline for the **LangGraph 4-agent** primary method:

```bash
make data
make absa-targets && make absa
make neo4j && make load-kg
make experiment        # or emorecagent run below
```

Processed splits include `verified_purchase` on every row (re-run `make data` after upgrading).

**Two evaluation protocols** (no change to `make data` / `make absa`):

| Protocol | Config | Use case |
|----------|--------|----------|
| **A — EmoRecAgent** | `configs/default.yaml` | Internal ablations: `per_row`, `verified_only`, `aggregation: row_mean` |
| **B — Paper baseline** | `configs/paper_baseline.yaml` | Compare vs LightGCN: `user_batch`, full catalog, `verified_only: false`, macro **user-mean** @10/@20 |

Both protocols also run **sampled eval** (1 positive + 100 negatives) when `eval.n_negatives: 100`; results appear under `"sampled"` in the JSON. Disable with `--no-sampled-eval`.

Metrics always include **HR@10** and **NDCG@10** (via `k_values`); existing @5/@20 and avg_hr are unchanged. The **sampled** block additionally reports hr/mrr/ndcg/recall at **@1, @3, @5** (plus merged `k_values`) via `eval.sampled_k_values`.

```bash
# Protocol A (default)
make experiment
python3 scripts/run_experiment.py --config configs/default.yaml --method emorecagent ...

# Protocol B (paper-aligned CF baselines)
make experiment-paper                    # SVD @20, user-batch
make experiment-paper METHOD=itemknn
```

By default (`eval.verified_only: true` in Protocol A), experiments evaluate **only** test rows with `verified_purchase=true`. Pass `--include-unverified` to use the full test set without switching configs.

Standard metrics: HR@K, **AvgHR@1,3,5**, NDCG@K, Recall@K, MRR@K at K ∈ {5, 10, 20}. Results JSON includes row-mean (`means`), user-mean (`means_per_user`), and bootstrap CI (`ci_per_user`).

```bash
make experiment        # SVD baseline → results/svd.json

# Full LangGraph system (requires Neo4j + Ollama; add --cumulative-history for leakage-safe eval)
python3 scripts/run_experiment.py \
  --config configs/default.yaml \
  --method emorecagent \
  --split data/processed/Beauty_and_Personal_Care \
  --out results/emorecagent.json \
  --cumulative-history

# Fast numeric-only ablation / CI smoke (no Neo4j):
python3 scripts/run_experiment.py \
  --config configs/default.yaml \
  --method emorecagent_fast \
  --split data/processed/Beauty_and_Personal_Care \
  --out results/emorecagent_fast.json

### HGT neural retriever (`emorecagent_hgt`)

Optional pyHGT-style heterogeneous graph encoder for Top-50 candidate retrieval under the same 4-agent LangGraph pipeline. **ABSA cache is read-only** — do not run `make clean-absa` before this path.

```bash
pip install -e ".[hgt]"    # torch + torch-geometric (+ sentence-transformers)
make data
make absa                  # existing cache; HGT only reads it
make build-hgt-graph       # → data/processed/.../hgt/
make train-hgt             # BPR link prediction → checkpoint + embeddings
make experiment-hgt        # memory KG + HGT retriever → results/emorecagent_hgt.json
```

Dev smoke: `python3 scripts/build_hgt_graph.py --max-users 10 --max-items 20`

# Dev cap (config experiment.max_test_rows or CLI):
python3 scripts/run_experiment.py ... --method emorecagent --max-test-rows 50

Experiment logs include per-stage LangGraph timing (`absa`, `profiling`, `reasoning`, `reflection`, `explanation`, `numeric_tail`) every `progress_interval` rows and at the end.

make ablations         # factorial ablation grid → results/ablations/*.json
```

Methods: `popularity`, `itemknn`, `svd` / `base_cf`, `sequential`, `aspect_aware`, `emorecagent` (LangGraph), `emorecagent_fast` (numeric profiling only).

**Compare two runs** (paired bootstrap significance):

```bash
make compare A=results/svd.json B=results/emorecagent.json METRIC=ndcg@10
```

**Shift-subpopulation report** (dynamic-mechanism slice; requires user signals JSON):

```bash
python3 scripts/report_shift_subset.py \
  --results results/emorecagent.json \
  --signals data/processed/Beauty_and_Personal_Care/signals_export.json \
  --metric ndcg@10
```

### ABSA quality evaluation (gold labels)

1. Sample reviews for manual labeling:

```bash
make sample-gold       # → data/labeled/absa_gold_candidates.jsonl
```

2. Label `(aspect, sentiment)` triples → save as `data/labeled/absa_gold.jsonl` (see [data/labeled/README.md](data/labeled/README.md)).

3. Run ABSA on gold review ids, then evaluate:

```bash
make absa
make absa-quality      # → results/absa_quality.json
```

Optional dual-annotation QA:

```bash
python3 scripts/eval_absa_quality.py \
  --dual-annotation data/labeled/absa_gold_v1.jsonl data/labeled/absa_gold_v2.jsonl
```

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for canonical metric formulas, aggregation rules (row-mean vs user-mean), and paper table templates.

## Project layout

| Path | Purpose |
|------|---------|
| `src/emorecagent/agents/` | ABSA, profiling, reasoning, reflection agents |
| `src/emorecagent/graph/` | LangGraph orchestration (U9) |
| `src/emorecagent/explain/` | Rationalized explanations (U10) |
| `src/emorecagent/scoring/` | Dynamic weights + \(S(u,i)\) |
| `src/emorecagent/eval/` | Metrics, bootstrap CI, faithfulness, experiment runner |
| `data/labeled/` | ABSA gold candidates + labels |
| `configs/` | Experiment + ablation YAML |
| `scripts/` | `run_experiment`, `run_absa`, `benchmark_absa_latency`, `compare_absa_quality`, `compare_results` |

## Hardware notes

Tested on a single GPU workstation (RTX 5070 Ti). ABSA over the full agentic subset is LLM-bound; use `data.max_users` / `data.max_items` in `configs/default.yaml` to cap runtime during development. CF baselines and the numeric eval harness run CPU-only.

## Citation

TBD — preprint link will be added before submission.
