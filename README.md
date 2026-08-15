# EmoRecAgent

TiSASRec sequential recommender with a **Stage-2 review-aware LLM rerank** over Stage-1 top-K.

**Default pipeline (Option B):** RecBole TiSASRec **CE** Stage-1 (full-catalog softmax; use the already-trained Option B bundle) + Stage-2 paper §III.F shell (`guardrail_mode: context_dependent`, fusion α=0.7, pool K=300, **promote_swap** + **3+4+1** narrow/T_u-snippets/**deep scorecard** reason-then-pick + LLM confidence gate, relaxed Eq. 21). Primary eval is full-catalog ranking (Protocol B: `user_batch`, `user_mean`).

Legacy Option A (in-repo ERA / `reorder_head`): see `[configs/legacy/](configs/legacy/)`.

## Requirements

- Python 3.11+ (conda env **ERA** recommended)
- NVIDIA GPU for Stage 1 training
- Docker optional (TGI for ABSA LLM / Stage-2 LLM eval)
- Amazon Reviews 2023 raw dumps under `data/amazon-reviews-2023/raw/`



## Setup

```bash
conda activate ERA
cp .env.example .env
export PYTHONPATH=src
pip install -e ".[dev,align]"
pytest -q
```

Download raw data if needed:

```bash
python3 scripts/download_amazon_reviews.py --category Beauty_and_Personal_Care
python3 scripts/download_amazon_reviews.py --category Sports_and_Outdoors
python3 scripts/download_amazon_reviews.py --category Toys_and_Games

# Yelp Open Dataset (accept ToS in browser, then unpack)
# https://business.yelp.com/data/resources/open-dataset/
# Expects: data/yelp-open-dataset/raw/yelp_dataset/yelp_academic_dataset_*.json
python3 scripts/download_yelp_open_dataset.py --archive /path/to/yelp_dataset.tar
# build_dataset reads native Yelp JSON via configs/categories/Yelp.yaml
```



## TGI (Docker LLM server)

ABSA hybrid and Stage-2 rerank call an **OpenAI-compatible** endpoint. The repo ships [Text Generation Inference](https://github.com/huggingface/text-generation-inference) via `docker-compose.yml`.

**Prerequisites:** Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), GPU with enough VRAM for the chosen model.

### Configure `.env`

```bash
cp .env.example .env
```


| Variable                       | Role                                                                 |
| ------------------------------ | -------------------------------------------------------------------- |
| `TGI_BASE_URL`                 | Stage-2 / main LLM (default `http://localhost:8080`)                 |
| `TGI_BASE_URL_SMALL`           | ABSA validate/repair when set (e.g. `http://localhost:8081`)         |
| `LLM_MODEL`                    | Model id sent in API calls (must match loaded weights)               |
| `LLM_MODEL_SMALL`              | ABSA model id (default `Qwen/Qwen2.5-3B-Instruct`)                   |
| `TGI_MODEL_ID`                 | HF model loaded by the **tgi** container                             |
| `TGI_MODEL_SMALL_ID`           | HF model loaded by the **tgi-small** container                       |
| `TGI_MAX_CONCURRENT_REQUESTS`  | Concurrent requests TGI may batch (default `128`)                    |
| `TGI_MAX_BATCH_PREFILL_TOKENS` | Prefill batch cap for **tgi** (default `8192`; raise if VRAM allows) |


`configs/default.yaml` → `tgi.base_url` / `llm.model*` are overridden by these env vars when set.

### Start (7B — Stage-2 rerank)

```bash
# Quick start (waits for /health)
make tgi
# or:
bash scripts/setup_tgi.sh

# Manual
docker compose --profile tgi up -d tgi
curl -sf http://localhost:8080/health && curl -s http://localhost:8080/v1/models | head
```

Container: `emorecagent-tgi` · image `ghcr.io/huggingface/text-generation-inference:3.2.1` · host port **8080** → container 80.

### Start (3B — ABSA, optimized for JSON-grammar validate/repair)

Uses a dedicated container on **:8081**. On a single GPU, **stop the 7B TGI first** (otherwise VRAM will OOM). `setup_tgi_absa.sh` does that automatically.

```bash
bash scripts/setup_tgi_absa.sh
# or: make tgi-absa

curl -sf http://localhost:8081/health
curl -s http://localhost:8081/info | head -c 400
```

Client caps ABSA generation (`max_tokens` 512 validate / 1024 repair). Truncated JSON is salvaged to complete triples when possible.


| Variable                             | Default | Role                                         |
| ------------------------------------ | ------- | -------------------------------------------- |
| `TGI_SMALL_MAX_CONCURRENT_REQUESTS`  | `32`    | Match `--workers`                            |
| `TGI_SMALL_MAX_BATCH_PREFILL_TOKENS` | `16384` | Prefill budget (smaller → lower latency)     |
| `TGI_SMALL_MAX_BATCH_TOTAL_TOKENS`   | `32768` | Total batch token cap                        |
| `TGI_SMALL_MAX_TOTAL_TOKENS`         | `8192`  | Caps context; leaves room for ~1k new tokens |
| `ABSA_WORKERS`                       | `32`    | `run_absa.py --workers` / `make absa`        |


```bash
python3 scripts/run_absa.py --config "$CONFIG" --log-dir "$LOG_DIR" --workers 32
# or: make absa ABSA_WORKERS=32
```

If you see `Request timed out` / `hybrid_llm_fallback`, lower workers to 16 (do not push 64 until timeouts are gone). Cache hits resume where you left off.

### Throughput notes

After changing TGI env vars, recreate the container (`setup_tgi_absa.sh`). Watch `nvidia-smi` and ABSA logs for timeouts.

**Stage-2 eval (7B TGI):** pass `--parallel-workers N` to `run_experiment.py` so multiple users hit TGI at once (`user_batch` = concurrent users; `per_row` = concurrent rows). Start at `8`; raise toward `16`–`32` if logs stay free of timeouts. Keep `N ≤ TGI_MAX_CONCURRENT_REQUESTS` (default `128`). ABSA uses `--workers` on the 3B container instead.

### Ops

```bash
make tgi-absa            # 3B for ABSA (stops 7B)
make tgi                 # 7B for Stage-2 (stop tgi-small first if VRAM tight)
make ensure-tgi          # start 7B only if /health is down
docker compose --profile tgi logs -f tgi
docker compose --profile tgi-small logs -f tgi-small
docker compose --profile tgi stop tgi
docker compose --profile tgi-small stop tgi-small
```

First start downloads weights from Hugging Face (can take several minutes). Check logs if `/health` stays down.

## Categories

One YAML per benchmark. Artifacts live under `data/processed/<Category>/tisasrec_option_b/`,
`results/<Category>/`, and `logs/<Category>/`. Paths must not cross categories —
`check_emorecagent_prereqs.py` fails if they do.


| Category            | `CONFIG`                                           | ABSA    | `T_u` source  | `test_history` | RecBole train YAML                                             |
| ------------------- | -------------------------------------------------- | ------- | ------------- | -------------- | -------------------------------------------------------------- |
| Beauty              | `configs/categories/Beauty_and_Personal_Care.yaml` | on      | review + ABSA | train_valid    | `baseline/RecBole-TiSASRec/configs/paper_tisasrec_beauty.yaml` |
| Sports              | `configs/categories/Sports_and_Outdoors.yaml`      | on      | review + ABSA | train_valid    | `…/paper_tisasrec_sports.yaml`                                 |
| Toys                | `configs/categories/Toys_and_Games.yaml`           | on      | review + ABSA | train_valid    | `…/paper_tisasrec_toys.yaml`                                   |
| Yelp (reviews)      | `configs/categories/Yelp.yaml`                     | on      | review + ABSA | train          | `…/paper_tisasrec_yelp_reviews.yaml`                           |
| Yelp_AC (no-review) | `configs/categories/Yelp_AC.yaml`                  | **off** | item metadata | train_valid    | `…/paper_tisasrec_yelp_ac.yaml`                                |




### Shell env (pick one category)

```bash
export PYTHONPATH=src
export ERA_PY=/home/ai/anaconda3/envs/ERA/bin/python   # RecBole deps
export PARALLEL_WORKERS=8   # Stage-2 concurrent TGI calls
export ABSA_WORKERS=32      # hybrid ABSA only (skip on Yelp_AC)

# --- Beauty ---
export CAT=Beauty_and_Personal_Care
export CONFIG=configs/categories/$CAT.yaml
export SPLIT=data/processed/$CAT
export OUT=results/$CAT
export LOG_DIR=logs/$CAT

# --- Sports ---
# export CAT=Sports_and_Outdoors
# export CONFIG=configs/categories/$CAT.yaml
# export SPLIT=data/processed/$CAT
# export OUT=results/$CAT
# export LOG_DIR=logs/$CAT

# --- Toys ---
# export CAT=Toys_and_Games
# …

# --- Yelp (reviews; needs Open Dataset) ---
# export CAT=Yelp
# …

# --- Yelp_AC (no ABSA; metadata T_u; LOO history = train+valid) ---
# export CAT=Yelp_AC
# …

mkdir -p "$LOG_DIR" "$OUT" "$SPLIT/tisasrec_option_b"
```



## Pipeline

Run after setting env vars above. Every step reads `$CONFIG` / `$LOG_DIR` / `$SPLIT`.

### 1. Dataset (+ ABSA when enabled)

```bash
python3 scripts/build_dataset.py --config "$CONFIG" --log-dir "$LOG_DIR"

# Review tracks only (Beauty / Sports / Toys / Yelp). Skip on Yelp_AC.
# Hybrid ABSA → TGI 3B (:8081). Classical → no LLM.
python3 scripts/run_absa.py --config "$CONFIG" --log-dir "$LOG_DIR" --workers "${ABSA_WORKERS:-32}"
# python3 scripts/run_absa.py --config "$CONFIG" --log-dir "$LOG_DIR" \
#   --backend classical --device cuda --batch-size 32
```

Writes `train/valid/test.jsonl` (and `absa_*.sqlite` when ABSA runs) under `$SPLIT`.

### 2. Stage-1 — RecBole TiSASRec CE + export bundle

Reuse the already-trained Option B bundle when present. Only retrain if the
bundle is missing (hparams in `paper_tisasrec_*.yaml` match that checkpoint).

```bash
"$ERA_PY" scripts/train_recbole_stage1_option_b.py \
  --config "$CONFIG" \
  --python "$ERA_PY" \
  --log-dir "$LOG_DIR" \
  --metrics-out "$OUT/recbole_tisasrec_option_b.json"
# Re-export only (existing RecBole .pth):
# "$ERA_PY" scripts/train_recbole_stage1_option_b.py --config "$CONFIG" --skip-train
```

Writes `$SPLIT/tisasrec_option_b/recbole_stage1_bundle.pt`, `e_i_matrix.pt`,
`item_token_to_idx.json`.

### 3. Stage-2 artifacts (T_u, cross-user, Alignment MLP)

InfoNCE needs **train**-split T_u keys; eval needs **test**-split T_u.
Amazon chrono configs use `test_history: train_valid` at Stage-1/2 eval (valid before
test → no leakage). Large train caches: sharded helper (`N=2–3` on ~24–62 GiB RAM).

```bash
# Train T_u: ABSA template is fine for InfoNCE (fast, --no-llm default in sharded script).
CONFIG="$CONFIG" LOG_DIR="$LOG_DIR" N=3 ./scripts/precompute_tu_cache_sharded.sh train

# Test T_u (P1): prefer LLM manifesto for richer Stage-2 preference text.
# Requires 7B TGI (:8080). Fallback template: omit NO_LLM=0 (keep default --no-llm).
make tgi   # or ensure TGI is up
NO_LLM=0 CONFIG="$CONFIG" LOG_DIR="$LOG_DIR" N=2 \
  ./scripts/precompute_tu_cache_sharded.sh test
# Rebuild after protocol / prompt changes:
# OVERWRITE=1 NO_LLM=0 CONFIG="$CONFIG" LOG_DIR="$LOG_DIR" N=2 \
#   ./scripts/precompute_tu_cache_sharded.sh test

python3 scripts/build_cross_user_lookup.py --config "$CONFIG"
# Alignment MLP InfoNCE — requires CUDA (ST encode + MLP on GPU; no CPU fallback)
"$ERA_PY" scripts/train_alignment_stage2_option_b.py \
  --config "$CONFIG" --log-dir "$LOG_DIR"
python3 scripts/check_emorecagent_prereqs.py --config "$CONFIG" --eval
```

Stage-2 LLM cards load Amazon `meta_*.jsonl` titles/categories (or RecBole `.item`
on Yelp_AC) into the paper evidence pack (titles + `π¹_rank`). Control flow stays
Algorithm 1: `guardrail_mode=context_dependent`, `fusion_alpha=0.7`,
`rerank_pool_k=300`, `llm_pool_cap=40`, `llm_rerank_mode=promote_swap`
(freeze π¹[:8], swap ≤2 picks into slots 9–10 — no insert-shift), **B4**
`llm_card_review_snippets=true` with **T_u-matched** snippets
(`llm_card_review_candidates=5`), **3+4+1** narrow shortlist
(`llm_narrow_cap=12`) + two-call **reason-then-pick**
(`llm_reason_then_pick=true`, `llm_reason_depth=deep`: rich extract +
contrastive scorecard with local fit+2 gate), **hybrid lexical gate**
(`llm_hybrid_gate_enabled=true`: cand overlap must beat displacee by δ,
stricter outside π¹ ranks 11–40). Beauty trial: `llm_narrow_cap=30`,
`llm_hybrid_overlap_delta=0` (softer shortlist / hybrid), plus
**argmax_llm_override** (`llm_pick_mode=argmax_llm_override`): quality-first LLM
scorecard (V4) may **override** T_u/overlap/LLM-gate constraints when evidence is
strong; otherwise falls back to **lexical_argmax** with `llm_hybrid_min_overlap=2`
and `promote_k=2` (slots 9–10). `llm_constraint_override=true` bypasses c_u/margin
skip and soft-accepts Eq. 21 when the frozen `protect_n` prefix is intact.
Primary metrics: hr/ndcg/recall @10 and @20.
`llm_gate_enabled=true`
(skip when Stage-1 margin high / c_u low), Eq. 21 with relaxed
(N_u,M_u). No catalog injection outside \pi^1[:K]. Re-runs that change
Stage-2 prompts should use `--fresh-checkpoint` so old rankings are not
resumed. Pick audit:

```bash
# Prefer ERA env for Stage-1 reload path.
"$ERA_PY" scripts/analyze_stage2_pick_audit.py \
  --baseline "$OUT/emorecagent_align_option_b_stage1_only.json" \
  --fused "$OUT/emorecagent_align_option_b.json" \
  --config "$CONFIG" --split "$SPLIT"
```

Oracle / near-miss ceiling (no re-rank dump):

```bash
python3 scripts/analyze_stage2_oracle_bound.py \
  --baseline "$OUT/emorecagent_align_option_b_stage1_only.json" \
  --fused "$OUT/emorecagent_align_option_b.json"
```



### 4. Eval — Stage-1-only vs Stage-2 LLM rerank

Needs 7B TGI (`make tgi` / `:8080`). Same `$CONFIG` for both; Stage-1 baseline uses
`--stage1-only` (no separate baseline YAML). Stage-2 knobs may change freely as
long as Algorithm 1 / Eqs. 16–21 stay intact; keep Stage-1 checkpoint fixed.

```bash
# Stage-1-only (no LLM / no fusion).
# Amazon chrono configs use test_history=train_valid (valid before test → no leakage).
python3 scripts/run_experiment.py \
  --config "$CONFIG" --method emorecagent_align --split "$SPLIT" \
  --stage1-only \
  --out "$OUT/emorecagent_align_option_b_stage1_only.json" \
  --eval-pass full --no-sampled-eval \
  --log-file "$LOG_DIR/experiment_emorecagent_align_option_b_stage1_only.log"

# fit φ
"$ERA_PY" scripts/analyze_item_potential.py --config "$CONFIG" --split "$SPLIT"

# Stage-2 LLM rerank — N concurrent users → TGI
# Use --fresh-checkpoint when Stage-2 prompts/knobs changed.
python3 scripts/run_experiment.py \
  --config "$CONFIG" --method emorecagent_align --split "$SPLIT" \
  --out "$OUT/emorecagent_align_option_b.json" \
  --eval-pass full --no-sampled-eval \
  --parallel-workers "${PARALLEL_WORKERS:-8}" \
  --fresh-checkpoint \
  --log-file "$LOG_DIR/experiment_emorecagent_align_option_b.log"

python3 scripts/compare_emorecagent_stage2.py \
  --baseline "$OUT/emorecagent_align_option_b_stage1_only.json" \
  --fused "$OUT/emorecagent_align_option_b.json"

# Cohort where Stage-1 already puts a relevant in top-K (use hr@100 vs K=300)
python3 scripts/analyze_stage2_pool_cohort.py \
  --baseline "$OUT/emorecagent_align_option_b_stage1_only.json" \
  --fused "$OUT/emorecagent_align_option_b.json" \
  --pool-k 100
```

Check Stage-2 `metadata.tisasrec_align`: `n_item_meta`, `mean_c_u`, `n_llm_calls`,
`guardrail_reject_rate`, `rerank_pool_k`, `guardrail_mode=context_dependent`.
Success: no overall regression vs Stage-1; **in_pool** hr@10/ndcg@10 ≥ 0.
Stage-2 still cannot help users whose relevant lies outside \pi^1[:K].

**Stage-2 (§III.F shell + promote_swap/3+4+1/deep/gate):** Stage-1 full-catalog →
x_u=\alpha s_u+(1-\alpha)p_u → c_u\to(N_u,M_u) → pool K=300 →
narrow outside-head by T_u overlap → rich extract → deep scorecard pick
(fit+2 vs displacee) → hybrid lexical gate → promote-swap C=40
(T_u-matched review snippet; freeze top-8, swap slots 9–10) → merge → Eq. 21
else fallback \pi^{(1)}.
Check metadata `n_stage2_swaps` / `n_stage2_empty_picks` /
`n_stage2_hybrid_blocked` / `n_stage2_hybrid_first_filtered` /
`n_stage2_lexical_argmax` / `n_stage2_llm_override` / `n_stage2_lexical_first`.
Compare Stage-1 vs Stage-2 on **hr@10, hr@20, ndcg@10, ndcg@20, recall@10,
recall@20**.
`--parallel-workers N` keeps `N` users in flight against TGI
(`N ≤ TGI_MAX_CONCURRENT_REQUESTS`).

## Yelp tracks

Two disjoint experiments — never share processed trees.


|                | `Yelp` (reviews)            | `Yelp_AC` (AC-TSR / no-review)                                                  |
| -------------- | --------------------------- | ------------------------------------------------------------------------------- |
| Source         | Yelp Open Dataset JSON      | RecBole `yelp.inter` + `yelp.item` under `data/yelp-open-dataset/raw/yelp_AC/…` |
| Filter / split | k=20; chrono 80/10/10       | 2019 window; k=5; leave-last-out                                                |
| ABSA           | required                    | **disabled**                                                                    |
| Stage-2 `T_u`  | review + ABSA               | item name / categories (`preference_source: item_metadata`)                     |
| Cross-user     | review-gated                | ID-only co-visit (`cross_user_mode: id_only`)                                   |
| Test history   | train-only                  | train+valid (RecBole LOO)                                                       |
| Artifacts      | `…/Yelp/tisasrec_option_b/` | `…/Yelp_AC/tisasrec_option_b/`                                                  |


Both use the same [Pipeline](#pipeline) with the matching `CAT` / `CONFIG`.

## Baselines (Protocol B)

Same processed `$SPLIT`. See `baseline/*/README-amazon.md`.

```bash
# TiSASRec BCE (upstream pytorch)
python3 baseline/TiSASRec.pytorch/amazon/run_experiment.py \
  --config baseline/TiSASRec.pytorch/configs/paper_tisasrec.yaml \
  --out results/paper/tisasrec.json

# TiSASRec CE via RecBole (AC-TSR fork) — Yelp_AC Table 1 recipe
"$ERA_PY" baseline/RecBole-TiSASRec/amazon/run_experiment.py \
  --config baseline/RecBole-TiSASRec/configs/paper_tisasrec_yelp_ac.yaml \
  --out results/Yelp_AC/recbole_tisasrec.json \
  --log-file logs/Yelp_AC/recbole_tisasrec.log

# TiSASRec CE via RecBole — Yelp (reviews)
"$ERA_PY" baseline/RecBole-TiSASRec/amazon/run_experiment.py \
  --config baseline/RecBole-TiSASRec/configs/paper_tisasrec_yelp_reviews.yaml \
  --out results/Yelp/recbole_tisasrec.json \
  --log-file logs/Yelp/recbole_tisasrec.log

# AC-TSR method (ACSASRec) — Yelp reviews track
"$ERA_PY" baseline/AC-TSR/amazon/run_experiment.py \
  --config baseline/AC-TSR/configs_amazon/paper_acsasrec_yelp.yaml \
  --out results/Yelp/ac_tsr_acsasrec.json \
  --log-file logs/Yelp/ac_tsr_acsasrec.log

python3 baseline/LightGCN-PyTorch/amazon/run_experiment.py \
  --config baseline/LightGCN-PyTorch/configs/paper_lightgcn.yaml \
  --out results/paper/lightgcn.json

python3 scripts/run_experiment.py \
  --config configs/legacy/paper_baseline.yaml \
  --method svd \
  --split "$SPLIT" \
  --out results/paper/svd.json \
  --log-dir "$LOG_DIR"
```



## Layout


| Path                              | Role                                  |
| --------------------------------- | ------------------------------------- |
| `src/emorecagent/tisasrec_align/` | Stage 1/2 train, rerank, eval         |
| `src/emorecagent/absa/`           | Aspect–sentiment extraction           |
| `configs/categories/`             | Option B: one YAML per benchmark      |
| `configs/legacy/`                 | Option A / ERA overlays (not default) |
| `scripts/`                        | CLI entry points                      |
| `baseline/`                       | External Protocol-B baselines         |
| `ref_paper/`                      | Method notes for writing              |




## Tests

```bash
export PYTHONPATH=src
pytest tests/tisasrec_align/ tests/test_config.py -q
```



## Baseline models (download separately)

Trained **baseline model weights / checkpoints are not stored in this GitHub
repository** (`baseline/` and large artifacts are gitignored). Download them
from the link below and place the archive contents under the repo root so paths
match the scripts in [Baselines (Protocol B)](#baselines-protocol-b) (typically
extract into `baseline/`).

**Download:** [Baseline models (external)](REPLACE_WITH_YOUR_DOWNLOAD_URL)

After download, verify expected trees exist (examples):

```bash
ls baseline/RecBole-TiSASRec/checkpoints/
ls baseline/TiSASRec.pytorch/  # if included in the pack
```

If the link above still shows `REPLACE_WITH_YOUR_DOWNLOAD_URL`, ask the
maintainer for the current sharing URL (Drive / Dropbox / object storage).