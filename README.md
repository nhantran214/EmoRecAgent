# EmoRecAgent

TiSASRec sequential recommender with a **Stage-2 review-aware LLM rerank** over Stage-1 top-K. Primary eval is full-catalog ranking (Protocol B: `user_batch`, `user_mean`).

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

# Yelp Open Dataset (accept ToS in browser, then unpack — method unchanged)
# https://business.yelp.com/data/resources/open-dataset/
# Expects: data/yelp-open-dataset/raw/yelp_dataset/yelp_academic_dataset_*.json
python3 scripts/download_yelp_open_dataset.py --archive /path/to/yelp_dataset.tar
# build_dataset reads native Yelp JSON directly via configs/categories/Yelp.yaml
# optional filtered Amazon-shaped export: python3 scripts/prepare_yelp_dataset.py --cities Philadelphia
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
python3 scripts/run_absa.py --config "$CONFIG" --log-dir logs --workers 32
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

Method is identical across categories; only paths change. Artifacts stay under
`data/processed/<Category>/`, `results/<Category>/`, and `logs/<Category>/`
(Beauty uses repo-root `results/` / `logs/`). Paths must not cross categories —
`check_emorecagent_prereqs.py` fails if they do.


| Category            | `CONFIG`                                      | `ALIGN_CONFIG`                                 | `BASELINE_CONFIG`                                        | `SPLIT` / `OUT` / `LOG_DIR`                       |
| ------------------- | --------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------- |
| Beauty              | `configs/default.yaml`                        | `configs/emorecagent_align.yaml`               | `configs/emorecagent_stage1_baseline.yaml`               | `…/Beauty_and_Personal_Care` · `results` · `logs` |
| Sports              | `configs/categories/Sports_and_Outdoors.yaml` | `…/Sports_and_Outdoors_emorecagent_align.yaml` | `…/Sports_and_Outdoors_emorecagent_stage1_baseline.yaml` | `…/Sports_and_Outdoors`                           |
| Toys                | `configs/categories/Toys_and_Games.yaml`      | `…/Toys_and_Games_emorecagent_align.yaml`      | `…/Toys_and_Games_emorecagent_stage1_baseline.yaml`      | `…/Toys_and_Games`                                |
| Yelp (reviews)      | `configs/categories/Yelp.yaml`                | `…/Yelp_emorecagent_align.yaml`                | `…/Yelp_emorecagent_stage1_baseline.yaml`                | `…/Yelp`                                          |
| Yelp_AC (no-review) | `configs/categories/Yelp_AC.yaml`             | `…/Yelp_AC_emorecagent_align.yaml`             | `…/Yelp_AC_emorecagent_stage1_baseline.yaml`             | `…/Yelp_AC`                                       |


### Shell env (pick one category)

```bash
export PYTHONPATH=src
export PARALLEL_WORKERS=8   # Stage-2 concurrent TGI calls
export ABSA_WORKERS=32      # hybrid ABSA only (skip on Yelp_AC)

# --- Beauty ---
export CONFIG=configs/default.yaml
export ALIGN_CONFIG=configs/emorecagent_align.yaml
export BASELINE_CONFIG=configs/emorecagent_stage1_baseline.yaml
export SPLIT=data/processed/Beauty_and_Personal_Care
export OUT=results
export LOG_DIR=logs

# --- Sports ---
# export CONFIG=configs/categories/Sports_and_Outdoors.yaml
# export ALIGN_CONFIG=configs/categories/Sports_and_Outdoors_emorecagent_align.yaml
# export BASELINE_CONFIG=configs/categories/Sports_and_Outdoors_emorecagent_stage1_baseline.yaml
# export SPLIT=data/processed/Sports_and_Outdoors
# export OUT=results/Sports_and_Outdoors
# export LOG_DIR=logs/Sports_and_Outdoors

# --- Toys ---
# export CONFIG=configs/categories/Toys_and_Games.yaml
# export ALIGN_CONFIG=configs/categories/Toys_and_Games_emorecagent_align.yaml
# export BASELINE_CONFIG=configs/categories/Toys_and_Games_emorecagent_stage1_baseline.yaml
# export SPLIT=data/processed/Toys_and_Games
# export OUT=results/Toys_and_Games
# export LOG_DIR=logs/Toys_and_Games

# --- Yelp (reviews; needs Open Dataset) ---
# export CONFIG=configs/categories/Yelp.yaml
# export ALIGN_CONFIG=configs/categories/Yelp_emorecagent_align.yaml
# export BASELINE_CONFIG=configs/categories/Yelp_emorecagent_stage1_baseline.yaml
# export SPLIT=data/processed/Yelp
# export OUT=results/Yelp
# export LOG_DIR=logs/Yelp

# --- Yelp_AC (BCE track; no ABSA) ---
# export CONFIG=configs/categories/Yelp_AC.yaml
# export ALIGN_CONFIG=configs/categories/Yelp_AC_emorecagent_align.yaml
# export BASELINE_CONFIG=configs/categories/Yelp_AC_emorecagent_stage1_baseline.yaml
# export SPLIT=data/processed/Yelp_AC
# export OUT=results/Yelp_AC
# export LOG_DIR=logs/Yelp_AC

mkdir -p "$LOG_DIR" "$OUT"
```

For the AC-TSR paper reproduction on Yelp_AC (CE Stage-1), see [Yelp_AC paper track](#yelp_ac-paper-track-ce--preferred) below — it uses a different config set.

## Pipeline

Run after setting env vars above. Replace nothing else: every step reads `$CONFIG` / `$ALIGN_CONFIG` / `$LOG_DIR`.

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

### 2. Stage 1 — train + test

```bash
python3 scripts/check_emorecagent_prereqs.py --config "$CONFIG"
PYTHONUNBUFFERED=1 python3 scripts/train_tisasrec_stage1.py \
  --config "$CONFIG" --log-dir "$LOG_DIR"
PYTHONUNBUFFERED=1 python3 scripts/eval_tisasrec_stage1_test.py \
  --config "$CONFIG" \
  --log-dir "$LOG_DIR" \
  --log-file "$LOG_DIR/test_emorecagent_latest.log" \
  --out "$OUT/emorecagent_stage1_test.json"
```

Checkpoint: `$SPLIT/tisasrec_align/stage1_checkpoint.pt` (+ `e_i_matrix.pt`).

### 3. Stage 2 artifacts

```bash
# Preference text T_u (omit --no-llm for LLM manifesto). Yelp_AC uses metadata.
python3 scripts/precompute_tu_cache.py \
  --config "$CONFIG" --split test --no-llm --log-dir "$LOG_DIR"
python3 scripts/build_cross_user_lookup.py --config "$CONFIG"
```

### 4. Eval — Stage-1 baseline vs Stage-2 LLM rerank

Needs 7B TGI (`make tgi` / `:8080`). Smoke without LLM: prefix with `NO_LLM=1`.

```bash
python3 scripts/check_emorecagent_prereqs.py --config "$ALIGN_CONFIG" --eval

# Stage-1-only (no LLM)
python3 scripts/run_experiment.py \
  --config "$BASELINE_CONFIG" \
  --method emorecagent_align \
  --split "$SPLIT" \
  --out "$OUT/emorecagent_stage1_baseline.json" \
  --eval-pass full \
  --no-sampled-eval \
  --log-file "$LOG_DIR/experiment_emorecagent_stage1_baseline.log"

# Stage-2 LLM rerank — N concurrent users → TGI
python3 scripts/run_experiment.py \
  --config "$ALIGN_CONFIG" \
  --method emorecagent_align \
  --split "$SPLIT" \
  --out "$OUT/emorecagent_align.json" \
  --eval-pass full \
  --no-sampled-eval \
  --parallel-workers "${PARALLEL_WORKERS:-8}" \
  --log-file "$LOG_DIR/experiment_emorecagent_latest.log"

hoặc
python3 scripts/run_experiment.py \
  --config "$ALIGN_CONFIG" --method emorecagent_align \
  --split "$SPLIT" --out "$OUT/emorecagent_align_paper.json" \
  --eval-pass full --no-sampled-eval \
  --parallel-workers "${PARALLEL_WORKERS:-8}" \
  --log-file "$LOG_DIR/experiment_emorecagent_align_paper.log"

python3 scripts/compare_emorecagent_stage2.py \
  --baseline "$OUT/emorecagent_stage1_baseline.json" \
  --fused "$OUT/emorecagent_align.json"
```

**Stage-2 (`stage2_mode: rerank`):** Stage-1 full-catalog rank → top-K pool → optional
cross-user boosts → one LLM call per user (when preference text exists) → merge +
guardrail. `--parallel-workers N` keeps `N` users in flight against TGI
(`N ≤ TGI_MAX_CONCURRENT_REQUESTS`).

## Yelp tracks

Two disjoint experiments — never share processed trees.


|                | `Yelp` (reviews)                      | `Yelp_AC` (AC-TSR / no-review)                                                  |
| -------------- | ------------------------------------- | ------------------------------------------------------------------------------- |
| Source         | Yelp Open Dataset JSON                | RecBole `yelp.inter` + `yelp.item` under `data/yelp-open-dataset/raw/yelp_AC/…` |
| Filter / split | track-specific; k=20; chrono 80/10/10 | 2019 window; k=5; leave-last-out                                                |
| ABSA           | required                              | **disabled**                                                                    |
| Stage-2 `T_u`  | review + ABSA                         | item name / categories                                                          |
| Cross-user     | review-gated                          | ID-only co-visit                                                                |
| Test history   | train-only                            | train+valid (RecBole LOO)                                                       |
| Artifacts      | `data/processed/Yelp/`                | `data/processed/Yelp_AC/`                                                       |


`Yelp` follows the [Pipeline](#pipeline) with the Yelp env block above.

### Yelp_AC paper track (RecBole TiSASRec Stage-1 — preferred)

Stage-1 is **RecBole TiSASRec** from `baseline/RecBole-TiSASRec` (AC-TSR fork),
exported into an EmoRecAgent bundle for Stage-2. Amazon / Yelp-review keep the
in-repo ERA trainer (`stage1_backend: era`). Gap notes for the old ERA CE
attempt: `[docs/yelp_ac_stage1_gap.md](docs/yelp_ac_stage1_gap.md)`.

```bash
export PYTHONPATH=src
export PARALLEL_WORKERS=8
export PAPER_CONFIG=configs/categories/Yelp_AC_tisasrec_paper.yaml
export ALIGN_CONFIG=configs/categories/Yelp_AC_emorecagent_align_paper.yaml
export BASELINE_CONFIG=configs/categories/Yelp_AC_emorecagent_stage1_baseline_paper.yaml
export SPLIT=data/processed/Yelp_AC
export OUT=results/Yelp_AC
export LOG_DIR=logs/Yelp_AC
export ERA_PY=/home/ai/anaconda3/envs/ERA/bin/python
mkdir -p "$LOG_DIR" "$OUT" data/processed/Yelp_AC/tisasrec_paper

# 0) Data (once; shared with BCE track)
python3 scripts/build_dataset.py --config configs/categories/Yelp_AC.yaml --log-dir "$LOG_DIR"

# 1) Stage-1 = RecBole TiSASRec (train + export bundle for Stage-2)
"$ERA_PY" scripts/train_yelp_ac_recbole_stage1.py \
  --python "$ERA_PY" \
  --log-dir "$LOG_DIR" \
  --metrics-out "$OUT/recbole_tisasrec.json"
# Re-export only from an existing .pth:
# "$ERA_PY" scripts/train_yelp_ac_recbole_stage1.py --skip-train \
#   --checkpoint baseline/RecBole-TiSASRec/checkpoints/TiSASRec-….pth
python3 scripts/check_emorecagent_prereqs.py --config "$PAPER_CONFIG"

# 2) Stage-2 prereqs (metadata T_u; no ABSA)
python3 scripts/precompute_tu_cache.py \
  --config "$ALIGN_CONFIG" --split test --no-llm --log-dir "$LOG_DIR"
python3 scripts/build_cross_user_lookup.py --config "$ALIGN_CONFIG"
python3 scripts/check_emorecagent_prereqs.py --config "$ALIGN_CONFIG" --eval

# 3) Baseline vs Stage-2 LLM rerank (TGI :8080)
python3 scripts/run_experiment.py \
  --config "$BASELINE_CONFIG" \
  --method emorecagent_align \
  --split "$SPLIT" \
  --out "$OUT/emorecagent_align_paper_stage1_only.json" \
  --eval-pass full --no-sampled-eval \
  --log-file "$LOG_DIR/experiment_emorecagent_align_paper_stage1_only.log"
python3 scripts/run_experiment.py \
  --config "$ALIGN_CONFIG" \
  --method emorecagent_align \
  --split "$SPLIT" \
  --out "$OUT/emorecagent_align_paper.json" \
  --eval-pass full --no-sampled-eval \
  --parallel-workers "${PARALLEL_WORKERS:-8}" \
  --log-file "$LOG_DIR/experiment_emorecagent_align_paper.log"
python3 scripts/compare_emorecagent_stage2.py \
  --baseline "$OUT/emorecagent_align_paper_stage1_only.json" \
  --fused "$OUT/emorecagent_align_paper.json"
```

Stage-1 only (no Stage-2): set `CONFIG=$PAPER_CONFIG` and run steps 0–1.

### Yelp_AC BCE track (alternate)

Same as the [Pipeline](#pipeline) with the Yelp_AC env block (skip ABSA). Checkpoints
live under `tisasrec_align/`, not `tisasrec_paper/` — do not mix with the CE track.

## Baselines (Protocol B)

Same processed `$SPLIT`. See `baseline/*/README-amazon.md`.

```bash
# TiSASRec BCE (upstream pytorch)
python3 baseline/TiSASRec.pytorch/amazon/run_experiment.py \
  --config baseline/TiSASRec.pytorch/configs/paper_tisasrec.yaml \
  --out results/paper/tisasrec.json

# TiSASRec CE via RecBole (AC-TSR fork) — Yelp_AC Table 1 recipe
/home/ai/anaconda3/envs/ERA/bin/python \
  baseline/RecBole-TiSASRec/amazon/run_experiment.py \
  --config baseline/RecBole-TiSASRec/configs/paper_tisasrec_yelp_ac.yaml \
  --out results/Yelp_AC/recbole_tisasrec.json \
  --log-file logs/Yelp_AC/recbole_tisasrec.log

python3 baseline/LightGCN-PyTorch/amazon/run_experiment.py \
  --config baseline/LightGCN-PyTorch/configs/paper_lightgcn.yaml \
  --out results/paper/lightgcn.json

python3 scripts/run_experiment.py \
  --config configs/paper_baseline.yaml \
  --method svd \
  --split "$SPLIT" \
  --out results/paper/svd.json \
  --log-dir "$LOG_DIR"
```

## Layout


| Path                              | Role                                |
| --------------------------------- | ----------------------------------- |
| `src/emorecagent/tisasrec_align/` | Stage 1/2 train, rerank, eval       |
| `src/emorecagent/absa/`           | Aspect–sentiment extraction         |
| `configs/`                        | Hyperparameters + category overlays |
| `scripts/`                        | CLI entry points                    |
| `baseline/`                       | External Protocol-B baselines       |
| `ref_paper/`                      | Method notes for writing            |


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

