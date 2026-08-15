#!/usr/bin/env bash
# Sharded multi-process precompute of T_u cache, then merge into the main JSONL.
#
# Usage (from repo root):
#   ./scripts/precompute_tu_cache_sharded.sh train
#   N=4 LOG_DIR=logs/Yelp ./scripts/precompute_tu_cache_sharded.sh test
#   DATA_SPLIT=valid N=1 ./scripts/precompute_tu_cache_sharded.sh
#   ./scripts/precompute_tu_cache_sharded.sh --split train -n 4
#
# Split (first match wins):
#   1) positional: train | test | valid
#   2) --split / -s
#   3) env DATA_SPLIT or TU_SPLIT
#   4) env SPLIT only if train|test|valid (README's SPLIT=data/processed/... is ignored)
#
# Env: ALIGN_CONFIG, CONFIG, LOG_DIR, N (or -n), NO_LLM, MERGE_ONLY, SKIP_MERGE,
#      OVERWRITE (1 = purge/recompute split keys; use for fixing T_u leakage), PYTHON

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: precompute_tu_cache_sharded.sh [OPTIONS] [train|test|valid]

Options:
  -s, --split SPLIT   train | test | valid  (default: train)
  -n, --num-shards N  shard count (default: 3)
  -h, --help          show this help

Env:
  OVERWRITE=1         recompute keys for this split (fix T_u leakage)

Examples:
  N=4 LOG_DIR=logs/Yelp ./scripts/precompute_tu_cache_sharded.sh train
  OVERWRITE=1 N=4 LOG_DIR=logs/Yelp ./scripts/precompute_tu_cache_sharded.sh test
  DATA_SPLIT=test N=4 ./scripts/precompute_tu_cache_sharded.sh
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${PYTHONPATH:-src}"

CONFIG="${ALIGN_CONFIG:-${CONFIG:-configs/categories/Yelp.yaml}}"
LOG_DIR="${LOG_DIR:-logs/Yelp}"
N="${N:-3}"
NO_LLM="${NO_LLM:-1}"
MERGE_ONLY="${MERGE_ONLY:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
OVERWRITE="${OVERWRITE:-0}"
PYTHON="${PYTHON:-python3}"

CLI_SPLIT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    train|test|valid)
      CLI_SPLIT="$1"
      shift
      ;;
    -s|--split)
      CLI_SPLIT="${2:?--split requires train|test|valid}"
      shift 2
      ;;
    -n|--num-shards)
      N="${2:?--num-shards requires a number}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

resolve_split() {
  if [[ -n "$CLI_SPLIT" ]]; then
    echo "$CLI_SPLIT"
    return
  fi
  if [[ -n "${DATA_SPLIT:-}" ]]; then
    echo "$DATA_SPLIT"
    return
  fi
  if [[ -n "${TU_SPLIT:-}" ]]; then
    echo "$TU_SPLIT"
    return
  fi
  if [[ "${SPLIT:-}" == "train" || "${SPLIT:-}" == "test" || "${SPLIT:-}" == "valid" ]]; then
    echo "$SPLIT"
    return
  fi
  if [[ -n "${SPLIT:-}" ]]; then
    echo "[precompute_tu_cache_sharded] ignoring SPLIT='$SPLIT' (data dir);" \
      "pass train|test|valid as arg or DATA_SPLIT=..." >&2
  fi
  echo "train"
}

DATA_SPLIT="$(resolve_split)"
case "$DATA_SPLIT" in
  train|test|valid) ;;
  *)
    echo "Invalid split: '$DATA_SPLIT' (use train, test, or valid)" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
  echo "N must be a positive integer (got: $N)" >&2
  exit 2
fi

NO_LLM_FLAG=()
if [[ "$NO_LLM" == "1" ]]; then
  NO_LLM_FLAG=(--no-llm)
fi
OVERWRITE_FLAG=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

echo "[precompute_tu_cache_sharded] config=$CONFIG split=$DATA_SPLIT N=$N log_dir=$LOG_DIR overwrite=$OVERWRITE"

if [[ "$MERGE_ONLY" != "1" ]]; then
  pids=()
  for i in $(seq 0 $((N - 1))); do
    echo "[precompute_tu_cache_sharded] starting shard $i/$N"
    "$PYTHON" scripts/precompute_tu_cache.py \
      --config "$CONFIG" \
      --split "$DATA_SPLIT" \
      "${NO_LLM_FLAG[@]}" \
      "${OVERWRITE_FLAG[@]}" \
      --num-shards "$N" \
      --shard-id "$i" \
      --log-dir "$LOG_DIR" &
    pids+=("$!")
  done

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[precompute_tu_cache_sharded] shard pid=$pid failed" >&2
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[precompute_tu_cache_sharded] one or more shards failed; skip merge" >&2
    exit 1
  fi
fi

if [[ "$SKIP_MERGE" != "1" ]]; then
  echo "[precompute_tu_cache_sharded] merging $N shards into tu_cache.jsonl"
  "$PYTHON" scripts/precompute_tu_cache.py \
    --config "$CONFIG" \
    --merge-shards \
    --num-shards "$N" \
    "${OVERWRITE_FLAG[@]}" \
    --log-dir "$LOG_DIR"
fi

echo "[precompute_tu_cache_sharded] done"
