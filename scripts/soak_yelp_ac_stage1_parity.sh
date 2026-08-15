#!/usr/bin/env bash
# One-liner launcher: soak G→H→I→J in background.
# Safe to re-run: skips finished labels (attempt_*_eval_done.flag).
# Do NOT start a second copy while train_tisasrec_stage1 is already running.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAIN=${EMOREC_MAIN:-/home/ai/ethan/EmoRecAgent}
LOG_DIR=$MAIN/logs/Yelp_AC
mkdir -p "$LOG_DIR"

if pgrep -f "train_tisasrec_stage1.py.*Yelp_AC" >/dev/null 2>&1 \
  || pgrep -f "run_yelp_ac_stage1_parity_queue|parity_arch_queue" >/dev/null 2>&1; then
  echo "A parity train/queue is already running. Refusing to start a duplicate."
  echo "Check:  pgrep -af 'parity_arch_queue|train_tisasrec_stage1'"
  echo "Log:    $LOG_DIR/parity_arch_queue.log"
  exit 1
fi

chmod +x "$ROOT/scripts/run_yelp_ac_stage1_parity_queue.sh"
nohup bash "$ROOT/scripts/run_yelp_ac_stage1_parity_queue.sh" \
  >"$LOG_DIR/parity_arch_queue.log" 2>&1 &
echo "started pid=$!"
echo "log=$LOG_DIR/parity_arch_queue.log"
echo "done flag → $LOG_DIR/parity_queue_done.flag"
