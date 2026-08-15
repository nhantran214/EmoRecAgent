#!/usr/bin/env bash
# Background launcher for U6 improve queue (K→N).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAIN=${EMOREC_MAIN:-/home/ai/ethan/EmoRecAgent}
LOG_DIR=$MAIN/logs/Yelp_AC
mkdir -p "$LOG_DIR"

if pgrep -f "train_tisasrec_stage1.py.*Yelp_AC" >/dev/null 2>&1 \
  || pgrep -f "run_yelp_ac_stage1_(parity|improve)_queue|parity_arch_queue" >/dev/null 2>&1; then
  echo "A Stage-1 train/queue is already running. Refusing duplicate."
  pgrep -af "train_tisasrec_stage1|parity_|improve_queue" || true
  exit 1
fi

chmod +x "$ROOT/scripts/run_yelp_ac_stage1_improve_queue.sh"
nohup bash "$ROOT/scripts/run_yelp_ac_stage1_improve_queue.sh" \
  >"$LOG_DIR/parity_improve_queue.log" 2>&1 &
echo "started pid=$!"
echo "log=$LOG_DIR/parity_improve_queue.log"
echo "done flag → $LOG_DIR/improve_queue_done.flag"
