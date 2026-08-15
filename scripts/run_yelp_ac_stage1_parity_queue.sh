#!/usr/bin/env bash
# Soak-run Yelp_AC Stage-1 paper-parity arch queue (G→H→I→J).
#
# Each step is single-factor vs Attempt D baseline (Xavier, wd=0, day time unit):
#   G  hidden_units=128
#   H  num_blocks=3
#   I  num_heads=4
#   J  early_stop_patience=20
#
# Stops early if R1 (Table 1) is met. Skips steps that already have
# logs/Yelp_AC/attempt_<label>_eval_done.flag
#
# Usage (from anywhere):
#   nohup bash scripts/run_yelp_ac_stage1_parity_queue.sh \
#     > logs/Yelp_AC/parity_arch_queue.log 2>&1 &
#
# Or via the one-liner in scripts/soak_yelp_ac_stage1_parity.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Prefer worktree checkout that owns this script; fall back to repo root.
WT=$(cd "$SCRIPT_DIR/.." && pwd)
MAIN=${EMOREC_MAIN:-/home/ai/ethan/EmoRecAgent}
if [[ -d $MAIN/logs/Yelp_AC ]]; then
  :
elif [[ -d $WT/logs/Yelp_AC ]]; then
  MAIN=$WT
else
  MAIN=$WT
fi

PY=${ERA_PYTHON:-/home/ai/anaconda3/envs/ERA/bin/python}
LOG_DIR=${LOG_DIR:-$MAIN/logs/Yelp_AC}
OUT=${OUT:-$MAIN/results/Yelp_AC}
CONFIG=$WT/configs/categories/Yelp_AC.yaml
YAML=$CONFIG
RUN_ROOT=data/processed/Yelp_AC/tisasrec_paper/runs

mkdir -p "$LOG_DIR" "$OUT"
cd "$WT"

echo "[parity-queue] WT=$WT MAIN=$MAIN PY=$PY"
echo "[parity-queue] started at $(date -Is)"

r1_pass() {
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("$OUT/tisasrec_paper_stage1_test.json")
m = json.loads(p.read_text())["metrics"]
ok = (
    m["link_recall_at_10"] >= 0.0618
    and m["link_recall_at_20"] >= 0.0909
    and m["link_ndcg_at_10"] >= 0.0387
    and m["link_ndcg_at_20"] >= 0.0460
)
raise SystemExit(0 if ok else 1)
PY
}

patch_yaml_d_baseline_plus() {
  local hidden=64 blocks=2 heads=2 wd=0.0 patience=10
  local kv
  for kv in "$@"; do
    case "$kv" in
      hidden_units=*) hidden=${kv#*=} ;;
      num_blocks=*) blocks=${kv#*=} ;;
      num_heads=*) heads=${kv#*=} ;;
      weight_decay=*) wd=${kv#*=} ;;
      early_stop_patience=*) patience=${kv#*=} ;;
      *) echo "unknown kv $kv"; exit 2 ;;
    esac
  done
  "$PY" - "$YAML" "$hidden" "$blocks" "$heads" "$wd" "$patience" <<'PY'
from pathlib import Path
import re, sys
path, hidden, blocks, heads, wd, patience = sys.argv[1:7]
text = Path(path).read_text()

def set_scalar(src: str, key: str, value: str) -> str:
    pat = rf"(^  {re.escape(key)}:\s*)([^\n#]+)"
    out, n = re.subn(pat, rf"\g<1>{value}", src, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"failed to set {key}={value} (matches={n})")
    return out

text = set_scalar(text, "weight_decay", wd)
text = set_scalar(text, "early_stop_patience", patience)
text = set_scalar(text, "num_blocks", blocks)
text = set_scalar(text, "num_heads", heads)
text = set_scalar(text, "hidden_units", hidden)
text = set_scalar(text, "weight_init", "xavier")
Path(path).write_text(text)
print(f"yaml -> hidden={hidden} blocks={blocks} heads={heads} wd={wd} patience={patience}")
PY
}

run_one() {
  local label=$1
  shift
  echo "========== START $label ($*) =========="
  patch_yaml_d_baseline_plus "$@"

  local stamp train_log
  stamp=$(date +%Y%m%d_%H%M%S)
  train_log=$LOG_DIR/train_attempt_${label}_${stamp}.stdout

  env -u LOG_DIR -u OUT -u CONFIG -u SPLIT PYTHONPATH="$WT/src" PYTHONUNBUFFERED=1 \
    "$PY" "$WT/scripts/train_tisasrec_stage1.py" --config "$CONFIG" --log-dir "$LOG_DIR" \
    >"$train_log" 2>&1 &
  local train_pid=$!
  echo "train_pid=$train_pid log=$train_log"
  wait "$train_pid"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "TRAIN FAILED rc=$rc"
    tail -40 "$train_log"
    exit "$rc"
  fi

  mkdir -p "$RUN_ROOT/$label"
  cp -a data/processed/Yelp_AC/tisasrec_paper/stage1_checkpoint.pt "$RUN_ROOT/$label/"
  cp -a data/processed/Yelp_AC/tisasrec_paper/e_i_matrix.pt "$RUN_ROOT/$label/" 2>/dev/null || true

  echo "[eval] $label ..."
  env -u LOG_DIR -u OUT -u CONFIG -u SPLIT PYTHONPATH="$WT/src" PYTHONUNBUFFERED=1 \
    "$PY" "$WT/scripts/eval_tisasrec_stage1_test.py" --config "$CONFIG" \
    --log-dir "$LOG_DIR" --out "$OUT/tisasrec_paper_stage1_test.json" \
    2>&1 | tee "$LOG_DIR/eval_attempt_${label}.log"

  cp -a "$OUT/tisasrec_paper_stage1_test.json" "$RUN_ROOT/$label/metrics.json"

  WT="$WT" MAIN="$MAIN" OUT="$OUT" LOG_DIR="$LOG_DIR" \
  "$PY" - "$label" "$train_log" <<'PY'
import json, os, re, sys
from pathlib import Path

label, train_log = sys.argv[1], Path(sys.argv[2])
wt = Path(os.environ["WT"])
main = Path(os.environ["MAIN"])
out = Path(os.environ["OUT"])
log_dir = Path(os.environ["LOG_DIR"])
m = json.loads((out / "tisasrec_paper_stage1_test.json").read_text())["metrics"]
r10, r20, n10, n20 = (
    m["link_recall_at_10"],
    m["link_recall_at_20"],
    m["link_ndcg_at_10"],
    m["link_ndcg_at_20"],
)
gate = r10 >= 0.0618 and r20 >= 0.0909 and n10 >= 0.0387 and n20 >= 0.0460
text = train_log.read_text(errors="replace")
best = re.search(r"best_valid_link_recall@10=([0-9.]+) at epoch=(\d+)", text)
best_s = f"valid peak R@10={best.group(1)} @{best.group(2)}" if best else "valid peak n/a"
notes = (
    f"Attempt {label}\n"
    f"Test R@10={r10:.4f} R@20={r20:.4f} N@10={n10:.4f} N@20={n20:.4f}\n"
    f"{best_s}\nR1={'PASS' if gate else 'FAIL'}\n"
)
(wt / f"data/processed/Yelp_AC/tisasrec_paper/runs/{label}/notes.md").write_text(notes)
print(notes)

change = {
    "G_hidden_128": "`hidden_units: 128` (vs D)",
    "H_blocks_3": "`num_blocks: 3` (vs D)",
    "I_heads_4": "`num_heads: 4` (vs D)",
    "J_patience_20": "`early_stop_patience: 20` (vs D)",
}.get(label, label)
row = (
    f"| {label} | {change} | {r10:.4f} | {r20:.4f} | {n10:.4f} | {n20:.4f} | "
    f"{'yes' if gate else 'no'} | {best_s} |"
)

for root in (wt, main):
    sc = root / "docs/yelp_ac_stage1_parity_scorecard.md"
    if not sc.is_file():
        continue
    s = sc.read_text()
    if f"| {label} |" in s:
        s = re.sub(rf"\| {re.escape(label)} \|.*\|", row, s, count=1)
    else:
        s = s.replace("\n## Queue (KTD6)", f"\n{row}\n\n## Queue (KTD6)")
    sc.write_text(s)

    gap = root / "docs/yelp_ac_stage1_gap.md"
    if gap.is_file():
        g = gap.read_text()
        short = label.split("_")[0]
        if f"| {short} |" not in g:
            g = re.sub(
                r"(\| F \|.*\|\n)",
                rf"\1| {short} | **`{change}`** | **{r10:.4f}** | **{n10:.4f}** |\n",
                g,
                count=1,
            )
        table = f"""Full-catalog leave-last-out test, `results/Yelp_AC/tisasrec_paper_stage1_test.json`
(30,449 test users, `verified_only: false`) — **Attempt {short}** (`{change}`,
ckpt `runs/{label}/`):

| Metric | Measured (Attempt {short}) | AC-TSR Table 1 | Gap |
|--------|----------------------------|----------------|-----|
| Recall@10 | {r10:.4f} | 0.0618 | {((r10 / 0.0618) - 1) * 100:+.0f}% |
| Recall@20 | {r20:.4f} | 0.0909 | {((r20 / 0.0909) - 1) * 100:+.0f}% |
| NDCG@10 | {n10:.4f} | 0.0387 | {((n10 / 0.0387) - 1) * 100:+.0f}% |
| NDCG@20 | {n20:.4f} | 0.0460 | {((n20 / 0.0460) - 1) * 100:+.0f}% |
"""
        g = re.sub(
            r"Full-catalog leave-last-out test.*?(?=\n## Cohort)",
            table + "\n",
            g,
            count=1,
            flags=re.S,
        )
        gap.write_text(g)

flag = log_dir / f"attempt_{label}_eval_done.flag"
flag.write_text(
    json.dumps(
        {"label": label, "r10": r10, "r20": r20, "n10": n10, "n20": n20, "r1": gate},
        indent=2,
    )
    + "\n"
)
print("wrote", flag)
raise SystemExit(0 if gate else 1)
PY
}

if [[ -f $MAIN/docs/yelp_ac_stage1_parity_scorecard.md ]]; then
  cp "$MAIN/docs/yelp_ac_stage1_parity_scorecard.md" "$WT/docs/yelp_ac_stage1_parity_scorecard.md" 2>/dev/null || true
fi

PYTHONPATH="$WT/src" "$PY" -m pytest \
  "$WT/tests/data/test_yelp_ac_config_isolation.py::test_yelp_ac_tisasrec_paper_config_uses_ce" \
  -q --tb=line

QUEUE=(
  "G_hidden_128|hidden_units=128"
  "H_blocks_3|num_blocks=3"
  "I_heads_4|num_heads=4"
  "J_patience_20|early_stop_patience=20"
)

for entry in "${QUEUE[@]}"; do
  label=${entry%%|*}
  args=${entry#*|}
  if [[ -f $LOG_DIR/attempt_${label}_eval_done.flag ]]; then
    echo "[skip] $label already has eval flag"
    if r1_pass; then
      echo "R1 already met; stopping queue"
      exit 0
    fi
    continue
  fi
  if run_one "$label" $args; then
    echo "R1 MET at $label — stopping queue"
    echo "R1_MET=$label" >"$LOG_DIR/parity_queue_done.flag"
    exit 0
  fi
  echo "R1 unmet after $label — continuing"
done

echo "QUEUE_EXHAUSTED" >"$LOG_DIR/parity_queue_done.flag"
echo "Arch/patience queue exhausted without R1 at $(date -Is)"
exit 0
