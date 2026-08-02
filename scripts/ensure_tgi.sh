#!/usr/bin/env bash
# Start TGI if not already healthy (idempotent; safe to call before experiments).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TGI_BASE_URL="${TGI_BASE_URL:-http://localhost:8080}"
PROFILE="${TGI_PROFILE:-tgi}"

if curl -sf "${TGI_BASE_URL}/health" >/dev/null 2>&1; then
  echo "TGI ready at ${TGI_BASE_URL}"
  exit 0
fi

echo "TGI not reachable at ${TGI_BASE_URL} — starting (profile=${PROFILE})..."
docker compose --profile "${PROFILE}" up -d tgi

echo "Waiting for TGI at ${TGI_BASE_URL}/health ..."
for _ in $(seq 1 60); do
  if curl -sf "${TGI_BASE_URL}/health" >/dev/null 2>&1; then
    echo "TGI is ready."
    curl -sf "${TGI_BASE_URL}/v1/models" | head -c 400 || true
    echo
    exit 0
  fi
  sleep 5
done

echo "TGI did not become healthy in time." >&2
exit 1
