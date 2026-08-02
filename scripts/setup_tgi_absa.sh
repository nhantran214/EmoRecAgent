#!/usr/bin/env bash
# Start TGI 3B for ABSA. Frees GPU by stopping the 7B container first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load .env if present (docker compose also reads it).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

TGI_BASE_URL_SMALL="${TGI_BASE_URL_SMALL:-http://localhost:8081}"
ABSA_WORKERS="${ABSA_WORKERS:-32}"

echo "Stopping 7B TGI (if running) to free VRAM for ABSA 3B..."
docker compose --profile tgi stop tgi >/dev/null 2>&1 || true

echo "Starting TGI 3B (profile=tgi-small) for ABSA (force-recreate to apply .env)..."
docker compose --profile tgi-small up -d --force-recreate tgi-small

echo "Waiting for TGI at ${TGI_BASE_URL_SMALL}/health ..."
for _ in $(seq 1 90); do
  if curl -sf "${TGI_BASE_URL_SMALL}/health" >/dev/null 2>&1; then
    echo "TGI ABSA (3B) is ready."
    curl -sf "${TGI_BASE_URL_SMALL}/info" 2>/dev/null | head -c 500 || \
      curl -sf "${TGI_BASE_URL_SMALL}/v1/models" | head -c 400 || true
    echo
    echo "Run ABSA with:"
    echo "  python3 scripts/run_absa.py --config \"\$CONFIG\" --log-dir logs --workers ${ABSA_WORKERS}"
    exit 0
  fi
  sleep 5
done

echo "TGI ABSA (3B) did not become healthy in time. Check: docker compose --profile tgi-small logs -f tgi-small" >&2
exit 1
