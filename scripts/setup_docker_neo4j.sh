#!/usr/bin/env bash
# One-time Docker + Neo4j setup for EmoRecAgent.
# Requires sudo (password prompt). Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[setup] EmoRecAgent Neo4j bootstrap"

if ! command -v docker >/dev/null 2>&1; then
  echo "[setup] ERROR: docker CLI not found. Install Docker Engine first."
  exit 1
fi

if ! systemctl is-active --quiet docker 2>/dev/null; then
  echo "[setup] Starting docker service..."
  sudo systemctl enable --now docker
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "[setup] Installing docker-compose-plugin..."
  sudo apt-get update -qq
  sudo apt-get install -y docker-compose-plugin
fi

if ! id -nG "$USER" | grep -qw docker; then
  echo "[setup] Adding $USER to docker group (re-login or 'newgrp docker' afterward)..."
  sudo usermod -aG docker "$USER"
  ADDED_GROUP=1
else
  ADDED_GROUP=0
fi

# Use sudo for compose when the current shell lacks docker-group membership.
compose() {
  if docker ps >/dev/null 2>&1; then
    docker compose "$@"
  else
    sudo docker compose "$@"
  fi
}

if ss -tln 2>/dev/null | grep -q ':7688 '; then
  echo "[setup] Port 7688 already in use — stop the conflicting service first."
  exit 1
fi

if ss -tln 2>/dev/null | grep -q ':7687 '; then
  echo "[setup] NOTE: another Neo4j (or service) is on :7687."
  echo "         This project uses host :7688 / :7475 to avoid conflicts."
fi

compose up -d
compose ps

echo ""
echo "[setup] Waiting for Neo4j healthcheck..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:7475 >/dev/null 2>&1; then
    echo "[setup] Neo4j HTTP ready at http://localhost:7475"
    break
  fi
  sleep 2
  if [[ $i -eq 30 ]]; then
    echo "[setup] WARN: HTTP not ready yet — check: docker compose logs neo4j"
  fi
done

if [[ -f .env ]]; then
  if grep -q 'bolt://localhost:7687' .env; then
    sed -i 's|bolt://localhost:7687|bolt://localhost:7688|g' .env
    echo "[setup] Updated .env NEO4J_URI -> bolt://localhost:7688"
  fi
fi

echo ""
echo "[setup] Verify with:"
echo "  PYTHONPATH=src python3 scripts/verify_neo4j.py"
if [[ "$ADDED_GROUP" -eq 1 ]]; then
  echo ""
  echo "[setup] Docker group added. For passwordless docker in NEW terminals:"
  echo "  newgrp docker    # or log out and back in"
fi
