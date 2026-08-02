#!/usr/bin/env bash
# One-time Docker + Neo4j setup for EmoRecAgent.
# Requires sudo (password prompt). Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE_CMD=()

echo "[setup] EmoRecAgent Neo4j bootstrap"

if ! command -v docker >/dev/null 2>&1; then
  echo "[setup] ERROR: docker CLI not found. Install: sudo apt-get install -y docker.io"
  exit 1
fi

if ! systemctl is-active --quiet docker 2>/dev/null; then
  echo "[setup] Starting docker service..."
  sudo systemctl enable --now docker
fi

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
    return 0
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
    return 0
  fi
  return 1
}

install_compose() {
  echo "[setup] Docker Compose not found — installing..."
  sudo apt-get update -qq

  # Ubuntu 24.04 (docker.io from distro): package name is docker-compose-v2.
  if apt-cache show docker-compose-v2 >/dev/null 2>&1; then
    echo "[setup] Installing docker-compose-v2 (Ubuntu package)..."
    sudo apt-get install -y docker-compose-v2
    return 0
  fi

  # Docker CE official repo uses docker-compose-plugin.
  if apt-cache show docker-compose-plugin >/dev/null 2>&1; then
    echo "[setup] Installing docker-compose-plugin..."
    sudo apt-get install -y docker-compose-plugin
    return 0
  fi

  # Legacy v1 standalone.
  if apt-cache show docker-compose >/dev/null 2>&1; then
    echo "[setup] Installing docker-compose (v1 standalone)..."
    sudo apt-get install -y docker-compose
    return 0
  fi

  echo "[setup] ERROR: no compose package found in apt."
  echo "         Try: sudo apt-get install -y docker-compose-v2"
  exit 1
}

if ! detect_compose; then
  install_compose
  detect_compose || {
    echo "[setup] ERROR: compose installed but still not on PATH."
    echo "         Open a new shell and re-run this script."
    exit 1
  }
fi

echo "[setup] Using: ${COMPOSE_CMD[*]} ($(${COMPOSE_CMD[@]} version 2>/dev/null | head -1))"

if ! id -nG "$USER" | grep -qw docker; then
  echo "[setup] Adding $USER to docker group (re-login or 'newgrp docker' afterward)..."
  sudo usermod -aG docker "$USER"
  ADDED_GROUP=1
else
  ADDED_GROUP=0
fi

# Use sudo when the current shell lacks docker-group membership.
compose() {
  if docker ps >/dev/null 2>&1; then
    "${COMPOSE_CMD[@]}" "$@"
  else
    sudo "${COMPOSE_CMD[@]}" "$@"
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
    echo "[setup] WARN: HTTP not ready yet — check: ${COMPOSE_CMD[*]} logs neo4j"
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
