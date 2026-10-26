#!/usr/bin/env bash
# (Re)build and start the engine container on a Docker network (IF_NETWORK, default "engines") that a Cloudflare
# Tunnel also joins (rule: path ^/indiafacts/ -> http://indiafacts-engine:8080). Postgres is the host's, reached
# through its unix socket. Secrets never enter the image: IF_SECRETS holds indiafacts-engine.key and, unless
# IF_ANTHROPIC_KEY names another file, indiafacts-anthropic.key; IF_DB_PASSWORD_FILE is the database password
# (IF_DB_USER, IF_DB_NAME name the role and DB).
set -euo pipefail
cd "$(dirname "$0")/.."
S=${IF_SECRETS:?set IF_SECRETS to the folder holding the key files}
DBPW=${IF_DB_PASSWORD_FILE:?set IF_DB_PASSWORD_FILE}
AKEY=${IF_ANTHROPIC_KEY:-$S/indiafacts-anthropic.key}
NET=${IF_NETWORK:-engines}
# Meaning search: the corpus vectors (data/vectors.npz) and the question model, downloaded once (python -m engine search "x").
# Peak memory with both loaded ~760 MB (measured 2026-10-17), hence --memory 900m.
MODEL=${IF_MODEL_DIR:-$HOME/.cache/indiafacts/embeddinggemma}
for f in "$AKEY" "$S/indiafacts-engine.key" "$DBPW" data/vectors.npz "$MODEL/onnx/model_q4.onnx_data"; do
  [ -s "$f" ] || { echo "error: no $f" >&2; exit 1; }
done
VERSION=$(git rev-parse --short HEAD)$(git status --porcelain -- engine | grep -q . && echo -dirty || true)
docker build -q -f deploy/Dockerfile --build-arg IF_VERSION="$VERSION" -t indiafacts-engine . >/dev/null
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
docker rm -f indiafacts-engine >/dev/null 2>&1 || true
IF_ENGINE_KEY=$(tr -d '[:space:]' < "$S/indiafacts-engine.key")
export IF_ENGINE_KEY
mkdir -p runs
docker run -d --name indiafacts-engine --network "$NET" --restart unless-stopped \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges --memory 900m --pids-limit 64 \
  -p 127.0.0.1:8091:8080 \
  -e IF_ENGINE_KEY -e IF_DAILY_USD="${IF_DAILY_USD:-3}" \
  -e IF_ANTHROPIC_KEY_FILE=/secrets/anthropic.key -v "$AKEY:/secrets/anthropic.key:ro" \
  -e IF_DB_HOST=/var/run/postgresql -e IF_DB_USER="${IF_DB_USER:-indiafacts}" -e IF_DB_NAME="${IF_DB_NAME:-${IF_DB_USER:-indiafacts}}" \
  -e IF_DB_PASSWORD_FILE=/secrets/db.password -v "$DBPW:/secrets/db.password:ro" -v /var/run/postgresql:/var/run/postgresql:ro \
  -v "$PWD/runs:/app/runs" -v "$PWD/data/vectors.npz:/app/data/vectors.npz:ro" -v "$PWD/data/vectors.json:/app/data/vectors.json:ro" \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/tmp/hf -e IF_MODEL_DIR=/models -v "$MODEL:/models" \
  indiafacts-engine >/dev/null
unset IF_ENGINE_KEY
docker ps --filter name=indiafacts-engine --format '{{.Names}}: {{.Status}}'
