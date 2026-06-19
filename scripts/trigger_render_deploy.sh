#!/usr/bin/env bash
# Trigger a Render deploy for BTCPred (requires RENDER_API_KEY).
set -euo pipefail

SERVICE_ID="${RENDER_SERVICE_ID:-srv-d7oijttckfvc73f6u0eg}"
API_KEY="${RENDER_API_KEY:?Set RENDER_API_KEY from https://dashboard.render.com/u/settings/api-keys}"

curl -fsS -X POST "https://api.render.com/v1/services/${SERVICE_ID}/deploys" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"clearCache":"do_not_clear"}'

echo ""
echo "Deploy triggered for ${SERVICE_ID}"
