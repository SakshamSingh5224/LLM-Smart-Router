#!/usr/bin/env bash
# Chaos test (plan section 4): kill the router pod/container mid-load-test and
# confirm the gateway keeps serving via its rules-based fallback instead of
# erroring out to the user.
#
# Run from the repo root, with the compose stack already up:
#   docker compose -f docker/docker-compose.yml up -d
#   bash scripts/chaos_test.sh
set -euo pipefail
COMPOSE="docker compose -f docker/docker-compose.yml"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"

echo "==> Baseline: confirm the classifier router is healthy"
curl -sf "$GATEWAY_URL/health" | python3 -m json.tool

echo -e "\n==> Baseline request (should route via the trained classifier)"
curl -s -X POST "$GATEWAY_URL/api/chat" -H "Content-Type: application/json" \
  -d '{"query": "hi", "use_cache": false, "explain": true}' | python3 -m json.tool

echo -e "\n==> Killing router_service NOW..."
$COMPOSE kill router_service

echo -e "\n==> Request immediately after the kill (must still succeed, via fallback)"
resp=$(curl -s -w '\n%{http_code}' -X POST "$GATEWAY_URL/api/chat" -H "Content-Type: application/json" \
  -d '{"query": "design a distributed system and analyze the trade-offs", "use_cache": false}')
body=$(echo "$resp" | head -n -1)
code=$(echo "$resp" | tail -n1)
echo "$body" | python3 -m json.tool
if [ "$code" = "200" ]; then
  echo "PASS: gateway kept serving (HTTP 200) after the router died"
else
  echo "FAIL: gateway returned HTTP $code instead of degrading gracefully"
fi

echo -e "\n==> Restarting router_service..."
$COMPOSE up -d router_service
sleep 3
echo -e "\n==> Post-recovery health"
curl -sf "$GATEWAY_URL/health" | python3 -m json.tool
