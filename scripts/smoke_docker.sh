#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
UI_URL="${UI_URL:-http://localhost:3000}"
API_AUTH_TOKEN="${API_AUTH_TOKEN:-change-me-dev-token}"
POLL_ATTEMPTS="${POLL_ATTEMPTS:-60}"
POLL_SECONDS="${POLL_SECONDS:-2}"
START_STACK="${START_STACK:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HANDBOOK_PATH="$ROOT_DIR/samples/knowledge/company-handbook.md"
RELEASE_NOTES_PATH="$ROOT_DIR/samples/knowledge/release-notes.md"
DATASET_NAME="portfolio_eval.jsonl"

log() {
  printf '[smoke] %s\n' "$*" >&2
}

fail() {
  printf '[smoke] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

json_get() {
  local path="$1"
  local input_file
  input_file="$(mktemp)"
  cat > "$input_file"
  python3 - "$path" "$input_file" <<'PY'
import json
import sys

path = sys.argv[1].split(".")
with open(sys.argv[2], encoding="utf-8") as handle:
    value = json.load(handle)
for part in path:
    if isinstance(value, list):
        value = value[int(part)]
    else:
        value = value[part]
print(value)
PY
  rm -f "$input_file"
}

assert_json() {
  local description="$1"
  local script="$2"
  local input_file
  input_file="$(mktemp)"
  cat > "$input_file"
  python3 - "$description" "$script" "$input_file" <<'PY'
import json
import sys

description, script, input_file = sys.argv[1], sys.argv[2], sys.argv[3]
with open(input_file, encoding="utf-8") as handle:
    payload = json.load(handle)
try:
    ok = eval(script, {"__builtins__": {}}, {"payload": payload, "any": any, "len": len, "str": str})
except Exception as exc:
    raise SystemExit(f"{description}: assertion errored: {exc}\npayload={payload}") from exc
if not ok:
    raise SystemExit(f"{description}: assertion failed\npayload={payload}")
PY
  rm -f "$input_file"
}

curl_json() {
  curl --fail --silent --show-error -H "x-api-key: ${API_AUTH_TOKEN}" "$@"
}

wait_for_json_endpoint() {
  local url="$1"
  local description="$2"
  local body

  for attempt in $(seq 1 "$POLL_ATTEMPTS"); do
    if body="$(curl_json "$url" 2>/dev/null)"; then
      printf '%s' "$body"
      return 0
    fi
    log "waiting for $description attempt=$attempt/$POLL_ATTEMPTS"
    sleep "$POLL_SECONDS"
  done

  fail "$description did not become available within $((POLL_ATTEMPTS * POLL_SECONDS)) seconds"
}

poll_job() {
  local job_id="$1"
  local label="$2"
  local body status progress

  for attempt in $(seq 1 "$POLL_ATTEMPTS"); do
    body="$(curl_json "$API_URL/api/v1/jobs/$job_id")"
    status="$(printf '%s' "$body" | json_get status)"
    progress="$(printf '%s' "$body" | json_get progress)"
    log "$label status=$status progress=$progress attempt=$attempt/$POLL_ATTEMPTS"
    if [[ "$status" == "completed" ]]; then
      printf '%s' "$body"
      return 0
    fi
    if [[ "$status" == "failed" ]]; then
      printf '%s\n' "$body" >&2
      fail "$label failed"
    fi
    sleep "$POLL_SECONDS"
  done

  fail "$label did not complete within $((POLL_ATTEMPTS * POLL_SECONDS)) seconds"
}

require_command curl
require_command python3

if [[ "$START_STACK" == "1" ]]; then
  require_command docker
  log "starting Docker Compose stack"
  (cd "$ROOT_DIR" && docker compose up -d --build)
fi

log "checking API liveness"
wait_for_json_endpoint "$API_URL/health/live" "API liveness" | assert_json "live health" 'payload["status"] == "ok"'

log "checking API readiness"
wait_for_json_endpoint "$API_URL/health/ready" "API readiness" | assert_json "ready health" 'payload["status"] == "ok" and payload["checks"]["database"] == "ok" and payload["checks"]["qdrant"] == "ok"'

log "checking UI availability"
curl --fail --silent --show-error "$UI_URL/" >/dev/null

log "uploading sample knowledge base"
upload_response="$(
  curl_json \
    -F "files=@${HANDBOOK_PATH};type=text/markdown" \
    -F "files=@${RELEASE_NOTES_PATH};type=text/markdown" \
    "$API_URL/api/v1/documents/files"
)"
ingest_job_id="$(printf '%s' "$upload_response" | json_get id)"
[[ -n "$ingest_job_id" ]] || fail "upload did not return a job id"

ingest_result="$(poll_job "$ingest_job_id" "ingestion job")"
printf '%s' "$ingest_result" | assert_json "ingestion completed documents" 'payload["result"]["total"] >= 2 and payload["progress"] == 100'

log "checking document inventory"
curl_json "$API_URL/api/v1/documents" | assert_json \
  "document inventory" \
  'payload["facets"]["document_count"] >= 2 and any(doc["document_id"] == "company-handbook" for doc in payload["documents"]) and any(doc["document_id"] == "release-notes-2026-q1" or doc["document_id"] == "release-notes" for doc in payload["documents"])'

log "checking grounded filtered query"
query_response="$(
  curl_json \
    -H "Content-Type: application/json" \
    -d '{"question":"What is the refund window for annual plans?","category":"policy","include_trace":true}' \
    "$API_URL/api/v1/query"
)"
printf '%s' "$query_response" | assert_json \
  "grounded query" \
  'payload["grounded"] is True and len(payload["citations"]) >= 1 and len(payload["used_citation_ids"]) >= 1 and "30 calendar days" in payload["answer"]'

log "queueing offline evaluation"
eval_response="$(
  curl_json \
    -H "Content-Type: application/json" \
    -d "{\"dataset_name\":\"${DATASET_NAME}\"}" \
    "$API_URL/api/v1/evals/runs"
)"
eval_job_id="$(printf '%s' "$eval_response" | json_get id)"
[[ -n "$eval_job_id" ]] || fail "evaluation did not return a job id"

eval_result="$(poll_job "$eval_job_id" "evaluation job")"
printf '%s' "$eval_result" | assert_json \
  "evaluation completed" \
  'payload["result"]["summary"]["examples"] >= 3 and "mrr" in payload["result"]["summary"] and "faithfulness_score" in payload["result"]["summary"]'

log "Docker E2E smoke passed"
