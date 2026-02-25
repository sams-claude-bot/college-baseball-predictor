#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-college-baseball-dashboard.service}"
PORT="${PORT:-5000}"
HOST="${HOST:-127.0.0.1}"
CURL_BIN="${CURL_BIN:-curl}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

log() {
  printf '[restart-dashboard] %s\n' "$*"
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY RUN: $*"
    return 0
  fi
  "$@"
}

run_systemctl() {
  if command -v systemctl >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1 && [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
      run_cmd sudo systemctl "$@"
    else
      run_cmd systemctl "$@"
    fi
    return 0
  fi
  log "systemctl not found"
  return 1
}

kill_stale_gunicorn_on_port() {
  if ! command -v lsof >/dev/null 2>&1; then
    log "lsof not found; skipping stale listener scan for port ${PORT}"
    return 0
  fi

  local pids
  pids="$(lsof -nP -iTCP:${PORT} -sTCP:LISTEN -t 2>/dev/null | sort -u || true)"
  if [[ -z "$pids" ]]; then
    log "No listeners on port ${PORT}"
    return 0
  fi

  local pid args
  for pid in $pids; do
    args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$args" == *gunicorn* ]]; then
      log "Found stale gunicorn listener on port ${PORT}: pid=${pid}"
      run_cmd kill -TERM "$pid"
      if [[ "$DRY_RUN" -eq 0 ]]; then
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
          log "Escalating to SIGKILL for pid=${pid}"
          kill -KILL "$pid"
        fi
      fi
    else
      log "Port ${PORT} listener pid=${pid} is not gunicorn; leaving it alone (${args:-unknown})"
    fi
  done
}

check_http() {
  local path="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY RUN: verify http://${HOST}:${PORT}${path}"
    return 0
  fi
  "${CURL_BIN}" -fsS --max-time 5 "http://${HOST}:${PORT}${path}" >/dev/null
}

verify_dashboard() {
  if check_http "/health"; then
    log "Health check passed at /health"
    return 0
  fi

  log "/health failed; trying /"
  if check_http "/"; then
    log "Fallback health check passed at /"
    return 0
  fi

  log "Dashboard health verification failed on /health and /"
  return 1
}

main() {
  log "Stopping ${SERVICE_NAME}"
  run_systemctl stop "${SERVICE_NAME}"

  log "Checking for stale gunicorn listeners on port ${PORT}"
  kill_stale_gunicorn_on_port

  log "Starting ${SERVICE_NAME}"
  run_systemctl start "${SERVICE_NAME}"

  log "Verifying dashboard on http://${HOST}:${PORT}"
  verify_dashboard
  log "Restart verification complete"
}

main "$@"
