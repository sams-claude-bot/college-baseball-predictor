#!/usr/bin/env bash
set -euo pipefail

# Build a template-driven Bootstrap CSS subset for faster first paint.
# Output: web/static/css/bootstrap-subset.min.css

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP_CSS="/tmp/bootstrap-full.min.css"
OUT_DIR="$ROOT/web/static/css"
OUT_FILE="$OUT_DIR/bootstrap-subset.min.css"

mkdir -p "$OUT_DIR"

curl -sSL https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css -o "$TMP_CSS"

# Safelist dynamic/bootstrap JS classes so collapse/modal/dropdown/tabs keep working.
SAFE_CLASSES=(
  show fade collapse collapsing modal modal-open
  dropdown-menu dropdown-item dropdown-toggle dropdown-menu-end
  navbar-toggler navbar-collapse navbar-toggler-icon
  btn-close active disabled focus
  input-group-text table-responsive table table-sm table-dark table-striped table-hover table-light
  alert alert-info alert-warning alert-success alert-danger tooltip popover
  tab-pane nav nav-tabs nav-link progress progress-bar badge
  card card-header card-body card-footer
  form-control form-select form-check form-check-input form-check-label
  btn btn-sm btn-lg btn-outline-secondary btn-outline-primary btn-secondary btn-primary btn-danger btn-success btn-warning btn-info btn-dark
  row col col-auto col-md-6 col-md-4 col-lg-3 col-lg-4 col-lg-6 col-lg-8 col-12 col-6 col-5 col-4 col-3 col-2 col-1
  d-flex d-none d-sm-inline d-sm-none d-lg-none d-lg-block
  justify-content-between justify-content-center
  align-items-center align-items-start align-items-end
  text-center text-start text-end text-muted text-success text-danger text-warning text-info text-dark
  bg-dark bg-light bg-white bg-info bg-success bg-danger bg-warning
)

npx --yes purgecss \
  --css "$TMP_CSS" \
  --content "$ROOT"/web/templates/*.html \
  --output "$OUT_DIR" \
  --safelist "${SAFE_CLASSES[@]}"

# PurgeCSS outputs using the same source filename.
mv "$OUT_DIR/bootstrap-full.min.css" "$OUT_FILE"

echo "Built $OUT_FILE"
ls -lh "$OUT_FILE"
