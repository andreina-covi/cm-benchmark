#!/usr/bin/env bash
# Regenerate the reference scenes and diff them against a saved baseline.
# Usage: scripts/check_baseline.sh [BASELINE_DIR] [generate_items args...]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${1:-$REPO/analysis/baseline}"
shift || true
CHECK="${CHECK_DIR:-$REPO/analysis/check}"
PY="${PYTHON:-python}"

rm -rf "$CHECK"
PYTHON="$PY" "$REPO/scripts/run_baseline.sh" "$CHECK" "$@" >/dev/null
"$PY" "$REPO/scripts/compare_outputs.py" "$BASE" "$CHECK"
