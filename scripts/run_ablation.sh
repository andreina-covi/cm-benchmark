#!/usr/bin/env bash
# Generate items for every configs/ablations/*.yaml against existing nav JSON.
# Usage: scripts/run_ablation.sh [NAV_DIR] [OUT_DIR] [extra generate_items args...]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAV="${1:-$REPO/analysis/baseline/nav_data}"
shift || true
OUT="${1:-$REPO/analysis/ablations}"
shift || true
PY="${PYTHON:-python}"
MODEL="${MODEL:-$REPO/analysis/dt_tune/visibility_filter.joblib}"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

if [[ ! -d "$NAV" ]]; then
  echo "nav JSON root not found: $NAV" >&2
  exit 1
fi

shopt -s nullglob
cfgs=("$REPO"/configs/ablations/*.yaml)
if [[ ${#cfgs[@]} -eq 0 ]]; then
  echo "no configs/ablations/*.yaml" >&2
  exit 1
fi

mkdir -p "$OUT"
for cfg in "${cfgs[@]}"; do
  name="$(basename "$cfg" .yaml)"
  dest="$OUT/$name"
  mkdir -p "$dest"
  echo "=== $name ==="
  "$PY" -m cm_benchmark.generation.generate_items \
    --episode_json "$NAV" \
    --output_path "$dest/items" \
    --config "$cfg" \
    --visibility_model_path "$MODEL" \
    --max_per_construct 5 \
    --swm_min_delay 5 --swm_max_delay 10 \
    --su_min_delay 5 --su_max_delay 10 \
    "$@"
done

echo "Ablation items under $OUT"
echo "Report: $PY scripts/ablation_report.py $OUT"
