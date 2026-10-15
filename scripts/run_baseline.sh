#!/usr/bin/env bash
# Rebuild episode GT + items for the three reference scenes into one folder.
# Usage: scripts/run_baseline.sh OUT_DIR [extra generate_items args...]
set -euo pipefail

OUT="${1:?usage: run_baseline.sh OUT_DIR [generate_items args...]}"
shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
COLLECTION="${COLLECTION:-/home/andreina/Documents/Programs/Dataset/Generated/navigation/files}"
MODEL="${MODEL:-$REPO/analysis/dt_tune/visibility_filter.joblib}"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

"$PY" -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder "$COLLECTION" \
  --db_path "$OUT/episodes" \
  --export_json \
  --output_path "$OUT/nav_data" \
  --visibility_model_path "$MODEL" >"$OUT.nav.log" 2>&1

"$PY" -m cm_benchmark.generation.generate_items \
  --episode_json "$OUT/nav_data" \
  --output_path "$OUT/items" \
  --visibility_model_path "$MODEL" \
  --max_per_construct 5 \
  --swm_min_delay 5 --swm_max_delay 10 \
  --su_min_delay 5 --su_max_delay 10 \
  "$@"
