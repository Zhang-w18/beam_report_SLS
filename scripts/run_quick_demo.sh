#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m beam_sls.run \
  --config configs/v2_one_site_three_sector.yaml \
  --out runs/v2_quick_demo \
  --num-drops 2 \
  --num-tti 5 \
  --skip-heatmap
