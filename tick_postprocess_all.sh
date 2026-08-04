#!/bin/bash
# Insert observations into every task of the 30-per-task corpus and re-validate.
#
# --revalidate replays each rewritten plan with the concurrent validator and
# records the REAL verdict, instead of the placeholder that has always claimed
# the record "must be revalidated" while nothing ever did. Tick records take the
# grid-dilating injector, so the coordination schedule survives.
#
# Symbolic only -- no rendering here, so this is minutes, not GPU-hours.

set -u
IN=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30
OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30_image
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
export PYTHONWARNINGS=ignore

ok=0; fail=0
for d in "$IN"/*/; do
  task=$(basename "$d")
  [ -f "$d/summary.json" ] || continue
  if "$PY" -m data_generation.task_level.generation.image.cli \
       --dataset "$d/summary.json" \
       --output-dataset "$OUT/$task/summary.json" \
       --revalidate --disable-progress --workers 8 >/dev/null 2>&1; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "FAILED: $task"
  fi
done
echo "post-processed ok=$ok failed=$fail"
