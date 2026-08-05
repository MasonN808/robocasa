#!/bin/bash
#SBATCH --job-name=render-order-smoke
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/render-order-smoke-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/render-order-smoke-%j.err

# One task rendered BOTH ways, so --step-order can be checked against something
# rather than merely observed not to crash. Same trajectories, same scene, same
# seed: every pixel difference is the within-tick ordering and nothing else.
#
# GL has to be egl in both the parent (exported, imported at module scope) and
# the workers (--gl-backend). See tick_render30.sbatch for why.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

export PYTHONWARNINGS=ignore
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
IN=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30_image
OUT=/work/umass/shlomo_umass/dbenhamougol_umass/render_order_smoke
TASK=${TASK:-seasoning_steak}

rm -rf "$OUT"
mkdir -p "$OUT"

for MODE in concurrent file; do
  echo "=== $MODE ==="
  "$PY" scripts/sweep_trajectories.py \
    --input-dir "$IN" \
    --output-dir "$OUT/$MODE" \
    --tasks "$TASK" \
    --indices 0 1 2 \
    --step-order "$MODE" \
    --gpu-ids 0 --gl-backend egl || echo "exited $?"
done

echo "=== reorder bookkeeping ==="
"$PY" - "$OUT/concurrent" <<'EOF'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).rglob("render_order.json")):
    d = json.loads(p.read_text())
    print(f"{p.parent.name:>12}  reordered={d['reordered_steps']:>3}  "
          f"fallback={d['fallback_reason']}")
EOF

echo "=== images that differ between the two orders ==="
"$PY" - "$OUT" <<'EOF'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
def digest(mode):
    out = {}
    for p in sorted((root / mode).rglob("*.jpg")):
        out[str(p.relative_to(root / mode))] = hashlib.md5(p.read_bytes()).hexdigest()
    return out
a, b = digest("concurrent"), digest("file")
shared = set(a) & set(b)
diff = [k for k in sorted(shared) if a[k] != b[k]]
print(f"concurrent {len(a)} images, file {len(b)} images, shared {len(shared)}")
print(f"differ: {len(diff)} ({len(diff)/max(len(shared),1):.1%})")
for k in diff[:15]:
    print("  ", k)
if set(a) ^ set(b):
    print("only in one order:", sorted(set(a) ^ set(b))[:10])
EOF

echo "SMOKE DONE"
