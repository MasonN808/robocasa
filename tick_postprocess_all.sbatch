#!/bin/bash
#SBATCH --job-name=tick-post30
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-post30-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-post30-%j.err

# Re-run of tick_postprocess_all.sh over the WHOLE 1588-trajectory corpus.
#
# The first pass was launched before the top-up jobs landed, so 38 trajectories
# across 14 tasks were never post-processed and never rendered. Since the render
# is being redone anyway (for --step-order concurrent), the cheap fix is to
# reprocess everything rather than merge two vintages.
#
# The previous output is backed up at tick_scale30_image.pre_topup.

set -u
bash /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration/tick_postprocess_all.sh

IN=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30
OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30_image
echo "raw:            $(ls $IN/*/trajectories/traj_*.json | wc -l)"
echo "post-processed: $(ls $OUT/*/trajectories/traj_*.json | wc -l)"
echo POST DONE
