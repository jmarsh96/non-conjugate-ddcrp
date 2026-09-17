#!/bin/bash
# ============================================================================
# Submit the dengue prior × proposal grid as a Slurm array, with the collect
# and report job queued after it.
#
# Nothing here runs Julia: this runs on a login node. The grid size is the
# N_CONFIGS constant below, and every array task checks it against the grid in
# dengue_sensitivity.jl, failing at once if they disagree. After editing the
# grid, update N_CONFIGS (the --test grid has 16, the full grid 224).
#
# Usage:
#   bash src/slurm/submit_dengue_sensitivity.sh
#   MAX_PARALLEL=40 bash src/slurm/submit_dengue_sensitivity.sh     # throttle
#   TASKS=5,17,203 bash src/slurm/submit_dengue_sensitivity.sh      # rerun chosen tasks
#   SENS_ARGS="--force" TASKS=17 bash src/slurm/submit_dengue_sensitivity.sh
#   PANDOC_MODULE=Pandoc/3.1.2 bash src/slurm/submit_dengue_sensitivity.sh
# ============================================================================

set -euo pipefail

mkdir -p logs

N_CONFIGS="${N_CONFIGS:-224}"
export N_CONFIGS
TASKS="${TASKS:-1-${N_CONFIGS}}"
echo "Grid: ${N_CONFIGS} configurations"

array="${TASKS}"
if [[ -n "${MAX_PARALLEL:-}" ]]; then array="${array}%${MAX_PARALLEL}"; fi

run=$(sbatch --parsable --export=ALL --array="${array}" src/slurm/04_dengue_sensitivity_array.slurm)
echo "Submitted array   : job ${run}  (tasks ${array})"

collect=$(sbatch --parsable --export=ALL --dependency=afterany:"${run}" \
  src/slurm/05_dengue_sensitivity_collect.slurm)
echo "Submitted collect : job ${collect}  (after the array finishes)"

echo
echo "Monitor: squeue -u \$USER"
echo "Failed tasks: sacct -j ${run} --format=JobID,State,Elapsed,MaxRSS | grep -v COMPLETED"
