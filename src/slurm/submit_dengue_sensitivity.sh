#!/bin/bash
# ============================================================================
# Submit the dengue prior × proposal grid as a Slurm array, with the collect
# and report job queued after it.
#
# The grid size is read from the Julia script (`--list`), so the array range
# follows any edit to the grid. Run from the repo root on a login node.
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

if [[ -z "${TASKS:-}" ]]; then
  n=$( ( module purge
         module load bear-apps/2024a
         module load bluebear
         module load "${JULIA_MODULE:-Julia/1.12.2-linux-x86_64}"
         julia --project=. src/dengue_sensitivity.jl --list ${SENS_ARGS:-} ) \
       | sed -n 's/^N_CONFIGS=//p')
  if [[ -z "$n" ]]; then
    echo "Could not read the grid size from dengue_sensitivity.jl --list" >&2
    exit 1
  fi
  TASKS="1-${n}"
  echo "Grid: ${n} configurations"
fi

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
