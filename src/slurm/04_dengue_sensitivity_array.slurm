#!/bin/bash

#SBATCH --job-name=ddcrp_dengue_sens
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ddcrp_dengue_sens_%A_%a.out
#SBATCH --error=logs/ddcrp_dengue_sens_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

# One configuration of the dengue prior × proposal grid per array task. Submit
# through submit_dengue_sensitivity.sh, which sets --array from the grid size
# and queues 05_dengue_sensitivity_collect.slurm after it.
#
# Memory: the sampler holds every iteration of c, rho and r (3 x 300k x 399 x
# 8 bytes, about 2.9 GB) plus the thinned copy and the reductions; 8G leaves
# headroom. A task whose result already exists exits at once, so resubmitting
# the whole array only reruns the tasks that failed.

set -euo pipefail

mkdir -p logs

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Not an array task. Submit with: bash src/slurm/submit_dengue_sensitivity.sh" >&2
  exit 1
fi

UF="${DENGUE_UF:-41}"
YEAR="${DENGUE_YEAR:-2024}"

for f in "data/cleaned/dengue_uf${UF}_${YEAR}.csv" "data/cleaned/dengue_uf${UF}_adjacency.csv"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING $f" >&2
    exit 1
  fi
done

module purge
module load bear-apps/2024a
module load Julia/1.12.2-linux-x86_64

julia --project=. src/dengue_sensitivity.jl --task "${SLURM_ARRAY_TASK_ID}" \
  --uf "${UF}" --year "${YEAR}" ${SENS_ARGS:-}
