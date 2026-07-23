#!/bin/bash
# ============================================================================
# Figures, tables, model comparison and maps for both real-data applications.
#
# Single-process: the analyse scripts do not use Distributed. They do load
# several full chains at once, which is what the memory request is for.
#
# Each application is guarded so this job is still useful when only one of the
# two run jobs has finished (or when one of them failed) — it analyses what is
# there and says so, rather than aborting on the first missing directory.
# ============================================================================

#SBATCH --job-name=ddcrp_rd_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddcrp_rd_analysis_%j.out
#SBATCH --error=logs/ddcrp_rd_analysis_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk


set -euo pipefail

module purge
module load bear-apps/2024a
module load bluebear
module load Julia/1.12.2-linux-x86_64
module load R/4.5.0-gfbf-2024a

mkdir -p logs

status=0
ran=0

if [[ -f results/real_data/dengue/run_metadata.jld2 ]]; then
  echo "=== dengue ==="
  ran=1
  julia --project=. src/analyse_dengue.jl || status=1
  Rscript src/map_dengue.R || status=1
else
  echo "SKIP dengue — results/real_data/dengue/run_metadata.jld2 not found."
fi

echo
if [[ -f results/real_data/aqs/run_metadata.jld2 ]]; then
  echo "=== AQS ==="
  ran=1
  julia --project=. src/analyse_aqs.jl || status=1
  Rscript src/map_aqs.R || status=1
else
  echo "SKIP AQS — results/real_data/aqs/run_metadata.jld2 not found."
fi

echo
if [[ $ran -eq 0 ]]; then
  # Both run jobs failed or were never submitted. Exit non-zero so this shows
  # as FAILED in sacct rather than quietly reporting success having done nothing.
  echo "NOTHING TO ANALYSE — neither application produced run_metadata.jld2." >&2
  echo "Check the run job logs in logs/." >&2
  exit 1
elif [[ $status -ne 0 ]]; then
  echo "One or more analysis steps FAILED — see above." >&2
else
  echo "Analysis complete. Figures under results/real_data/*/figures/."
fi
exit $status
