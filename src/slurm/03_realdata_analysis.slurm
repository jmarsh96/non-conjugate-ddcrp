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

mkdir -p logs

# The Julia and R module trees must NOT be loaded at the same time: the
# gfbf/2024a toolchain that R pulls in puts its own GLib on LD_LIBRARY_PATH,
# which shadows the GLib that Julia's Glib_jll artifact (used by the GR/Plots
# backend) expects. That mismatch makes Julia die at startup with
# "libgobject-2.0.so: undefined symbol: g_string_copy". So each language runs
# in its own subshell with only its own modules loaded.

run_julia() {
  ( module purge
    module load bear-apps/2024a
    module load bluebear
    module load Julia/1.12.2-linux-x86_64
    julia --project=. "$@" )
}

run_r() {
  ( module purge
    module load bear-apps/2024a
    module load bluebear
    module load R/4.5.0-gfbf-2024a
    Rscript "$@" )
}

status=0
ran=0

if [[ -f results/real_data/dengue/run_metadata.jld2 ]]; then
  echo "=== dengue ==="
  ran=1
  run_julia src/analyse_dengue.jl || status=1
  run_r src/map_dengue.R || status=1
else
  echo "SKIP dengue — results/real_data/dengue/run_metadata.jld2 not found."
fi

echo
if [[ -f results/real_data/aqs/run_metadata.jld2 ]]; then
  echo "=== AQS ==="
  ran=1
  run_julia src/analyse_aqs.jl || status=1
  run_r src/map_aqs.R || status=1
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
