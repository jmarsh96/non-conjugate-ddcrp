#!/bin/bash
# ============================================================================
# Brazil dengue real-data application — full RJMCMC pipeline.
#
# NOT an array job: run_dengue.jl already parallelises internally with
# Distributed/pmap. `addprocs(SlurmManager())` turns every task in this
# allocation into a Julia worker, so --ntasks IS the worker count.
#
# --ntasks=18 matches the widest stage (Part 1 sweeps 18 configs), so each of
# Parts 1-4 completes in a single parallel round.
#
# Memory: the heavy stage is Part 3, where each worker holds a 250k x 399 chain
# for c, rho and r — about 2.4 GB, plus a thinned copy. ~6 GB per worker, hence
# 120G across the 18 tasks on one node.
#
# Timing: measured at ~44-64 s per 1k iterations at n=399, giving roughly 5-8 h
# for all four parts. 24 h is requested for headroom.
# ============================================================================

#SBATCH --job-name=ddcrp_dengue
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ddcrp_dengue_%j.out
#SBATCH --error=logs/ddcrp_dengue_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source src/slurm/env.sh

mkdir -p logs

# The dataset is committed to data/, so no internet access is needed here.
# Re-fetching it (prepare_dengue_data.jl) must be done on a login node.
if [[ ! -f data/dengue_uf41_2024.csv ]]; then
  echo "ERROR: data/dengue_uf41_2024.csv is missing." >&2
  echo "  Run this on a login node first (it needs internet access):" >&2
  echo "    julia --project=. src/prepare_dengue_data.jl --uf 41 --year 2024" >&2
  exit 1
fi

julia --project=. src/run_dengue.jl --slurm

echo
echo "Dengue chains complete. Analysis runs as a separate job (03_realdata_analysis.slurm)."
