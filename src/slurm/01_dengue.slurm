#!/bin/bash
# ============================================================================
# Brazil dengue real-data application: the analysis reported in the paper.
#
# NOT an array job: run_dengue.jl already parallelises internally with
# Distributed/pmap. `addprocs(SlurmManager())` turns every task in this
# allocation into a Julia worker, so --ntasks IS the worker count.
#
# --ntasks=18 covers the widest stage (Part 1 sweeps 16 configs; Part 2 tunes
# at most 14), so each part completes in a single parallel round.
#
# Memory: the heavy stage is Part 3, where each of 3 workers holds a 250k x 399
# chain for c, rho and r (about 2.4 GB) plus a thinned copy; the 60k-iteration
# sweep chains are about 0.6 GB each. 120G on one node leaves ample headroom.
#
# Timing: the hop-window prior (radius 2) measured ~34 s per 1k iterations at
# n=399, giving roughly 1 h each for Parts 1 and 2 and ~3 h for Part 3.
# 24 h is requested for headroom.
# ============================================================================

#SBATCH --job-name=ddcrp_dengue
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ddcrp_dengue_%j.out
#SBATCH --error=logs/ddcrp_dengue_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk


set -euo pipefail

module purge
module load bear-apps/2024a
module load bluebear
module load Julia/1.12.2-linux-x86_64

mkdir -p logs

for f in data/cleaned/dengue_uf41_2024.csv data/cleaned/dengue_uf41_adjacency.csv; do
  if [[ ! -f "$f" ]]; then
    echo "Missing $f: run the prepare_dengue_* scripts on a login node first." >&2
    exit 1
  fi
done

julia --project=. src/run_dengue.jl --slurm

echo
echo "Dengue chains complete. Analysis runs as a separate job (03_realdata_analysis.slurm)."
