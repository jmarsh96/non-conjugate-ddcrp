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

#SBATCH --job-name=ddcrp_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ddcrp_setup_%j.out
#SBATCH --error=logs/ddcrp_setup_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk


set -euo pipefail

module purge
module load bear-apps/2024a
module load bluebear
module load Julia/1.12.2-linux-x86_64


mkdir -p logs

julia --project=. src/setup_environment.jl --fresh

module purge
module load bear-apps/2024a
module load bluebear
module load R/4.5.0-gfbf-2024a

Rscript -e 'install.packages(c("sf","ggplot2","dplyr"), repos="https://cloud.r-project.org")'

echo
echo "Setup complete."
