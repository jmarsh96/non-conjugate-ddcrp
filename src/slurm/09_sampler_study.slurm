#!/bin/bash
# ============================================================================
# Sampler study confirm stage (run_sampler_study.jl --stage confirm): the finalist
# moves on replicates 1-3 of the simulated skew-normal sets, 600 s chains, then
# its analysis. Submit through submit_gibbs_comparison.sh (CONFIGS is set there).
# Every task is a worker (SlurmClusterManager); the first run measures
# results/sampler_study/confirm/pilot_timing.csv on these nodes.
# ============================================================================

#SBATCH --job-name=ddcrp_sampler_study
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ddcrp_sampler_study_%j.out
#SBATCH --error=logs/ddcrp_sampler_study_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail

module purge
module load bear-apps/2024a
module load Julia/1.12.2-linux-x86_64

export SLURM_EXPORT_ENV=ALL
export JULIA_WORKER_TIMEOUT=300
export GKSwstype=nul

mkdir -p logs

julia --project=. src/run_sampler_study.jl --stage confirm --configs "${CONFIGS}" --slurm ${CMP_ARGS:-}
julia --project=. src/analyse_sampler_study.jl --stage confirm ${CMP_ARGS:-}
