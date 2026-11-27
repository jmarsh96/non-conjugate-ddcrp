#!/bin/bash
# ============================================================================
# Skew-normal simulation study: tuning, the multi-seed proposal comparison, the
# prior-recovery (Geweke) check, then the analysis that writes the paper's
# skew-normal figures and tables into docs/ddcrp-paper/.
#
# NOT an array job: the run scripts parallelise internally with
# Distributed/pmap, and `addprocs(SlurmManager())` turns every task in this
# allocation into a Julia worker, so --ntasks IS the worker count.
#
# Stages (run_skewnormal.jl unless noted):
#   Part 1  birth-σ tuning: 3 rounds of at most 8 chains of 40k iterations.
#   Part 2  16 configs x 2 scenarios x 4 seeds = 128 chains of 110k iterations,
#           i.e. 4 rounds on 32 workers.
#   geweke_skewnormal.jl  16 prior-only chains of 100k iterations.
#   analyse_skewnormal.jl figures and tables (serial).
#
# Timing: ~1000 s per 110k-iteration chain, so Part 2 takes about 75 minutes,
# Part 1 about 20 and the Geweke check about 15; 12 h is requested for headroom.
#
# Memory: each worker holds a raw 110k x 150 chain for c, xi, omega and alpha,
# about 0.55 GB, plus its thinned copy and the Julia runtime, so ~1.5 GB per
# worker at peak; 96G covers 32 workers with room for the GC to lag.
# ============================================================================

#SBATCH --job-name=ddcrp_skewnormal
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ddcrp_skewnormal_%j.out
#SBATCH --error=logs/ddcrp_skewnormal_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk


set -euo pipefail

module purge
module load bear-apps/2024a
module load Julia/1.12.2-linux-x86_64

# BlueBEAR defaults SLURM_EXPORT_ENV=NONE, so tasks launched by srun start
# without the module environment. SlurmClusterManager launches every worker with
# srun, and the module's `julia` is a wrapper that finds `julia.bin` on PATH, so
# without this the workers die with "julia.bin: command not found".
export SLURM_EXPORT_ENV=ALL

# Timeout for a worker connecting back to the master. SlurmClusterManager's own
# launch_timeout is a Julia keyword and is set in the run scripts.
export JULIA_WORKER_TIMEOUT=300

# Headless GR for the analysis step.
export GKSwstype=nul

mkdir -p logs

julia --project=. src/run_skewnormal.jl --slurm
julia --project=. src/geweke_skewnormal.jl --slurm
julia --project=. src/analyse_skewnormal.jl

echo
echo "Skew-normal study complete: results in results/simulation_study/skewnormal/,"
echo "paper figures and tables copied into docs/ddcrp-paper/."
