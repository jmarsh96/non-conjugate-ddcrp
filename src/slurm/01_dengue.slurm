#!/bin/bash
# ============================================================================
# Brazil dengue real-data application: the analysis reported in the paper.
#
# NOT an array job: run_dengue.jl already parallelises internally with
# Distributed/pmap. `addprocs(SlurmManager())` turns every task in this
# allocation into a Julia worker, so --ntasks IS the worker count.
#
# --ntasks=18 covers the widest stage (Part 1 sweeps 16 configs; Part 2 tunes
# at most 14), so each part completes in a single parallel round. Part 3 uses
# only 5 of them (4 chains of the selected sampler plus the Poisson comparator).
#
# Memory: Part 3 is the heavy stage. Each of its 5 workers holds a raw
# 300k x 399 chain for c, rho and r, about 2.9 GB, plus the thinned copy it
# saves; the 60k-iteration sweep chains are about 0.6 GB each. The two stages
# peak at a similar place, roughly 20 GB including the runtime on each worker:
# Part 1 with 16 workers holding small chains, Part 3 with 5 holding large ones.
# 72G covers that with room for the GC to lag behind pmap handing out new tasks.
#
# Timing: measured at ~13 s per 1k iterations at n=399 under the hop-window
# prior, so Parts 1 and 2 take roughly 15 minutes each and Part 3 about 1 h.
# Links are proposed from the ddCRP prior, which both mixes far better and
# costs less per iteration than the uniform proposal it replaced. 24 h is
# requested for headroom.
# ============================================================================

#SBATCH --job-name=ddcrp_dengue
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1
#SBATCH --mem=72G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ddcrp_dengue_%j.out
#SBATCH --error=logs/ddcrp_dengue_%j.err
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
# launch_timeout is a Julia keyword and is set in the run script.
export JULIA_WORKER_TIMEOUT=300

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
