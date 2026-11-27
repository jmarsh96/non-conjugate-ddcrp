#!/bin/bash
# ============================================================================
# Auxiliary Gibbs vs RJ comparison for one study. Submit through
# submit_gibbs_comparison.sh, which sets STUDY, --ntasks, --mem and --time.
#
# NOT an array job: run_gibbs_comparison.jl parallelises with Distributed/pmap and
# `addprocs(SlurmManager())` makes every task a worker, so --ntasks IS the worker
# count. The first run of a study measures pilot_timing.csv on these nodes, which
# sets every chain's length; keep it if you rerun on the same hardware.
# ============================================================================

#SBATCH --job-name=ddcrp_gibbs_cmp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ddcrp_gibbs_cmp_%x_%j.out
#SBATCH --error=logs/ddcrp_gibbs_cmp_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail

if [[ -z "${STUDY:-}" ]]; then
  echo "STUDY is not set. Submit with: bash src/slurm/submit_gibbs_comparison.sh" >&2
  exit 1
fi

module purge
module load bear-apps/2024a
module load Julia/1.12.2-linux-x86_64

# srun-launched workers need the module environment (BlueBEAR defaults to NONE).
export SLURM_EXPORT_ENV=ALL
export JULIA_WORKER_TIMEOUT=300

mkdir -p logs

julia --project=. src/run_gibbs_comparison.jl --study "${STUDY}" --slurm ${CMP_ARGS:-}
