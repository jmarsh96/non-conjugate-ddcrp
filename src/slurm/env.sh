#!/bin/bash
# ============================================================================
# env.sh — shared module/environment setup for the Julia MCMC jobs.
#
# Sourced by every .slurm script in this directory so the module stack is
# defined in one place.
#
# JULIA MODULE: BlueBear's Julia module name changes between bear-apps
# releases, and this project targets Julia 1.12. Check what is available with
#
#     module load bear-apps/2024a && module avail Julia
#
# then either export JULIA_MODULE before submitting, e.g.
#
#     export JULIA_MODULE=Julia/1.11.3-linux-x86_64
#     bash src/slurm/submit_realdata.sh
#
# or leave JULIA_MODULE unset and put a juliaup//rds install of julia on PATH
# in your ~/.bashrc. The scripts fail loudly if julia cannot be found, rather
# than queueing a job that dies twenty minutes in.
# ============================================================================

set -euo pipefail

module purge
module load bluebear
module load bear-apps/2024a

# R is needed only by the mapping step, but loading it everywhere is harmless
# and keeps the two job types identical in setup.
module load R/4.5.0-gfbf-2024a

if [[ -n "${JULIA_MODULE:-}" ]]; then
  module load "${JULIA_MODULE}"
fi

if ! command -v julia >/dev/null 2>&1; then
  echo "ERROR: julia not found on PATH." >&2
  echo "  Set JULIA_MODULE (see the comment at the top of src/slurm/env.sh)," >&2
  echo "  or add a local Julia install to PATH in ~/.bashrc." >&2
  exit 1
fi

# One BLAS thread per worker. Each Slurm task is a separate Julia process, so
# letting BLAS spawn its own threads on top would oversubscribe the node badly.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1

echo "host          : $(hostname)"
echo "julia         : $(julia --version)  ($(command -v julia))"
echo "SLURM_JOB_ID  : ${SLURM_JOB_ID:-<none>}"
echo "SLURM_NTASKS  : ${SLURM_NTASKS:-<none>}  (= number of Julia workers)"
echo

# Fail fast if the environment was never instantiated. Compute nodes have no
# internet, so Pkg cannot fix this here — better to lose 20 seconds now than a
# 24-hour queue slot. Skipped with SKIP_ENV_CHECK=1.
if [[ "${SKIP_ENV_CHECK:-0}" != "1" ]]; then
  if ! julia --project=. -e 'using DistanceDependentCRP' >/dev/null 2>&1; then
    echo "ERROR: the Julia environment is not usable on this node." >&2
    echo "  Run this on a LOGIN node (it needs internet access):" >&2
    echo "    julia --project=. src/setup_environment.jl" >&2
    echo "  If the cluster's Julia differs from the 1.12 used locally, use:" >&2
    echo "    julia --project=. src/setup_environment.jl --fresh" >&2
    exit 1
  fi
  echo "environment   : OK (DistanceDependentCRP loads)"
  echo
fi
