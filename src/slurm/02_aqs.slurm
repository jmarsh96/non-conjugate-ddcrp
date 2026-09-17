#!/bin/bash
# ============================================================================
# EPA AQS PM2.5 real-data application — full RJMCMC pipeline.
#
# NOT an array job: run_aqs.jl parallelises internally with Distributed/pmap,
# and `addprocs(SlurmManager())` makes every task in this allocation a worker,
# so --ntasks IS the worker count.
#
# --ntasks=14 matches Part 1's 14 configs. Part 2 has 20 configs and so takes
# two rounds; that costs ~40 min and is the price of keeping the memory request
# schedulable (see below).
#
# Memory: at the default --region conus (n=697) a Part 3 worker holds a
# 250k x 697 chain for c, xi, omega and alpha — about 5.6 GB, plus a thinned
# copy. ~9 GB per worker, hence 130G across the 14 tasks. A smaller region
# (--region east/west/ca) needs proportionally less; override at submit time
# with `sbatch --mem=...`.
#
# Timing: measured at ~80 s per 1k iterations at n=697, giving roughly 8 h for
# all four parts. 24 h is requested for headroom.
# ============================================================================

#SBATCH --job-name=ddcrp_aqs
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=1
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ddcrp_aqs_%j.out
#SBATCH --error=logs/ddcrp_aqs_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk


set -euo pipefail

module purge
module load bear-apps/2024a
module load Julia/1.12.2-linux-x86_64

mkdir -p logs

REGION="${AQS_REGION:-conus}"

# Only the small derived per-site CSV is needed, and it is committed to data/cleaned/;
# the 305 MB raw daily file is not required here. Building a region that has
# not been prepared needs internet, so do it on a login node.
if [[ ! -f "data/cleaned/aqs_pm25_2023_${REGION}.csv" ]]; then
  echo "ERROR: data/cleaned/aqs_pm25_2023_${REGION}.csv is missing." >&2
  echo "  Run this on a login node first (it needs internet access):" >&2
  echo "    julia --project=. src/prepare_aqs_data.jl --year 2023 --region ${REGION}" >&2
  exit 1
fi

julia --project=. src/run_aqs.jl --slurm --region "${REGION}"

echo
echo "AQS chains complete. Analysis runs as a separate job (03_realdata_analysis.slurm)."
