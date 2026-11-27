#!/bin/bash
# ============================================================================
# Submit the two real-data applications and the analysis that depends on them.
#
# The two run jobs are independent and queue in parallel. The analysis waits
# for both, using `afterany` rather than `afterok` so that if one application
# fails the other is still written up — 03_realdata_analysis.slurm skips
# whichever results are missing.
#
# Usage (from the repo root):
#   bash src/slurm/submit_realdata.sh              # both applications
#   bash src/slurm/submit_realdata.sh dengue       # dengue only
#   bash src/slurm/submit_realdata.sh aqs          # AQS only
#
#   AQS_REGION=east bash src/slurm/submit_realdata.sh aqs
#   JULIA_MODULE=Julia/1.11.3-linux-x86_64 bash src/slurm/submit_realdata.sh
# ============================================================================

set -euo pipefail

which_app="${1:-both}"
mkdir -p logs

# JULIA_MODULE and AQS_REGION are read by the job scripts, so they must survive
# into the job environment; --export=ALL is the sbatch default but is stated
# explicitly here because the jobs are unusable without JULIA_MODULE.
export_opts="--export=ALL"

deps=()

if [[ "$which_app" == "both" || "$which_app" == "dengue" ]]; then
  dengue=$(sbatch --parsable $export_opts src/slurm/01_dengue.slurm)
  echo "Submitted dengue run   : job ${dengue}"
  deps+=("${dengue}")
fi

if [[ "$which_app" == "both" || "$which_app" == "aqs" ]]; then
  aqs=$(sbatch --parsable $export_opts src/slurm/02_aqs.slurm)
  echo "Submitted AQS run      : job ${aqs}  (region ${AQS_REGION:-conus})"
  deps+=("${aqs}")
fi

if [[ ${#deps[@]} -eq 0 ]]; then
  echo "Nothing submitted: expected 'dengue', 'aqs' or 'both', got '${which_app}'." >&2
  exit 1
fi

dep_spec=$(IFS=:; echo "${deps[*]}")
analysis=$(sbatch --parsable $export_opts --dependency=afterany:"${dep_spec}" \
                  src/slurm/03_realdata_analysis.slurm)
echo "Submitted analysis     : job ${analysis} (afterany:${dep_spec})"

echo
echo "Track with: squeue -u \$USER"
echo "Logs in:    logs/"
