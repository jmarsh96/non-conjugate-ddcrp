#!/bin/bash

#SBATCH --job-name=ddcrp_dengue_sens_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddcrp_dengue_sens_collect_%j.out
#SBATCH --error=logs/ddcrp_dengue_sens_collect_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

# Merges the array tasks' results into the report's tables, then renders the
# report if pandoc is available. Queued with afterany, so it runs even if some
# tasks failed; the tables cover whatever finished and the log lists the rest.

set -euo pipefail

mkdir -p logs

UF="${DENGUE_UF:-41}"
YEAR="${DENGUE_YEAR:-2024}"
OUTDIR="results/real_data/dengue_sensitivity_proposals"

# Julia and R module trees in separate subshells: R's gfbf/2024a toolchain
# shadows the GLib that Julia's Glib_jll expects.
run_julia() {
  ( module purge
    module load bear-apps/2024a
    module load bluebear
    module load Julia/1.12.2-linux-x86_64
    julia --project=. "$@" )
}

run_r() {
  ( module purge
    module load bear-apps/2024a
    module load bluebear
    module load "${R_MODULE:-R/4.5.0-gfbf-2024a}"
    if [[ -n "${PANDOC_MODULE:-}" ]]; then module load "${PANDOC_MODULE}"; fi
    Rscript "$@" )
}

echo "=== collect ==="
run_julia src/dengue_sensitivity.jl --collect --uf "${UF}" --year "${YEAR}" ${SENS_ARGS:-}

echo
echo "=== report ==="
if run_r -e 'quit(status = as.integer(!rmarkdown::pandoc_available("1.12.3")))'; then
  run_r -e "rmarkdown::render('src/dengue_sensitivity_analysis.rmd',
                              output_file = 'dengue_sensitivity_analysis.html',
                              output_dir  = '${OUTDIR}',
                              params = list(results_dir = '${OUTDIR}'))"
  echo "Report: ${OUTDIR}/dengue_sensitivity_analysis.html"
else
  echo "pandoc not found, so the report was not rendered. The tables are complete."
  echo "Either set PANDOC_MODULE (see 'module avail Pandoc') and resubmit this job, or copy"
  echo "${OUTDIR}/ back without chains/ and tasks/ and render locally."
fi
