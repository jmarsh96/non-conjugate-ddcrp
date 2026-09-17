#!/bin/bash

#SBATCH --job-name=ddcrp_dengue_sens
#SBATCH --nodes=1
#SBATCH --ntasks=19
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=logs/ddcrp_dengue_sens_%j.out
#SBATCH --error=logs/ddcrp_dengue_sens_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk


set -euo pipefail

mkdir -p logs

UF="${DENGUE_UF:-41}"
YEAR="${DENGUE_YEAR:-2024}"

# Both inputs are committed to data/cleaned/, and compute nodes have no internet, so
# fail loudly here rather than several minutes into the allocation.
for f in "data/cleaned/dengue_uf${UF}_${YEAR}.csv" "data/cleaned/dengue_uf${UF}_adjacency.csv"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING $f" >&2
    echo "Build it on a LOGIN node:" >&2
    echo "  julia --project=. src/prepare_dengue_data.jl --uf ${UF} --year ${YEAR}" >&2
    echo "  Rscript src/prepare_dengue_adjacency.R --uf ${UF}" >&2
    exit 1
  fi
done

# The Julia and R module trees must NOT be loaded at the same time: the
# gfbf/2024a toolchain that R pulls in puts its own GLib on LD_LIBRARY_PATH,
# which shadows the GLib that Julia's Glib_jll artifact expects, and Julia then
# dies at startup with "libgobject-2.0.so: undefined symbol: g_string_copy".
# So each language runs in its own subshell with only its own modules loaded.

run_julia() {
  ( module purge
    module load bear-apps/2024a
    module load bluebear
    module load "${JULIA_MODULE:-Julia/1.12.2-linux-x86_64}"
    julia --project=. "$@" )
}

run_r() {
  ( module purge
    module load bear-apps/2024a
    module load bluebear
    module load "${R_MODULE:-R/4.5.0-gfbf-2024a}"
    Rscript "$@" )
}

echo "=== chains ==="
run_julia src/dengue_sensitivity.jl --slurm --uf "${UF}" --year "${YEAR}"

echo
echo "=== report ==="
# Rendered next to the tables it reads, so the outputs of one study stay
# together. knitr writes its cache and figures into the working directory, so
# render from the repo root exactly as the interactive command does.
run_r -e 'rmarkdown::render("src/dengue_sensitivity_analysis.rmd",
                            output_file = "dengue_sensitivity_analysis.html",
                            output_dir  = "results/real_data/dengue_sensitivity")'

echo
echo "Done. Report: results/real_data/dengue_sensitivity/dengue_sensitivity_analysis.html"
