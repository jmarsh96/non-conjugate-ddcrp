#!/bin/bash
# ============================================================================
# Submit the Gibbs vs RJ comparison: an environment setup job, one job per
# study after it, and the analysis after all of them (afterany, so finished
# studies are written up even if one fails). Runs on a login node; no Julia here.
#
# Usage (from the repo root):
#   bash src/slurm/submit_gibbs_comparison.sh                      # all four studies
#   bash src/slurm/submit_gibbs_comparison.sh skewnormal dengue    # a subset
#   CMP_ARGS="--test" bash src/slurm/submit_gibbs_comparison.sh    # quick check
#   CMP_ARGS="--pilot" bash src/slurm/submit_gibbs_comparison.sh   # remeasure timings
#
# Sizing (chains / workers, wall-clock per chain, memory per worker at the N_MAX cap):
#   skewnormal    88 chains / 30 workers, 10 min, ~2.5 GB  -> 3 rounds, ~45 min
#   poisson       48 chains / 24 workers, 10 min, ~1.5 GB  -> 2 rounds, ~30 min
#   dengue        40 chains / 20 workers, 60 min, ~5 GB    -> 2 rounds, ~2.5 h
#   old_faithful  48 chains / 24 workers, 15 min, ~2.5 GB  -> 2 rounds, ~45 min
# Each also runs a pilot first (a few minutes). Times requested are ~3x that.
# ============================================================================

set -euo pipefail
mkdir -p logs

studies=("$@")
if [[ ${#studies[@]} -eq 0 ]]; then studies=(skewnormal poisson dengue old_faithful); fi

declare -A NTASKS=([skewnormal]=30 [poisson]=24 [dengue]=20 [old_faithful]=24)
declare -A MEM=([skewnormal]=90G [poisson]=48G [dengue]=110G [old_faithful]=72G)
declare -A TIME=([skewnormal]=04:00:00 [poisson]=03:00:00 [dengue]=10:00:00 [old_faithful]=03:00:00)

setup=$(sbatch --parsable --job-name=ddcrp_gibbs_setup --ntasks=1 --mem=8G --time=01:00:00 \
  --output=logs/ddcrp_gibbs_setup_%j.out --error=logs/ddcrp_gibbs_setup_%j.err \
  --wrap="module purge; module load bear-apps/2024a; module load Julia/1.12.2-linux-x86_64; \
          julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'")
echo "Submitted setup     : job ${setup}"

deps=()
for s in "${studies[@]}"; do
  if [[ -z "${NTASKS[$s]:-}" ]]; then
    echo "Unknown study '$s' (skewnormal, poisson, dengue, old_faithful)" >&2
    exit 1
  fi
  if [[ "$s" == "dengue" ]]; then
    for f in data/cleaned/dengue_uf41_2024.csv data/cleaned/dengue_uf41_adjacency.csv; do
      [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 1; }
    done
  fi
  id=$(sbatch --parsable --export=ALL,STUDY="$s",CMP_ARGS="${CMP_ARGS:-}" \
       --job-name="gibbs_${s}" --ntasks="${NTASKS[$s]}" --mem="${MEM[$s]}" --time="${TIME[$s]}" \
       --dependency=afterok:"${setup}" src/slurm/07_gibbs_comparison.slurm)
  echo "Submitted ${s}$(printf '%*s' $((14 - ${#s})) ''): job ${id}  (${NTASKS[$s]} workers, ${MEM[$s]}, ${TIME[$s]})"
  deps+=("${id}")
done

dep_spec=$(IFS=:; echo "${deps[*]}")
analysis=$(sbatch --parsable --export=ALL,CMP_ARGS="${CMP_ARGS:-}" --dependency=afterany:"${dep_spec}" \
  src/slurm/08_gibbs_comparison_analysis.slurm)
echo "Submitted analysis  : job ${analysis}  (afterany:${dep_spec})"

echo
echo "Monitor: squeue -u \$USER"
echo "Results: results/gibbs_comparison/"
