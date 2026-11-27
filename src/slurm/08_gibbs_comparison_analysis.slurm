#!/bin/bash
# Tables, numbers and figure for the Gibbs vs RJ comparison, over every study that finished.

#SBATCH --job-name=ddcrp_gibbs_cmp_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ddcrp_gibbs_cmp_analysis_%j.out
#SBATCH --error=logs/ddcrp_gibbs_cmp_analysis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail

module purge
module load bear-apps/2024a
module load Julia/1.12.2-linux-x86_64
export GKSwstype=nul

mkdir -p logs

julia --project=. src/analyse_gibbs_comparison.jl ${CMP_ARGS:-}
