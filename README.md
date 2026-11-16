# non-conjugate-ddcrp

Code to reproduce the results in "Bayesian Inference for Non-Conjugate Distance-Dependent
Chinese Restaurant Process Models".

The samplers are implemented in a separate Julia package,
[DDCRP.jl](https://github.com/jmarsh96/DDCRP.jl). This repo only contains the scripts that run
the experiments, the data, and the summary output.

## Getting started

You need Julia 1.12. From the repo root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs the package versions recorded in `Manifest.toml`, which are the ones used for
the paper. The dengue maps are drawn in R and need `sf`, `ggplot2` and `dplyr`.

## Running things

There are four studies, plus a comparison of samplers that reuses all four. For each one, the `run_` script does the MCMC and the `analyse_` script
makes the tables and figures from its output, so run them in that order.

Skew-normal simulation study (Section 3.1 and Supplement S6). Takes about 5 hours.

```bash
julia --project=. src/run_skewnormal.jl
julia --project=. src/geweke_skewnormal.jl
julia --project=. src/geweke_skewnormal.jl --rerun
julia --project=. src/analyse_skewnormal.jl
```

`geweke_skewnormal.jl` is the prior-recovery check, and `--rerun` repeats the configurations
it flagged with longer chains.

Dengue in Paraná (Section 3.2 and Supplement S8). This is the expensive one: about 5 hours
with 18 workers, which we ran on a cluster. Use `--workers N` to change the number of workers.

```bash
julia --project=. src/run_dengue.jl
julia --project=. src/analyse_dengue.jl
Rscript src/map_dengue.R
```

Poisson simulation study (Supplement S5). A few hours.

```bash
julia --project=. src/run_poisson.jl
julia --project=. src/analyse_poisson.jl
```

Old Faithful (Supplement S7). One chain, so this is quick by comparison.

```bash
julia --project=. src/run_old_faithful.jl
julia --project=. src/analyse_old_faithful.jl
```

Auxiliary Gibbs against reversible jump, on all four studies. Reversible jump runs both as
published and as the Metropolised Algorithm 8 ("RJ pool"), which adds an auxiliary pool to births
and deaths and an optional partition-changing link proposal. Run it after the studies, since it
reads their tuned proposals. A timing pilot sets the chain lengths (`--pilot` repeats it). About 2
hours per study on 8 workers; dengue needs a cluster. On dengue, Gibbs scores each moving set only
under the clusters inside the hop window and reaches the rest by an exact Metropolis-Hastings move.
`--configs "Gibbs"` reruns only the configs whose labels start with that text, keeping the other rows.

```bash
julia --project=. src/run_gibbs_comparison.jl --study skewnormal
julia --project=. src/run_gibbs_comparison.jl --study poisson
julia --project=. src/run_gibbs_comparison.jl --study dengue
julia --project=. src/run_gibbs_comparison.jl --study old_faithful
julia --project=. src/analyse_gibbs_comparison.jl [--tables]
```

`--tables` rebuilds the tables and figure from the tracked `summary_<study>.csv` files without the
chains, which are not tracked.

The birth oracle diagnostic. `diagnose_births.jl` compares the acceptance of the births RJ pool
proposes with the best any parameter proposal could achieve, and `analyse_births.jl` writes its
tables.

```bash
julia --project=. src/diagnose_births.jl
julia --project=. src/analyse_births.jl
```

Scaling with n and the decay rate (Supplement S12). Reruns the best RJ pool and auxiliary Gibbs
configurations, the collapsed sampler and the first RJ implementation on Poisson data with n up to
2,400 and decay rates from 0.125 to 8, for a fixed number of sweeps. Needs the tuned Poisson
proposal, so run `run_poisson.jl` first. About an hour on 12 workers.

```bash
julia --project=. src/run_scaling.jl
julia --project=. src/analyse_scaling.jl
```

Dengue prior sensitivity (Supplement S8). Reruns the final dengue sampler with the prior on the
incidence centred at half and twice the state-wide incidence, one chain each, and compares them
with the four main chains. Needs the main dengue chains and `similarity_matrix.jld2`, so run
`run_dengue.jl` and `analyse_dengue.jl` first. About an hour per chain; `--analyse` rebuilds the
table from existing chains.

```bash
julia --project=. src/run_dengue_prior.jl
```

The run scripts use 8 local worker processes unless told otherwise, and `--slurm` makes them
use a Slurm allocation instead. Seeds are set inside the scripts.

If you just want to check that everything works, add `--test` to any run script. This runs
very short chains (a few minutes at most) and writes to a separate `_test` folder, so it will
not touch the real results.

## Data

The data are in `data/cleaned/`.

`dengue_uf41_2024.csv` has the 2024 dengue case counts, population and centroid of each of the
399 municipalities in Paraná, downloaded from InfoDengue. `dengue_uf41_adjacency.csv` is the
contiguity graph between them and `malha_uf41.geojson` holds the IBGE boundaries used for the
maps. `faithful.csv` is the Old Faithful data set from R.

`src/prepare_dengue_data.jl` and `src/prepare_dengue_adjacency.R` are the scripts that built
the dengue files. You do not need to run them, but they will download everything again if you
pass `--force`.

## Output

Everything is written to `results/`. The summary CSVs, tables and figures that the paper uses
are included in the repo, so you can look at the numbers without running anything. The chains
themselves are not included as they come to tens of gigabytes; the run scripts recreate them.

- `results/simulation_study/skewnormal/`
- `results/real_data/dengue/` (`dengue_numbers.tex` lists the numbers quoted in the text)
- `results/simulation_study/poisson/`
- `results/old_faithful/`
- `results/gibbs_comparison/` (`gibbs_comparison_numbers.tex` lists the numbers for the text)
- `results/sampler_study/` (birth oracle diagnostic)
- `results/scaling/` (scaling study)

## Which script writes what

Every table and figure in the paper is written by one of the analyse scripts, which copy it into
`docs/ddcrp-paper/{tables,figures}/` when that directory exists.

| Paper | Output | Script |
|---|---|---|
| Tables 2, 3, Figure 1, Table S8 (a to c) | `table_gibbs_vs_rj.tex`, `table_rj_ladder.tex`, `gibbs_vs_rj_ess.png`, `table_gibbs_vs_rj_full_{a,b,c}.tex` | `analyse_gibbs_comparison.jl` |
| Table 4, Table S9 | `table_births_oracle_main.tex`, `table_births_oracle.tex` | `analyse_births.jl` |
| Table 5, Figure 2, Tables S2 to S5, Figures S4, S5 | `table_skew_*.tex`, `skew_*.png` | `analyse_skewnormal.jl` |
| Table S1, Figures S1 to S3 | `table_overlapping.tex`, `data_overview.png`, `k_alpha_traces.png`, `pk_distribution.png` | `analyse_poisson.jl` |
| Figures S6 to S8 | `k_posterior.png`, `traces.png`, `point_clustering_arrows.png`, `ppc.png` | `analyse_old_faithful.jl` |
| Figures 3, 4, S9, S10, Table S6, numbers in Section 6 | `dengue_*.png`, `table_dengue.tex`, `dengue_numbers.tex` | `analyse_dengue.jl`, `map_dengue.R` |
| Table S7 | `table_dengue_prior.tex` | `run_dengue_prior.jl` |
| Table S10 | `table_scaling.tex` | `analyse_scaling.jl` |

## Compute

The paper's runs used Julia 1.12 with the package versions in `Manifest.toml`
(`DistanceDependentCRP` v1.1.1 at the pinned commit), one Julia thread and one BLAS thread per
worker, and one chain per core, on Intel Xeon Platinum 8360Y (Ice Lake) nodes of the University
of Birmingham's BlueBEAR cluster. The reported times are per chain, burn-in included, and exclude
compilation and the pilot runs.

The long or memory-heavy runs are the dengue study (about 5 hours on 18 workers), the sampler
comparison (one to six hours per study; the dengue comparison needs about 5 GB per worker at the
iteration cap) and the skew-normal study (about 5 hours on 8 workers). Everything else runs in
minutes to an hour on a laptop. `DDCRP.jl` is released under the MIT licence.

The paper lives in its own repo. If it is cloned into `docs/ddcrp-paper/`, the analyse scripts
will also copy the figures and tables into it.
