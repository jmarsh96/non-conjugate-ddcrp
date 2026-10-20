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
hours per study on 8 workers; dengue needs a cluster.

```bash
julia --project=. src/run_gibbs_comparison.jl --study skewnormal
julia --project=. src/run_gibbs_comparison.jl --study poisson
julia --project=. src/run_gibbs_comparison.jl --study dengue
julia --project=. src/run_gibbs_comparison.jl --study old_faithful
julia --project=. src/analyse_gibbs_comparison.jl
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

The paper lives in its own repo. If it is cloned into `docs/ddcrp-paper/`, the analyse scripts
will also copy the figures and tables into it.
