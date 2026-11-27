# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

The experiment scripts behind a paper on **non-conjugate RJMCMC samplers for the
distance-dependent CRP (ddCRP)**. The repo is kept to exactly what the paper and its supplement
use, so that the work is reproducible from a clone: four studies, a sampler comparison across them,
their data, and their small outputs. `README.md` is the user-facing account; keep it in step with this file.

| Study | Paper | Scripts |
|---|---|---|
| Skew-normal simulation | Section 3.1, S6 | `run_skewnormal.jl`, `geweke_skewnormal.jl`, `analyse_skewnormal.jl` |
| Dengue in Paraná | Section 3.2, S8 | `run_dengue.jl`, `analyse_dengue.jl`, `map_dengue.R` |
| Poisson simulation | S5 | `run_poisson.jl`, `analyse_poisson.jl` |
| Old Faithful | S7 | `run_old_faithful.jl`, `analyse_old_faithful.jl` |
| Aux Gibbs vs RJ | (revision) | `run_gibbs_comparison.jl --study <s>`, `analyse_gibbs_comparison.jl` |
| Birth oracle bound | Section 4.5, S10 | `diagnose_births.jl`, `analyse_births.jl` |
| Scaling with n and s | S12 (revision) | `run_scaling.jl`, `analyse_scaling.jl` |
| Replicated simulated data sets | S13 (revision 2) | `run_replicates.jl`, `analyse_replicates.jl` |
| Dengue prior-centre sensitivity | S8 (revision) | `run_dengue_prior.jl [--analyse]` |

Do not add code the paper does not use. The AQS application, the dengue prior-sensitivity
study and the mode-partition map helpers were removed for that reason (they are in git history).
So were the sampler-study moves (coupled birth/death, informed link, two-anchor split/merge;
`run_sampler_study.jl`, and in DDCRP.jl `Coupled`, `InformedLink`, `TwoAnchor`): none beat its
base sampler per second, so do not re-propose them without a new idea.

The paper source is a **separate repo** cloned at `docs/ddcrp-paper/` (`docs/` is git-ignored
here). The analyse scripts and `map_dengue.R` copy the paper's figures and tables into
`docs/ddcrp-paper/{figures,tables}/` when that directory exists, and skip the copy under `--test`.

## The DistanceDependentCRP dependency (critical)

All modelling code lives in the external package **`DistanceDependentCRP`**, developed at
`/home/marshjs/Documents/Projects/DDCRP/` and pulled from GitHub
(`https://github.com/jmarsh96/DDCRP.jl.git`, `rev="main"`). `Manifest.toml` is committed and
pins the exact commit.

- To add/change models, proposals, or the MCMC engine, **edit files in `../DDCRP/`**, not
  this repo. The user pushes those changes to GitHub; the run scripts here are unchanged.
- **Never** switch the run scripts to `Pkg.develop(path=...)`.
- After pushing a change to DDCRP.jl, refresh it here with
  `julia --project=. -e 'using Pkg; Pkg.update("DistanceDependentCRP")'` and commit the Manifest.

The two exceptions are the locally prototyped models `src/skewnormal_model.jl` and
`src/negbin_population_model.jl`, which add methods to the package's generic functions
(`cluster_param_dicts`, `table_contribution`, `sample_birth_param`, ...) and are `include`d by
the scripts that need them.

## Commands

Julia 1.12. Setup is `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.

```bash
# Skew-normal simulation (full: ~5 h on 8 local workers)
julia --project=. src/run_skewnormal.jl [--test] [--slurm] [--infer_s]
julia --project=. src/geweke_skewnormal.jl [--test] [--rerun]    # after run_skewnormal
julia --project=. src/analyse_skewnormal.jl [--test]

# Dengue (full: ~5 h on 18 workers)
julia --project=. src/prepare_dengue_data.jl --uf 41 --year 2024   # cached in data/cleaned/
Rscript src/prepare_dengue_adjacency.R --uf 41                     # cached in data/cleaned/
julia --project=. src/run_dengue.jl [--pilot] [--test] [--slurm] [--workers N]
julia --project=. src/analyse_dengue.jl [--test]
Rscript src/map_dengue.R [--test]

# Poisson simulation (full: hours on 8 workers)
julia --project=. src/run_poisson.jl [--test] [--slurm] [--infer_s]
julia --project=. src/analyse_poisson.jl

# Old Faithful
julia --project=. src/run_old_faithful.jl [--test]
julia --project=. src/analyse_old_faithful.jl

# Aux Gibbs vs RJ (after the four studies: reads their tuned proposals)
julia --project=. src/run_gibbs_comparison.jl --study {skewnormal|poisson|dengue|old_faithful} [--configs "a;b"] [--pilot] [--tune] [--test] [--slurm] [--workers N]
julia --project=. src/analyse_gibbs_comparison.jl [--test] [--tables]

# Birth/death bottleneck (simulated skew-normal; docs/sampler_improvements.md)
julia --project=. src/diagnose_births.jl [--test] [--workers N]
julia --project=. src/analyse_births.jl [--test]

# Revision experiments: scaling (after run_poisson) and dengue prior sensitivity (after analyse_dengue)
julia --project=. src/run_scaling.jl [--test] [--slurm] [--workers N]
julia --project=. src/analyse_scaling.jl [--test]
julia --project=. src/run_dengue_prior.jl [--analyse] [--test] [--slurm] [--workers N]

# Revision 2: replicated simulated data sets (after the simulated studies and the comparison)
julia --project=. src/run_replicates.jl [--test] [--slurm] [--workers N]
julia --project=. src/analyse_replicates.jl [--test]
```

There is no test suite; `--test` is a reduced-iteration mode that writes to a `*_test/`
directory. Tests for the models belong in `../DDCRP/`. Smoke-test at `--test` scale and
leave the multi-hour runs to the user.

Slurm batch scripts for BlueBear live in `src/slurm/`, which is **git-ignored** (local helpers,
not part of the reproducible repo). The `--slurm` flag in the run scripts is tracked: it turns
every task of the allocation into a worker via `SlurmClusterManager`.

## Architecture

Each study is a **run** script (MCMC; writes chains and summary CSVs) paired with an **analyse**
script (reads those outputs; makes figures and tables). `run_and_save(cfg)` is the unit of work
on each worker: it runs `mcmc(...)`, burns and thins, computes diagnostics (ARI, VI, ESS, ESJD,
acceptance rates), saves the chain, and returns a NamedTuple that becomes a CSV row.

### `run_poisson.jl`

`n=150` Poisson counts, `K=3`, rates `(1, 4, 7)`, 1-D covariate. Part 1 sweeps 7 configs (the
Gibbs baseline `PoissonClusterRatesMarg` plus `PoissonClusterRates` under birth ∈ {Prior, NMM,
LNMM} × fixed-dim ∈ {`NoUpdate`, `Resample`}); Part 2 tunes σ by **ESJD(K)**; Part 3 runs the
best-tuned variants at full length. `analyse_poisson.jl` also writes `validation_overlapping.csv`
(TV(K), co-clustering and rate differences against Gibbs) for Supplementary Table S1.

### `run_skewnormal.jl`

`n=150` skew-normal observations, `K=3`, two scenarios (`overlapping`, `separated`). Three
parameters (ξ, ω, and shape α in the code, **κ in the paper**), fully non-conjugate, **no Gibbs
baseline**: correctness rests on split-R̂ across 4 seeds, agreement across proposal families,
and the prior-recovery check.

- A **coordinate design** around the baseline `ξ:NMM ω:LNMM α:NMM`, swapping one parameter's
  family at a time, plus all-Prior and `ξ:NMM ω:Prior α:Prior`, each × {`NoUpdate`, `Resample`}:
  16 configs × 2 scenarios × 4 seeds = 128 chains.
- Part 1 tunes the birth σ per parameter by ESJD(K) and writes `tuned_sigma.csv`. Part 2 runs
  every chain; only the first seed's is saved, already thinned (burn = 0, thin = 1).
- **Selection is by ESJD(K); ESS(K) is reported beside it.**
- Small moving sets (|M| < `MM_MIN_SIZE` = 3): ξ centres on the moving-set mean, ω and α are
  drawn from their priors.
- **Resample does not mix for this model**: its P(K=3) depends on the starting state while its
  ESS(K) and split-R̂ look healthy. The paper compares birth families under No Update only.
- `geweke_skewnormal.jl` sets `LIKELIHOOD_ON[] = false` so the target is the prior, and compares
  P(K) with 200k `simulate_ddcrp` draws and each parameter's PIT with Uniform(0,1) as z-scores.
  `--rerun` reruns the three flagged Resample configs at 4x length. `analyse_skewnormal.jl`
  turns both CSVs into Supplementary Tables S3 and S4.

### `run_dengue.jl`

`y_i ~ NegBin(mean = E_i ρ_k, size = r_k)` over the 399 municipalities of Paraná (UF 41), 2024,
`E_i = pop_i / 1e5`. Both cluster parameters are non-conjugate; no Gibbs baseline.

- **The ddCRP prior is the hop window of radius `HOP_RADIUS = 2`** on the queen contiguity
  graph: `D` is 0 within 2 hops and `LOG_FLOOR = 30` beyond, with kernel rate 1 (`hop_window_D`
  in `dengue_common.jl`). A hard zero weight would put `-Inf` into the log-posterior.
- **Links are proposed from the prior** (`link_proposal = PriorLink()`). Do not revert to
  `UniformLink` while the prior is windowed: almost every uniform proposal lands outside it.
- Part 1: 16-config proposal sweep; Part 2: σ tuning of the winner; Part 3: four chains of the
  winner at 250k post-burn-in iterations (`FINAL_SEEDS`) plus a Poisson comparator scored by
  WAIC/LOO. Selection is ESJD(K), ties broken by ESS(K).
- **Partition point estimates are Dahl's least-squares partition** (`dahl_point_estimate`), here
  and for Old Faithful. Do not reintroduce the highest log-posterior draw at the modal K: it is
  not a MAP partition.
- `analyse_dengue.jl` writes `dengue_numbers.tex`, the reference for every number the paper's
  dengue section quotes. The paper hard-codes those numbers, so update the text by hand after a
  rerun.
- The contiguity graph is built in R (`prepare_dengue_adjacency.R`, `sf::st_touches`) and cached
  in `data/cleaned/`. Maps are R + `sf` (`map_dengue.R`).

### `run_gibbs_comparison.jl`

Exact auxiliary Gibbs (`AuxiliaryGibbs(q, m)`, Supplement S1) against RJ on the same posterior.
Each study's setup is in a shared file (`*_common.jl`, `load_dengue_study`) included by both its
run script and this one.

- Gibbs q is the tuned RJ birth proposal (`q:mm`) or the prior (`q:prior`), m ∈ {1, 3, 10, 30}.
- RJ runs as published (`RJ q:... link:...`) and as "RJ pool": `AuxiliaryMH(q, m)`, the Metropolised
  Algorithm 8, m ∈ {1, 3, 10, 30}, with `PriorLink` or `MetropolisedPriorLink`. At m = 1 with `PriorLink`
  it is the published sampler in distribution, scored on the moving set only, so the gap between the
  two is implementation cost alone.
- Four starts shared across samplers via `mcmc(...; init_c, init_proposal)`.
- Equal wall-clock per chain from `pilot_timing.csv` (`BUDGET`, capped by `N_MAX` for memory).
- Dengue's Gibbs rows use `AuxiliaryGibbs(q, m; skip = 10)` (`GIBBS_SKIP`): outside the hop window
  every block has log link weight about -30, so the moving set is scored only under the near blocks.
- `--configs "a;b"` runs only the configs whose labels start with a or b and replaces their rows in
  `chains_<study>.csv`; with `--pilot` it re-pilots only those (needed when their cost changes).
- Old Faithful's `q:prior` is its published proposal: IGMM falls back to the prior for a Gamma shape.
  Its `q:mm` is LNMM with σ tuned by ESJD(K) in the comparison script itself
  (`results/old_faithful/tuned_sigma.csv`; `--tune` reruns it).

### `diagnose_births.jl`

Oracle bound A* = min{1, m(y_M)/p(y_M|ρ_O)} vs observed acceptance of the births the RJ pool
baseline (m = 1, prior links) proposes (m(y_M) by quadrature over log γ for Old Faithful, defensive
IS otherwise), and the links separating the Dahl partitions of each K. Result: A* is itself ~1%, so
better parameter proposals alone cannot fix the bottleneck. Scenarios from `make_scenario(name)` in
`skewnormal_common.jl`: overlapping, separated (the paper's data), `aba` (A1 | B | A2 along x, a
multi-link barrier) and `k5` (n = 300). Outputs go to `results/sampler_study/`; `analyse_births.jl`
turns `diagnose_births_summary.csv` into the oracle tables.

### Paper tables

The paper `\input`s every comparison table exactly as the analyse scripts write it; do not hand-edit
the copies in `docs/ddcrp-paper/tables/`. `analyse_gibbs_comparison.jl` writes `table_gibbs_vs_rj`,
`table_gibbs_vs_rj_full_{a,b,c}` (skew-normal; Poisson and dengue; Old Faithful, one supplement page
each) and `table_rj_ladder`, with config labels typeset by `texlabel` (MM, PL/ML/UL);
`--tables` rebuilds them from the tracked summary CSVs, since the chains are not tracked.
`analyse_births.jl` writes, from `diagnose_births_summary.csv`, `table_births_oracle` (supplement)
and `table_births_oracle_main` (main text).

### The ddCRP decay scale `s` is a RATE

`exp_decay(d; scale) = exp(-d * scale)`: **larger `s` means faster decay and a more local
prior**. Do not write the kernel as `exp(-d/s)`.

### RJMCMC building blocks (from `DistanceDependentCRP`)

- **Birth proposals**: `PriorProposal`, `NormalMomentMatch(σ)`, `LogNormalMomentMatch(σ)`,
  `InverseGammaMomentMatch()`, `ConjugateProposal()`; composed per parameter with `MixedProposal`.
- **Fixed-dim proposals**: `NoUpdate()` or `Resample(<proposal>)` (`MixedFixedDim` per parameter).
- **Auxiliary Gibbs**: pass `AuxiliaryGibbs(q, m)` as the birth proposal for the exact
  Algorithm 8 link update (fixed-dim and link proposals are then ignored). `skip` (keyword) leaves
  blocks whose link weight from `i` is below `exp(-skip)` of the largest out of the Gibbs draw and
  reaches them by an exact MH move between near and far blocks.
- **RJ pool**: `AuxiliaryMH(q, m)` is the Metropolised Algorithm 8 (fixed-dim proposal ignored;
  link proposal used, including `MetropolisedPriorLink()`, which only it supports).
- **Moving-set scoring**: a model that defines `cluster_logprior(model, params, priors)` declares
  that its `table_contribution` factorises over observations, and the auxiliary samplers then score
  the moving set in O(|M|). Defined for `PoissonClusterRates`, `SkewNormalClusters` and
  `NegBinPopulationRates`; Old Faithful's partially marginalised model uses the generic path.
- In `run_gibbs_comparison.jl` a config missing from `pilot_timing.csv` is piloted on its own, so
  earlier pilots are kept.
- `analyse_gibbs_comparison.jl --study <s>` recomputes the chain metrics of one study from its chains
  and reads the tracked `chain_metrics_*.csv` for the others; `--tables` reads them all. Table 2 and
  the full tables carry min--max ranges of ESS(K)/s over the four chains, from `chain_metrics_*.csv`.
- **Moving-set statistics**: a model that defines `suff_stats` and `stats_contribution` (additive
  sufficient statistics; `GammaClusterShapeMarg` since DDCRP.jl 1.2.0) is scored by the auxiliary
  samplers in O(|M|) + O(1) per block from a per-state memo, although its likelihood does not
  factorise. Old Faithful's comparison numbers therefore come from 1.2.0; the other studies from 1.1.1.
- `mcmc(model, y, D, ddcrp_params, priors, birth; fixed_dim_proposal, opts)` returns
  `(samples, diagnostics)`.
- Helpers: `calculate_n_clusters`, `compute_table_assignments`, `compute_ari_trace`,
  `compute_vi_trace`, `effective_sample_size`, `acceptance_rates`.

### RJMCMC improvement constraint

Improvements to the non-conjugate samplers **must generalise to non-conjugate models**. Do not
propose Rao-Blackwellisation or marginalisation tricks.

## Data representation

Cluster assignments use the **pointer representation** `c`: each observation points to the first
member of its cluster. Use `compute_table_assignments(c)` for flat `1..K` labels.

## Outputs and what is tracked

`.gitignore` ignores everything under `results/` except `*.csv`, `*.tex`, `*.png` (and the
skew-normal `*.txt`) in the four paper studies' directories:
`results/simulation_study/{skewnormal,poisson}/`, `results/real_data/dengue/`,
`results/old_faithful/`, `results/gibbs_comparison/`, `results/sampler_study/`, `results/scaling/`,
`results/replicates/`. Chains, `.jld2` files and `*_test/` directories are never tracked. A
new output a reader needs must match that whitelist.

`data/cleaned/` is committed and holds everything a run, analyse or map script reads;
`data/raw/` is a git-ignored download cache for the `prepare_*` scripts.

`poster/`, `talk/`, `logs/` and `docs/` are git-ignored local directories and not part of the
reproducible repo.

Chain filenames are derived from the config label via
`fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")`; a label change in a run script must
stay consistent with its analyse script.

## Conventions

- Keep code comments short: a brief file header with usage, single-line section markers, and a
  comment only where the code is not self-explanatory.
- Plotting is **StatsPlots (GR backend)** with `ENV["GKSwstype"] = "nul"` for headless runs.
  Every analyse script includes `src/plot_style.jl`: the Okabe-Ito colourblind-safe palette
  (`CATEGORICAL`, `OI_*`), `GREY_RAMP` for magnitudes, 300 dpi. Do not use `tab10` or named colours.
  Maps use viridis scales in R.
- Spatial work uses **R + `sf`**, not Python/geopandas.
- No em dashes in prose.
- Paper drafting uses the `paper-writing` skill.
