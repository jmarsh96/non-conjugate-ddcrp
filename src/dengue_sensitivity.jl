# ============================================================================
# dengue_sensitivity.jl
#
# Prior × proposal study for the Brazil dengue application. Three things vary,
# with the likelihood, the priors on ρ and r and the seed held fixed:
#
#   1. The ddCRP prior: the hop window at s = 1, 2, 3 hops, and the great-circle
#      metric at s = 1, 5, 10 and with s inferred.
#   2. Whether the ddCRP concentration α is inferred (Gamma(1, 0.01) prior) or
#      fixed at α = 1.
#   3. The RJMCMC proposal: a coordinate design over the birth family for ρ and
#      r (the run_dengue.jl Part 1 design), crossed with fixed-dim
#      ∈ {NoUpdate, Resample}.
#
# 7 priors × 2 α settings × 16 proposals = 224 configurations. Axes 1 and 2
# change the posterior; axis 3 does not, so within a prior × α cell the 16
# proposals must agree up to Monte Carlo error and differ only in how well they
# mix. The report picks the best-mixing proposal per cell and compares the
# cells on predictive fit.
#
# FOR THE HOP METRIC, s IS THE WINDOW RADIUS IN HOPS, not a rate. The kernel is
# uniform inside the window and effectively zero outside it. The package still
# needs a rate for exp(-rate*d), so hop configurations pass rate 1 on a
# 0 / LOG_FLOOR distance matrix (`hop_window_D` in dengue_common.jl). `s` is the
# reported value; `kernel_rate` is what reaches DDCRPParams.
#
# FOR THE GREAT-CIRCLE METRIC, s IS A RATE: exp_decay(d; scale) = exp(-d*scale)
# (priors.jl:55), so LARGER s is a MORE LOCAL prior, with decay length
# D_SCALE / s on distances rescaled by the median pairwise distance.
#
# WHY METRICS ARE DISTANCE MATRICES, NOT CUSTOM KERNELS. `inference/
# hyperparams.jl` hard-codes exp(-s*d_ij) in the α and s updates regardless of
# `decay_fn`, so a custom kernel would leave the α update inconsistent with the
# prior in force.
#
# Run `src/prepare_dengue_data.jl` and `src/prepare_dengue_adjacency.R` first.
#
# Usage:
#   julia --project=. src/dengue_sensitivity.jl --list          # grid size (writes nothing)
#   julia --project=. src/dengue_sensitivity.jl --task 17       # one config (Slurm array task)
#   julia --project=. src/dengue_sensitivity.jl --collect       # merge finished tasks into tables
#   julia --project=. src/dengue_sensitivity.jl --pilot         # per-task cost, before the grid
#   julia --project=. src/dengue_sensitivity.jl --test          # tiny grid, run locally, collected
#   julia --project=. src/dengue_sensitivity.jl --workers 12    # whole grid locally, collected
#
#   --iterations / --burn / --thin override the chain length.
#   --force reruns a --task whose result already exists.
#
# On Slurm: bash src/slurm/submit_dengue_sensitivity.sh
# Then:     Rscript -e 'rmarkdown::render("src/dengue_sensitivity_analysis.rmd")'
# ============================================================================

using DistanceDependentCRP, CSV, DataFrames, Distances, Statistics, StatsBase,
      Random, JLD2, Printf, Distributions, LinearAlgebra
using Distributed

const IS_TEST    = "--test"    in ARGS
const IS_PILOT   = "--pilot"   in ARGS
const IS_LIST    = "--list"    in ARGS
const IS_COLLECT = "--collect" in ARGS
const FORCE      = "--force"   in ARGS

function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) ? default : parse(Int, ARGS[i+1])
end

const UF      = arg_value("--uf", 41)      # 41 = Paraná (see prepare_dengue_data.jl)
const YEAR    = arg_value("--year", 2024)
const TASK_ID = arg_value("--task", 0)     # 1-based index into the grid

# Workers only when this process runs several chains itself; an array task runs
# one chain, and --list / --collect run none.
const RUN_LOCAL = TASK_ID == 0 && !IS_LIST && !IS_COLLECT
RUN_LOCAL && addprocs(arg_value("--workers", 8))

@everywhere begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using DistanceDependentCRP, Statistics, StatsBase, Distributions, JLD2, Random
    include(joinpath(@__DIR__, "negbin_population_model.jl"))
    include(joinpath(@__DIR__, "model_comparison.jl"))
    include(joinpath(@__DIR__, "dengue_common.jl"))
end

# ============================================================================
# Sampler settings shared by every configuration
#
# Proposal scales are the run_dengue.jl tuning winners. Only the proposal
# FAMILIES vary here; the scales are held fixed so a family is not favoured by
# having been tuned for.
# ============================================================================

@everywhere const PROP_SDS = (ρ = 0.15, r = 0.30)   # within-cluster MH, log scale
const SIGMA_BIRTH    = 0.50
const SIGMA_RESAMPLE = 0.10
const ALPHA_FIXED    = 1.0

@everywhere _birth_family(fam::Symbol, σ) =
    fam === :prior ? PriorProposal() :
    fam === :nmm   ? NormalMomentMatch(σ) :
    fam === :lnmm  ? LogNormalMomentMatch(σ) :
    fam === :ig    ? InverseGammaMomentMatch() :
    error("unknown family $fam")

@everywhere make_birth(fρ, fr, σ) =
    MixedProposal((ρ = _birth_family(fρ, σ), r = _birth_family(fr, σ)))

@everywhere make_fdim(fρ, fr, σr, resample::Bool) =
    resample ? MixedFixedDim((ρ = Resample(_birth_family(fρ, σr)),
                              r = Resample(_birth_family(fr, σr)))) : NoUpdate()

@everywhere thin_chain(s::NegBinPopulationRatesSamples, keep) =
    NegBinPopulationRatesSamples(s.c[keep, :], s.ρ[keep, :], s.r[keep, :],
                                 s.logpost[keep], s.α_ddcrp[keep], s.s_ddcrp[keep])

# ============================================================================
# Unit of work
#
# One configuration: run the chain, save it, and reduce it to the small tables
# the report needs. Each array task writes its reduction to tasks/<id>.jld2, and
# --collect stitches those together, so no chain is ever opened twice.
# ============================================================================

@everywhere function run_config(cfg)
    Random.seed!(cfg.seed)
    data = CountDataWithPopulation(cfg.y, cfg.E, cfg.D)

    opts = MCMCOptions(
        n_samples = cfg.n_samples,
        verbose = false,
        track_diagnostics = true,
        infer_params = Dict{Symbol,Bool}(:α_ddcrp => cfg.infer_alpha, :s_ddcrp => cfg.infer_s),
        prop_sds = Dict{Symbol,Float64}(
            :ρ => PROP_SDS.ρ, :r => PROP_SDS.r, :s_ddcrp => 0.3),
    )

    # α starts at 1 either way; with infer_alpha = false it stays there.
    ddcrp  = DDCRPParams(cfg.alpha_init, cfg.kernel_rate, DistanceDependentCRP.exp_decay,
                         1.0, 0.01, 1.0, 0.01)
    model  = NegBinPopulationRates()
    birth  = make_birth(cfg.fam_ρ, cfg.fam_r, cfg.σ_birth)
    fdim   = make_fdim(cfg.fam_ρ, cfg.fam_r, cfg.σ_resample, cfg.resample)

    total_time = @elapsed begin
        samples, diag = mcmc(model, data, ddcrp, cfg.priors, birth;
                             fixed_dim_proposal = fdim, opts = opts)
    end

    keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
    post = thin_chain(samples, keep)
    samples = nothing; GC.gc()          # the full chain is ~3 GB at 300k iterations
    n_post = length(keep)

    chain_dir = joinpath(cfg.outdir, "chains"); mkpath(chain_dir)
    jldsave(joinpath(chain_dir, cfg.config_id * ".jld2");
            samples = post, burn = 0, thin = 1,
            burn_applied = cfg.burn, thin_applied = cfg.thin,
            label = cfg.label, config_id = cfg.config_id)

    # ---- mixing diagnostics -------------------------------------------------
    k_trace = calculate_n_clusters(post.c)
    acc     = acceptance_rates(diag)
    ess_K   = effective_sample_size(Float64.(k_trace))
    ess_lp  = effective_sample_size(post.logpost)
    esjd_K  = mean(diff(Float64.(k_trace)).^2)
    med_ess(M) = median([effective_sample_size(M[:, i]) for i in axes(M, 2)])
    ess_s   = cfg.infer_s     ? effective_sample_size(post.s_ddcrp) : NaN
    ess_α   = cfg.infer_alpha ? effective_sample_size(post.α_ddcrp) : NaN

    summary = (
        config_id = cfg.config_id, prior_id = cfg.prior_id, proposal_id = cfg.proposal_id,
        label = cfg.label,
        metric = cfg.metric, hop_radius = cfg.hop_radius, s_decay = cfg.s,
        kernel_rate = cfg.kernel_rate, infer_s = cfg.infer_s, infer_alpha = cfg.infer_alpha,
        fam_rho = String(cfg.fam_ρ), fam_r = String(cfg.fam_r),
        fixed_dim = cfg.resample ? "Resample" : "NoUpdate",
        d_scale = cfg.d_scale,
        sigma_birth = cfg.σ_birth, sigma_resample = cfg.σ_resample,
        n_obs = length(cfg.y), n_samples = cfg.n_samples, n_post = n_post,
        mean_K = mean(k_trace), mode_K = Int(mode(k_trace)),
        std_K = std(Float64.(k_trace)),
        min_K = minimum(k_trace), max_K = maximum(k_trace),
        acc_birth = acc.birth, acc_death = acc.death,
        acc_fixed = acc.fixed, acc_overall = acc.overall,
        ess_K = ess_K, ess_logpost = ess_lp,
        ess_s = ess_s, ess_alpha = ess_α,
        med_ess_rho = med_ess(post.ρ), med_ess_r = med_ess(post.r),
        esjd_K = esjd_K,
        min_ess = min(ess_K, ess_lp), min_ess_per_sec = min(ess_K, ess_lp) / total_time,
        total_time = total_time,
        mean_logpost = mean(post.logpost),
        s_post = mean(post.s_ddcrp), alpha_post = mean(post.α_ddcrp),
    )

    # ---- traces, subsampled so the CSV stays small --------------------------
    tidx = n_post <= cfg.n_trace ? (1:n_post) :
           round.(Int, range(1, n_post; length = cfg.n_trace))
    traces = (draw = collect(tidx), K = k_trace[tidx], logpost = post.logpost[tidx],
              alpha_ddcrp = post.α_ddcrp[tidx], s_ddcrp = post.s_ddcrp[tidx])

    cm = countmap(k_trace)
    ks = sort(collect(keys(cm)))
    kpost = (K = ks, prob = [cm[k] / length(k_trace) for k in ks])

    # ---- point estimates ----------------------------------------------------
    sim   = compute_similarity_matrix(post.c)
    z_mod = mode_point_estimate(post.c, post.logpost)
    z_dahl = dahl_point_estimate(post.c, sim)
    conf  = [mean(sim[i, findall(==(z_mod[i]), z_mod)]) for i in eachindex(z_mod)]
    assign = (cluster_mode = z_mod, cluster_dahl = z_dahl,
              cocluster_confidence = conf,
              rho_mean = vec(mean(post.ρ, dims = 1)),
              r_mean = vec(mean(post.r, dims = 1)))

    # ---- likelihood scoring -------------------------------------------------
    scores, pointwise = score_fit_pointwise(cfg.label, loglik_matrix(model, post, data))

    # ---- posterior predictive ----------------------------------------------
    Random.seed!(11)
    y_rep = posterior_predictive(model, post, data; n_rep = cfg.n_rep)
    R = size(y_rep, 1)

    stat_names = String[]; T_obs = Float64[]; ppp = Float64[]
    q = NamedTuple[]
    rep_stat = String[]; rep_idx = Int[]; rep_val = Float64[]
    sub = R <= cfg.n_rep_out ? (1:R) : round.(Int, range(1, R; length = cfg.n_rep_out))
    for (sname, f) in ppc_statistics(cfg.E, cfg.nbrs)
        t_obs = f(cfg.y)
        t_rep = [f(view(y_rep, s, :)) for s in 1:R]
        push!(stat_names, sname); push!(T_obs, t_obs)
        push!(ppp, ppc_pvalue(t_obs, t_rep))
        push!(q, (mean = mean(t_rep), sd = std(t_rep),
                  q025 = quantile(t_rep, 0.025), q25 = quantile(t_rep, 0.25),
                  q50  = quantile(t_rep, 0.5),   q75 = quantile(t_rep, 0.75),
                  q975 = quantile(t_rep, 0.975)))
        append!(rep_stat, fill(sname, length(sub)))
        append!(rep_idx, collect(sub)); append!(rep_val, t_rep[sub])
    end
    ppc_stats = (statistic = stat_names, T_obs = T_obs, ppp = ppp,
                 mean = [x.mean for x in q], sd = [x.sd for x in q],
                 q025 = [x.q025 for x in q], q25 = [x.q25 for x in q],
                 q50 = [x.q50 for x in q], q75 = [x.q75 for x in q],
                 q975 = [x.q975 for x in q])
    ppc_reps = (statistic = rep_stat, rep = rep_idx, T_rep = rep_val)

    # Randomised PIT, so the discreteness of the counts does not by itself make
    # the histogram look miscalibrated.
    n = length(cfg.y)
    pit = [ (count(<(cfg.y[i]), view(y_rep, :, i)) +
             0.5 * count(==(cfg.y[i]), view(y_rep, :, i))) / R for i in 1:n ]
    ppc_point = (y_obs = cfg.y,
                 yrep_mean = vec(mean(y_rep, dims = 1)),
                 yrep_q025 = [quantile(view(y_rep, :, i), 0.025) for i in 1:n],
                 yrep_q50  = [quantile(view(y_rep, :, i), 0.5)   for i in 1:n],
                 yrep_q975 = [quantile(view(y_rep, :, i), 0.975) for i in 1:n],
                 pit = pit)

    result = (summary = summary, traces = traces, kpost = kpost, assign = assign,
              scores = scores, pointwise = pointwise, ppc_stats = ppc_stats,
              ppc_reps = ppc_reps, ppc_point = ppc_point, z_mode = z_mod)

    task_dir = joinpath(cfg.outdir, "tasks"); mkpath(task_dir)
    jldsave(joinpath(task_dir, cfg.config_id * ".jld2"); result = result)
    return result
end

# ============================================================================
# Constants
# ============================================================================

const N_SAMPLES = arg_value("--iterations", IS_TEST ? 1_000 : 300_000)
const BURN      = arg_value("--burn",       IS_TEST ?   200 : 100_000)
const THIN      = arg_value("--thin",       IS_TEST ?     5 :      20)
BURN < N_SAMPLES || error("--burn ($BURN) must be below --iterations ($N_SAMPLES)")

const N_TRACE   = 1_000      # trace rows exported per config
const N_REP     = IS_TEST ? 200 : 1_000   # posterior predictive replicates
const N_REP_OUT = IS_TEST ? 100 :   200   # replicate statistics exported

const N_PILOT    = 4_000
const PILOT_BURN = 500

# A new directory rather than dengue_sensitivity/, which holds the earlier
# 19-prior study and must not be overwritten by this one's tables.
const OUTDIR = joinpath(@__DIR__, "..", "results", "real_data",
                        IS_TEST ? "dengue_sensitivity_proposals_test" : "dengue_sensitivity_proposals")

# ============================================================================
# Load data — identical to run_dengue.jl, so the likelihood side is unchanged
# ============================================================================

const DATA_FILE = joinpath(@__DIR__, "..", "data", "cleaned", "dengue_uf$(UF)_$(YEAR).csv")
isfile(DATA_FILE) || error("""
    Missing $DATA_FILE
    Run: julia --project=. src/prepare_dengue_data.jl --uf $UF --year $YEAR""")

const DF    = CSV.read(DATA_FILE, DataFrame)
const Y_ALL = Int.(DF.y)
const E_ALL = Float64.(DF.pop) ./ 1e5          # exposure: population per 100k
const N     = nrow(DF)

const COORDS  = permutedims(hcat(Float64.(DF.lon), Float64.(DF.lat)))
const D_KM    = pairwise(Haversine(6371.0), COORDS)
const D_SCALE = median(D_KM[i, j] for i in 1:N for j in 1:N if i != j)

const ρ_BAR = sum(Y_ALL) / sum(E_ALL)
const ρ_A   = 1.5
const PRIORS = NegBinPopulationRatesPriors(ρ_a = ρ_A, ρ_b = ρ_A / ρ_BAR,
                                           r_a = 2.0, r_b = 0.5)

const ADJ_FILE = joinpath(@__DIR__, "..", "data", "cleaned", "dengue_uf$(UF)_adjacency.csv")
isfile(ADJ_FILE) || error("""
    Missing $ADJ_FILE
    Run: Rscript src/prepare_dengue_adjacency.R --uf $UF""")

const A_ADJ = load_adjacency(ADJ_FILE, Int.(DF.geocode))
const NBRS  = neighbour_lists(A_ADJ)
const H_HOP = hop_distances(NBRS)
any(isinf, H_HOP) && error("contiguity graph is disconnected; rebuild the edge list")

const D_GEODESIC = D_KM ./ D_SCALE

# ============================================================================
# The configuration grid
# ============================================================================

# (metric, s, infer_s). For hop, s is the window radius; for geodesic, a rate.
# The inferred-s chain starts at s = 1.
const PRIOR_GRID = IS_TEST ?
    [("hop", 1.0, false), ("geodesic", 5.0, false)] :
    [("hop", 1.0, false), ("hop", 2.0, false), ("hop", 3.0, false),
     ("geodesic", 1.0, false), ("geodesic", 5.0, false), ("geodesic", 10.0, false),
     ("geodesic", 1.0, true)]

const ALPHA_GRID = [true, false]            # infer α, or fix it at ALPHA_FIXED

# Coordinate design over (ρ family, r family): the lnmm/lnmm baseline, one
# parameter swapped to each alternative, and an all-prior reference. Each is
# crossed with fixed-dim ∈ {NoUpdate, Resample}.
const FAMILY_GRID = IS_TEST ?
    [(:lnmm, :lnmm), (:prior, :prior)] :
    [(:lnmm, :lnmm),
     (:prior, :lnmm), (:nmm, :lnmm), (:ig, :lnmm),
     (:lnmm, :prior), (:lnmm, :nmm), (:lnmm, :ig),
     (:prior, :prior)]
const RESAMPLE_GRID = [false, true]

fmt_s(s) = replace(@sprintf("%g", s), "." => "p", "-" => "m")

prior_id(metric, s, infer_s, infer_alpha) =
    (metric == "hop" ? "hop" : "geo") * "_s" * (infer_s ? "inf" : fmt_s(s)) *
    "__a" * (infer_alpha ? "inf" : "fix")

proposal_id(fρ, fr, resample) = "$(fρ)_$(fr)_" * (resample ? "rs" : "nu")

prior_label(metric, s, infer_s, infer_alpha) =
    (metric == "hop" ? "hop s=$(Int(s))" : infer_s ? "geodesic s inferred" :
                                               "geodesic s=$(@sprintf("%g", s))") *
    (infer_alpha ? ", α inferred" : ", α=$(@sprintf("%g", ALPHA_FIXED))")

proposal_label(fρ, fr, resample) = "ρ:$(fρ) r:$(fr) + " * (resample ? "Resample" : "NoUpdate")

# Grid specs are cheap NamedTuples; the 399×399 distance matrix is attached only
# when a config is about to run (`materialise`).
const GRID = [(metric = m, s = s, infer_s = is, infer_alpha = ia,
               fam_ρ = fρ, fam_r = fr, resample = rs)
              for (m, s, is) in PRIOR_GRID for ia in ALPHA_GRID
              for (fρ, fr) in FAMILY_GRID for rs in RESAMPLE_GRID]

config_id(g) = prior_id(g.metric, g.s, g.infer_s, g.infer_alpha) * "__" *
               proposal_id(g.fam_ρ, g.fam_r, g.resample)

function build_D(metric::String, s::Float64, infer_s::Bool)
    metric == "geodesic" && return (D_GEODESIC, D_SCALE, 0, s)
    metric == "hop" || error("unknown metric $metric")
    infer_s && error("s cannot be inferred for the hop window")
    r = Int(s)
    return (hop_window_D(H_HOP, r), 1.0, r, 1.0)
end

const METRIC_DESC = Dict(
    "geodesic" => "great-circle distance between centroids, exp(-s d / median distance)",
    "hop"      => "uniform link weight within s graph hops, suppressed beyond",
)

function materialise(g; n_samples = N_SAMPLES, burn = BURN, thin = THIN, cid = config_id(g),
                     outdir = OUTDIR)
    D, d_scale, r, rate = build_D(g.metric, g.s, g.infer_s)
    (
        config_id = cid,
        prior_id = prior_id(g.metric, g.s, g.infer_s, g.infer_alpha),
        proposal_id = proposal_id(g.fam_ρ, g.fam_r, g.resample),
        label = prior_label(g.metric, g.s, g.infer_s, g.infer_alpha) * " | " *
                proposal_label(g.fam_ρ, g.fam_r, g.resample),
        metric = g.metric, hop_radius = r, d_scale = d_scale,
        s = g.s, kernel_rate = rate, infer_s = g.infer_s,
        infer_alpha = g.infer_alpha, alpha_init = ALPHA_FIXED,
        fam_ρ = g.fam_ρ, fam_r = g.fam_r, resample = g.resample,
        σ_birth = SIGMA_BIRTH, σ_resample = SIGMA_RESAMPLE,
        y = Y_ALL, E = E_ALL, D = D, nbrs = NBRS, priors = PRIORS,
        n_samples = n_samples, burn = burn, thin = thin,
        n_trace = N_TRACE, n_rep = N_REP, n_rep_out = N_REP_OUT,
        outdir = outdir, seed = 42,
    )
end

function write_config_index()
    CSV.write(joinpath(OUTDIR, "config_index.csv"), DataFrame([
        (task = k, config_id = config_id(g),
         prior_id = prior_id(g.metric, g.s, g.infer_s, g.infer_alpha),
         proposal_id = proposal_id(g.fam_ρ, g.fam_r, g.resample),
         prior_label = prior_label(g.metric, g.s, g.infer_s, g.infer_alpha),
         proposal_label = proposal_label(g.fam_ρ, g.fam_r, g.resample),
         metric = g.metric, metric_description = METRIC_DESC[g.metric],
         s = g.s, infer_s = g.infer_s, infer_alpha = g.infer_alpha,
         alpha_fixed = g.infer_alpha ? NaN : ALPHA_FIXED,
         fam_rho = String(g.fam_ρ), fam_r = String(g.fam_r),
         fixed_dim = g.resample ? "Resample" : "NoUpdate",
         n_samples = N_SAMPLES, burn = BURN, thin = THIN)
        for (k, g) in enumerate(GRID)]))
end

# ============================================================================
# --list: the grid size, for the Slurm array range
# ============================================================================

if IS_LIST
    for (k, g) in enumerate(GRID)
        @printf("%4d  %s\n", k, config_id(g))
    end
    println("N_CONFIGS=$(length(GRID))")
    exit(0)
end

# ============================================================================
# --task N: one configuration, as one Slurm array task
# ============================================================================

if TASK_ID > 0
    1 <= TASK_ID <= length(GRID) ||
        error("--task $TASK_ID is outside the grid (1..$(length(GRID)))")
    # The Slurm submit script sets N_CONFIGS without running Julia; a stale value
    # would silently skip or misnumber configs, so refuse to run on a mismatch.
    if haskey(ENV, "N_CONFIGS") && parse(Int, ENV["N_CONFIGS"]) != length(GRID)
        error("N_CONFIGS=$(ENV["N_CONFIGS"]) from the submit script, but the grid has " *
              "$(length(GRID)) configs; update N_CONFIGS in submit_dengue_sensitivity.sh")
    end
    g = GRID[TASK_ID]
    cid = config_id(g)
    done = joinpath(OUTDIR, "tasks", cid * ".jld2")
    if isfile(done) && !FORCE
        println("Task $TASK_ID ($cid) already has a result at $done; pass --force to rerun.")
        exit(0)
    end
    cfg = materialise(g)
    @printf("Task %d/%d: %s\n  %s\n  %d iterations (burn %d, thin %d)\n",
            TASK_ID, length(GRID), cid, cfg.label, N_SAMPLES, BURN, THIN)
    flush(stdout)
    r = run_config(cfg)
    sm = r.summary
    @printf("Done in %.0f s: mean K %.2f, ESS(K) %.0f, ESS(logpost) %.0f, elpd_loo %.1f\n",
            sm.total_time, sm.mean_K, sm.ess_K, sm.ess_logpost, r.scores.elpd_loo)
    exit(0)
end

# ============================================================================
# --pilot: per-task cost at the corners of the grid
# ============================================================================

if IS_PILOT
    pilot = [(metric = "hop", s = 1.0, infer_s = false, infer_alpha = true,
              fam_ρ = :lnmm, fam_r = :lnmm, resample = rs) for rs in RESAMPLE_GRID]
    append!(pilot, [(metric = "geodesic", s = 1.0, infer_s = true, infer_alpha = true,
                     fam_ρ = :prior, fam_r = :prior, resample = rs) for rs in RESAMPLE_GRID])
    # Written under pilot/, so pilot chains never pass for grid results.
    cfgs = [materialise(g; n_samples = N_PILOT, burn = PILOT_BURN, thin = 1,
                        cid = "pilot_" * config_id(g), outdir = joinpath(OUTDIR, "pilot"))
            for g in pilot]
    println("\nPILOT — $(length(cfgs)) configs × $N_PILOT iterations")
    res = collect(pmap(run_config, cfgs))
    rows = DataFrame([r.summary for r in res])
    CSV.write(joinpath(OUTDIR, "summary_pilot.csv"), rows)
    @printf("\n%-44s %9s %10s\n", "config", "mean K", "s/1k it")
    for r in eachrow(rows)
        @printf("%-44s %9.2f %10.2f\n", r.config_id, r.mean_K, r.total_time / (N_PILOT / 1000))
    end
    worst = maximum(rows.total_time) / (N_PILOT / 1000)
    @printf("\nSlowest: %.1f s per 1k iterations, so about %.1f h per task at %d iterations.\n",
            worst, worst * N_SAMPLES / 1000 / 3600, N_SAMPLES)
    println("Cost rises with K and these chains are too short to reach it, so set the")
    println("array job's --time well above that.")
    exit(0)
end

# ============================================================================
# Local run: every config via pmap (the --test grid, or the full grid on a big
# machine). Results land in tasks/ exactly as array tasks write them.
# ============================================================================

if RUN_LOCAL
    todo = [g for g in GRID
            if FORCE || !isfile(joinpath(OUTDIR, "tasks", config_id(g) * ".jld2"))]
    @printf("\nDENGUE PRIOR × PROPOSAL STUDY — %d configs (%d to run) × %d iterations on %d workers\n",
            length(GRID), length(todo), N_SAMPLES, nworkers())
    pmap(run_config, (materialise(g) for g in todo))
end

# ============================================================================
# Collect: assemble the flat tables the R Markdown report reads
# ============================================================================

println("\nCollecting task results from $(joinpath(OUTDIR, "tasks"))...")
isdir(joinpath(OUTDIR, "tasks")) || error("no task results under $(joinpath(OUTDIR, "tasks"))")
write_config_index()

ids_all = [config_id(g) for g in GRID]
present = [isfile(joinpath(OUTDIR, "tasks", id * ".jld2")) for id in ids_all]
missing_ids = ids_all[.!present]
if !isempty(missing_ids)
    @warn "$(length(missing_ids)) of $(length(ids_all)) configs have no result yet; tables cover the rest" missing_ids
end
ids = ids_all[present]
isempty(ids) && error("no task results found under $(joinpath(OUTDIR, "tasks"))")
results = [load(joinpath(OUTDIR, "tasks", id * ".jld2"), "result") for id in ids]

summary_df = DataFrame([r.summary for r in results])
CSV.write(joinpath(OUTDIR, "summary_sensitivity.csv"), summary_df)

# One long table per per-draw quantity, config_id identifying the block.
stack_field(field, cols) =
    vcat([DataFrame(:config_id => ids[k],
                    (c => getfield(r, field)[c] for c in cols)...)
          for (k, r) in enumerate(results)]...)

CSV.write(joinpath(OUTDIR, "traces.csv"),
          stack_field(:traces, (:draw, :K, :logpost, :alpha_ddcrp, :s_ddcrp)))
CSV.write(joinpath(OUTDIR, "k_posterior.csv"), stack_field(:kpost, (:K, :prob)))
CSV.write(joinpath(OUTDIR, "ppc_stats.csv"),
          stack_field(:ppc_stats, (:statistic, :T_obs, :ppp, :mean, :sd,
                                   :q025, :q25, :q50, :q75, :q975)))
CSV.write(joinpath(OUTDIR, "ppc_replicates.csv"),
          stack_field(:ppc_reps, (:statistic, :rep, :T_rep)))

CSV.write(joinpath(OUTDIR, "model_comparison.csv"),
          DataFrame(:config_id => ids,
                    (c => [getfield(r.scores, c) for r in results]
                     for c in (:waic, :lppd, :p_waic, :lpml, :elpd_loo, :n_bad_k))...))

# Pointwise elpd: the report computes every difference and its standard error
# from this, for whichever subset of configs it compares.
CSV.write(joinpath(OUTDIR, "pointwise_scores.csv"),
    vcat([DataFrame(config_id = ids[k], geocode = Int.(DF.geocode),
                    elpd_waic_i = r.pointwise.elpd_waic_i, elpd_loo_i = r.pointwise.elpd_loo_i,
                    lcpo_i = r.pointwise.lcpo_i, k_hat = r.pointwise.k_hat)
          for (k, r) in enumerate(results)]...))

CSV.write(joinpath(OUTDIR, "cluster_assignments_sensitivity.csv"),
    vcat([DataFrame(config_id = ids[k], geocode = Int.(DF.geocode), name = String.(DF.name),
                    lat = Float64.(DF.lat), lon = Float64.(DF.lon),
                    y = Y_ALL, pop = E_ALL .* 1e5, incidence = Y_ALL ./ E_ALL,
                    cluster_mode = r.assign.cluster_mode,
                    cluster_dahl = r.assign.cluster_dahl,
                    cocluster_confidence = r.assign.cocluster_confidence,
                    rho_mean = r.assign.rho_mean, r_mean = r.assign.r_mean)
          for (k, r) in enumerate(results)]...))

CSV.write(joinpath(OUTDIR, "ppc_pointwise.csv"),
    vcat([DataFrame(config_id = ids[k], geocode = Int.(DF.geocode),
                    y_obs = r.ppc_point.y_obs, yrep_mean = r.ppc_point.yrep_mean,
                    yrep_q025 = r.ppc_point.yrep_q025, yrep_q50 = r.ppc_point.yrep_q50,
                    yrep_q975 = r.ppc_point.yrep_q975, pit = r.ppc_point.pit)
          for (k, r) in enumerate(results)]...))

# Agreement between mode partitions, for every pair of configs: within a prior
# it measures Monte Carlo agreement between proposals, across priors how much
# of the partition the prior decides.
CSV.write(joinpath(OUTDIR, "cross_config_ari.csv"),
    DataFrame(config_a = repeat(ids, inner = length(ids)),
              config_b = repeat(ids, outer = length(ids)),
              ari = vec([ari(results[i].z_mode, results[j].z_mode)
                         for j in eachindex(results), i in eachindex(results)])))

# What each prior allows, for the report's metric section.
let rows = NamedTuple[]
    for (m, s, is) in PRIOR_GRID
        is && continue
        D, scale, r, rate = build_D(m, s, is)
        W = exp.(-rate .* D)
        off_w = [W[i, j] for i in 1:N for j in 1:N if i != j]
        inwin = r > 0 ? mean(H_HOP[i, j] <= r for i in 1:N for j in 1:N if i != j) : NaN
        push!(rows, (metric = m, s = s, description = METRIC_DESC[m], hop_radius = r,
                     d_scale = scale, mean_link_weight = mean(off_w),
                     mean_row_weight = mean(sum(W, dims = 2) .- 1.0),
                     frac_in_window = inwin,
                     mean_degree = sum(A_ADJ) / N, n_edges = Int(sum(A_ADJ) / 2),
                     graph_diameter = Int(maximum(H_HOP))))
    end
    CSV.write(joinpath(OUTDIR, "metric_summary.csv"), DataFrame(rows))
end

let
    Random.seed!(7)
    pairs = [(i, j) for i in 1:N for j in (i+1):N]
    sel = length(pairs) <= 20_000 ? pairs : pairs[randperm(length(pairs))[1:20_000]]
    CSV.write(joinpath(OUTDIR, "metric_pairs.csv"),
        DataFrame(i = first.(sel), j = last.(sel),
                  geodesic_km = [D_KM[p...] for p in sel],
                  geodesic    = [D_GEODESIC[p...] for p in sel],
                  hop         = [H_HOP[p...] for p in sel]))
end

jldsave(joinpath(OUTDIR, "run_metadata_sensitivity.jld2");
        uf = UF, year = YEAR, y = Y_ALL, E = E_ALL,
        lat = Float64.(DF.lat), lon = Float64.(DF.lon),
        geocode = Int.(DF.geocode), name = String.(DF.name),
        d_scale_km = D_SCALE, adjacency = A_ADJ, hop = H_HOP,
        log_floor = LOG_FLOOR, alpha_fixed = ALPHA_FIXED, config_ids = ids)

# ============================================================================
# Console summary: the best-mixing proposal per prior × α cell, then those
# cells ranked by elpd_loo with standard errors against the best
# ============================================================================

best_rows = combine(groupby(summary_df, :prior_id)) do sdf
    sdf[argmax(sdf.min_ess), :]
end
best_ids = String.(best_rows.config_id)
res_of = Dict(zip(ids, results))
loo = [res_of[id].scores.elpd_loo for id in best_ids]
order = sortperm(loo; rev = true)
top = best_ids[order[1]]

println("\n" * "="^118)
println("BEST-MIXING PROPOSAL PER PRIOR × α (by min(ESS(K), ESS(logpost))), ranked by elpd_loo")
println("="^118)
@printf("%-18s %-26s %7s %7s %8s %6s %9s %8s %6s %6s\n",
        "prior", "proposal", "mean K", "ESS(K)", "ESS(lp)", "acc.D%", "elpd_loo", "shortfall", "se", "in se")
println("-"^118)
for k in order
    id = best_ids[k]; r = res_of[id]; sm = r.summary
    if id == top
        @printf("%-18s %-26s %7.1f %7.0f %8.0f %6.2f %9.1f %8s %6s %6s\n",
                sm.prior_id, sm.proposal_id, sm.mean_K, sm.ess_K, sm.ess_logpost,
                100 * sm.acc_death, r.scores.elpd_loo, "best", "", "")
    else
        d = elpd_difference(res_of[top].pointwise.elpd_loo_i, r.pointwise.elpd_loo_i)
        @printf("%-18s %-26s %7.1f %7.0f %8.0f %6.2f %9.1f %8.1f %6.1f %6.1f\n",
                sm.prior_id, sm.proposal_id, sm.mean_K, sm.ess_K, sm.ess_logpost,
                100 * sm.acc_death, r.scores.elpd_loo, d.diff, d.se, d.diff / d.se)
    end
end

println("\nReading this table:")
println("  geodesic s is a RATE (larger = more local); hop s is a window RADIUS (larger = less local).")
println("  Proposals do not change the posterior, so within a prior × α cell they should agree.")
println("  A shortfall under ~2 se is no real preference between priors.")

println("\nAll tables in $OUTDIR")
println("Report: Rscript -e 'rmarkdown::render(\"src/dengue_sensitivity_analysis.rmd\"" *
        (IS_TEST ? ", params = list(results_dir = \"$(normpath(OUTDIR))\")" : "") * ")'")
