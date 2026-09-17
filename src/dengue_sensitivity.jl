# ============================================================================
# dengue_sensitivity.jl
#
# Prior-sensitivity study for the Brazil dengue application: how much of the
# fitted clustering is a property of the DATA, and how much of the ddCRP prior
# we chose to put over it?
#
# Two axes are varied, with the likelihood and the sampler held completely
# fixed, so every difference between configurations is attributable to the
# ddCRP prior alone:
#
#   1. The decay scale s, over seven orders of magnitude, from the flat limit
#      (a distance-independent CRP) to the strongly-local limit.
#   2. The distance metric itself. Municipalities are areal units, so the
#      great-circle distance between centroids is only one reasonable choice.
#      The graph analogues are included: the matrix exponential of the
#      contiguity adjacency matrix, and hop-distance windows that allow links
#      only within s steps.
#
# FOR THE HOP METRIC, s IS THE WINDOW RADIUS IN HOPS, not a rate. The kernel is
# uniform inside the window and effectively zero outside it, so the radius is
# the only thing the prior has to say; a within-window decay rate would be
# almost entirely absorbed by the inferred α_ddcrp. The package still needs a
# rate for exp(-rate*d), so hop configurations pass rate 1 on a 0 / LOG_FLOOR
# distance matrix (see build_D). Everywhere below `s` is the reported value and
# `kernel_rate` is what reaches DDCRPParams; they coincide for every other
# metric.
#
# THE DECAY SCALE s IS A RATE, NOT A LENGTH. The package defines
#
#     exp_decay(d; scale) = exp(-d * scale)            (priors.jl:55)
#
# so the link weight is exp(-d*s): LARGER s means FASTER decay and a MORE
# LOCAL prior. The decay length in kilometres is D_SCALE / s, not D_SCALE * s.
# Every reading of this study depends on getting that direction right.
#
# WHY THE ALTERNATIVE METRICS ARE DISTANCE MATRICES, NOT CUSTOM KERNELS.
# `DDCRPParams` accepts a `decay_fn`, but the hyperparameter samplers in
# `inference/hyperparams.jl` hard-code exp(-s*d_ij) in `compute_R` and in the
# α and s updates regardless of it. These runs infer α_ddcrp, so a custom
# decay_fn would leave the α update inconsistent with the prior actually in
# use. Every metric is therefore encoded as a distance matrix and the stock
# `exp_decay` is kept throughout.
#
# Run `src/prepare_dengue_data.jl` and `src/prepare_dengue_adjacency.R` first.
#
# Usage:
#   julia --project=. src/dengue_sensitivity.jl --pilot   # cost, before committing
#   julia --project=. src/dengue_sensitivity.jl --test    # tiny/fast smoke run
#   julia --project=. src/dengue_sensitivity.jl           # full grid, 8 workers
#   julia --project=. src/dengue_sensitivity.jl --workers 12
#   julia --project=. src/dengue_sensitivity.jl --slurm
#   --iterations / --burn / --thin override the chain length below.
#
# Then: Rscript -e 'rmarkdown::render("src/dengue_sensitivity_analysis.rmd")'
# ============================================================================

using DistanceDependentCRP, CSV, DataFrames, Distances, Statistics, StatsBase,
      Random, JLD2, Printf, Distributions, LinearAlgebra
using Distributed

const USE_SLURM = "--slurm" in ARGS
const IS_TEST   = "--test"  in ARGS
const IS_PILOT  = "--pilot" in ARGS

function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) ? default : parse(Int, ARGS[i+1])
end

const UF   = arg_value("--uf", 41)      # 41 = Paraná (see prepare_dengue_data.jl)
const YEAR = arg_value("--year", 2024)

if USE_SLURM
    using SlurmClusterManager
    addprocs(SlurmManager())
else
    addprocs(arg_value("--workers", 8))
end

@everywhere begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using DistanceDependentCRP, Statistics, StatsBase, Distributions, JLD2, Random
    include(joinpath(@__DIR__, "negbin_population_model.jl"))
    include(joinpath(@__DIR__, "model_comparison.jl"))
    include(joinpath(@__DIR__, "dengue_common.jl"))
end

# ============================================================================
# Sampler settings, held fixed across every configuration
#
# These are the Part 2 winners from `run_dengue.jl` (read off
# results/real_data/dengue/summary_sensitivity.csv, which records the σ pair the
# tuning stage selected). Nothing about the sampler varies in this study; only
# the ddCRP prior does.
# ============================================================================

@everywhere const PROP_SDS = (ρ = 0.15, r = 0.30)   # within-cluster MH, log scale
const SIGMA_BIRTH    = 0.50
const SIGMA_RESAMPLE = 0.10
const FAM_ρ = :lnmm
const FAM_r = :lnmm

@everywhere _birth_family(fam::Symbol, σ) =
    fam === :prior ? PriorProposal() :
    fam === :nmm   ? NormalMomentMatch(σ) :
    fam === :lnmm  ? LogNormalMomentMatch(σ) :
    fam === :ig    ? InverseGammaMomentMatch() :
    error("unknown family $fam")

@everywhere make_birth(fρ, fr, σ) =
    MixedProposal((ρ = _birth_family(fρ, σ), r = _birth_family(fr, σ)))

@everywhere make_fdim(fρ, fr, σr) =
    MixedFixedDim((ρ = Resample(_birth_family(fρ, σr)),
                   r = Resample(_birth_family(fr, σr))))

@everywhere thin_chain(s::NegBinPopulationRatesSamples, keep) =
    NegBinPopulationRatesSamples(s.c[keep, :], s.ρ[keep, :], s.r[keep, :],
                                 s.logpost[keep], s.α_ddcrp[keep], s.s_ddcrp[keep])

# Point estimates (`dahl_point_estimate`, `mode_point_estimate`, `ari`) and the
# posterior predictive statistics (`moran_I`, `ppc_statistics`) live in
# dengue_common.jl, shared with run_dengue.jl and analyse_dengue.jl, so the
# whole study still runs on the workers in one pass.

# ============================================================================
# Unit of work
#
# One configuration: run the chain, save it, and reduce it on the worker to the
# handful of small tables the report needs. Reducing here rather than in a
# second script means the 19 chains are opened once, not twice.
# ============================================================================

@everywhere function run_config(cfg)
    Random.seed!(cfg.seed)
    data = CountDataWithPopulation(cfg.y, cfg.E, cfg.D)

    opts = MCMCOptions(
        n_samples = cfg.n_samples,
        verbose = false,
        track_diagnostics = true,
        infer_params = Dict{Symbol,Bool}(:α_ddcrp => true, :s_ddcrp => cfg.infer_s),
        prop_sds = Dict{Symbol,Float64}(
            :ρ => PROP_SDS.ρ, :r => PROP_SDS.r, :s_ddcrp => 0.3),
    )

    ddcrp  = DDCRPParams(1.0, cfg.kernel_rate, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
    model  = NegBinPopulationRates()
    birth  = make_birth(cfg.fam_ρ, cfg.fam_r, cfg.σ_birth)
    fdim   = make_fdim(cfg.fam_ρ, cfg.fam_r, cfg.σ_resample)

    total_time = @elapsed begin
        samples, diag = mcmc(model, data, ddcrp, cfg.priors, birth;
                             fixed_dim_proposal = fdim, opts = opts)
    end

    keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
    post = thin_chain(samples, keep)
    n_post = length(keep)

    chain_dir = joinpath(cfg.outdir, "chains"); mkpath(chain_dir)
    jldsave(joinpath(chain_dir, cfg.chain_file);
            samples = post, burn = 0, thin = 1,
            burn_applied = cfg.burn, thin_applied = cfg.thin,
            label = cfg.label, config_id = cfg.config_id)

    # ---- mixing diagnostics -------------------------------------------------
    k_trace = calculate_n_clusters(post.c)
    acc     = acceptance_rates(diag)
    ess_K   = effective_sample_size(Float64.(k_trace))
    esjd_K  = mean(diff(Float64.(k_trace)).^2)
    med_ess(M) = median([effective_sample_size(M[:, i]) for i in axes(M, 2)])

    summary = (
        config_id = cfg.config_id, label = cfg.label,
        metric = cfg.metric, metric_family = cfg.metric_family,
        hop_radius = cfg.hop_radius, s_decay = cfg.s, kernel_rate = cfg.kernel_rate,
        infer_s = cfg.infer_s,
        d_scale = cfg.d_scale,
        sigma_birth = cfg.σ_birth, sigma_resample = cfg.σ_resample,
        n_obs = length(cfg.y), n_post = n_post,
        mean_K = mean(k_trace), mode_K = Int(mode(k_trace)),
        std_K = std(Float64.(k_trace)),
        min_K = minimum(k_trace), max_K = maximum(k_trace),
        acc_birth = acc.birth, acc_death = acc.death,
        acc_fixed = acc.fixed, acc_overall = acc.overall,
        ess_K = ess_K, ess_logpost = effective_sample_size(post.logpost),
        med_ess_rho = med_ess(post.ρ), med_ess_r = med_ess(post.r),
        esjd_K = esjd_K, ess_per_sec = ess_K / total_time,
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

    return (summary = summary, traces = traces, kpost = kpost, assign = assign,
            scores = scores, pointwise = pointwise, ppc_stats = ppc_stats, ppc_reps = ppc_reps,
            ppc_point = ppc_point, z_mode = z_mod)
end

# ============================================================================
# Constants
# ============================================================================

# Full length matches the 150k-iteration headline dengue chains, so the
# sensitivity configurations are compared at the same Monte Carlo precision.
const N_SAMPLES = arg_value("--iterations", IS_TEST ? 1_000 : 150_000)
const BURN      = arg_value("--burn",       IS_TEST ?   200 :  50_000)
const THIN      = arg_value("--thin",       IS_TEST ?     5 :      10)
BURN < N_SAMPLES || error("--burn ($BURN) must be below --iterations ($N_SAMPLES)")

const N_TRACE   = 2_000      # trace rows exported per config
const N_REP     = IS_TEST ? 200 : 1_000   # posterior predictive replicates
const N_REP_OUT = IS_TEST ? 100 :   500   # replicate statistics exported

const N_PILOT   = 4_000
const PILOT_BURN = 500

const OUTDIR = IS_TEST ?
    joinpath(@__DIR__, "..", "results", "real_data", "dengue_sensitivity_test") :
    joinpath(@__DIR__, "..", "results", "real_data", "dengue_sensitivity")
mkpath(OUTDIR)

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

@printf("Loaded %s: %d municipalities, %d total cases, %.2f M people\n",
        basename(DATA_FILE), N, sum(Y_ALL), sum(DF.pop) / 1e6)
@printf("  median pairwise centroid distance %.0f km (every D is rescaled by its own median)\n",
        D_SCALE)

# ============================================================================
# The contiguity graph
# ============================================================================

const ADJ_FILE = joinpath(@__DIR__, "..", "data", "cleaned", "dengue_uf$(UF)_adjacency.csv")
isfile(ADJ_FILE) || error("""
    Missing $ADJ_FILE
    Run: Rscript src/prepare_dengue_adjacency.R --uf $UF""")

const A_ADJ = load_adjacency(ADJ_FILE, Int.(DF.geocode))
const NBRS  = neighbour_lists(A_ADJ)

const H_HOP = hop_distances(NBRS)
any(isinf, H_HOP) && error("contiguity graph is disconnected; rebuild the edge list")

# Communicability G = exp(A). The spectral radius of a planar contiguity graph
# is small (about 6 here), so the entries stay well inside Float64 range; the
# far tail underflows towards zero and is floored below.
const G_COMM = exp(Matrix(A_ADJ))

@printf("  contiguity graph: %d edges, mean degree %.2f, diameter %d hops\n",
        Int(sum(A_ADJ) / 2), sum(A_ADJ) / N, Int(maximum(H_HOP)))

# ============================================================================
# Distance metrics
#
# Every metric is rescaled by its own median off-diagonal entry (the great-
# circle distances by the median pairwise distance), so a given s means the
# same "decay per median separation" under all of them.
#
# The hop windows use a soft floor rather than a hard zero; see `hop_window_D`
# and LOG_FLOOR in dengue_common.jl.
# ============================================================================

rescale(D) = (m = median(D[i, j] for i in 1:N for j in 1:N if i != j); (D ./ m, m))

const D_GEODESIC = rescale(D_KM)

# Communicability distance ξ_ij = sqrt(G_ii + G_jj - 2 G_ij) (Estrada & Hatano).
# A genuine metric, so it drops straight into the exp(-s*d) kernel.
const D_COMMDIST = rescale([sqrt(max(G_COMM[i,i] + G_COMM[j,j] - 2G_COMM[i,j], 0.0))
                            for i in 1:N, j in 1:N])

# Log-communicability: the most direct reading of "use exp(A) as the kernel",
# since exp(-s*d) then makes the link weight proportional to G_ij^s. G is
# floored at eps()*max(G) so the far tail cannot produce Inf.
const D_LOGCOMM = let
    off = [G_COMM[i,j] for i in 1:N for j in 1:N if i != j]
    gmax = maximum(off); floor_g = eps() * gmax
    rescale([i == j ? 0.0 : -log(max(G_COMM[i,j], floor_g) / gmax) for i in 1:N, j in 1:N])
end

# Hop distance rescaled by its median; descriptive only (metric_pairs.csv), since
# the hop windows use raw hop counts.
const D_HOP = rescale(H_HOP)

# The rate passed to DDCRPParams. The hop windows encode everything in D, so
# they run at a fixed rate of 1; every other metric uses s itself.
kernel_rate(metric, s) = metric == "hop" ? 1.0 : s

"""
    build_D(metric, s) -> (D, d_scale, hop_radius)

The distance matrix for one configuration. For `hop`, `s` is the integer
window radius: pairs within `s` hops get distance 0 and the rest LOG_FLOOR, so
under kernel rate 1 the in-window weights are uniform. The other metrics ignore
`s` here, since it enters as the kernel rate instead.
"""
function build_D(metric::String, s::Float64)
    metric == "geodesic"  && return (D_GEODESIC[1], D_GEODESIC[2], 0)
    metric == "comm_dist" && return (D_COMMDIST[1], D_COMMDIST[2], 0)
    metric == "log_comm"  && return (D_LOGCOMM[1],  D_LOGCOMM[2],  0)
    metric == "hop" || error("unknown metric $metric")
    (isinteger(s) && s >= 1) || error("hop window radius must be a positive integer, got $s")
    r = Int(s)
    return (hop_window_D(H_HOP, r), 1.0, r)
end

const METRIC_DESC = Dict(
    "geodesic"  => "great-circle distance between municipality centroids",
    "comm_dist" => "communicability distance sqrt(G_ii + G_jj - 2 G_ij), G = exp(A)",
    "log_comm"  => "-log(G_ij / max G), G = exp(A); link weight proportional to G_ij^s",
    "hop"       => "uniform link weight within s graph hops, suppressed beyond",
)

# ============================================================================
# The configuration grid
#
# Axis 1: the requested seven-point s grid under the great-circle metric,
#         spanning the flat limit (s = 1e-4, where exp(-d*s) is 1 to within
#         1e-4 for every pair and the ddCRP degenerates into a
#         distance-independent CRP) through to the strongly local limit
#         (s = 10, a decay length of about 23 km).
# Axis 2: each communicability metric at three points drawn from that same
#         grid, so the two axes stay comparable.
# Axis 3: the hop window at radius s = 1..5 hops.
# Plus one inferred-s chain under the great-circle metric, as the reference
# for the degeneracy run_dengue.jl documents.
# ============================================================================

const S_GRID     = IS_TEST ? [0.01, 0.5] : [0.0001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
const S_GRID_ALT = IS_TEST ? [0.5]       : [0.1, 0.5, 5.0]
const ALT_METRICS = IS_TEST ? ["comm_dist"] : ["comm_dist", "log_comm"]
const HOP_GRID    = IS_TEST ? [1.0, 3.0] : [1.0, 2.0, 3.0, 4.0, 5.0]

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")
fmt_s(s) = replace(@sprintf("%g", s), "." => "p", "-" => "m")

function make_cfg(metric, s; infer_s = false, n_samples = N_SAMPLES, burn = BURN,
                  config_id = nothing)
    D, d_scale, r = build_D(metric, s)
    lbl = infer_s ? "$(metric) s inferred" : "$(metric) s=$(@sprintf("%g", s))"
    cid = something(config_id,
                    infer_s ? "$(metric)__infer_s" : "$(metric)__s$(fmt_s(s))")
    (
        config_id = cid, label = lbl,
        metric = metric,
        metric_family = metric,
        hop_radius = r, d_scale = d_scale,
        s = s, kernel_rate = kernel_rate(metric, s), infer_s = infer_s,
        fam_ρ = FAM_ρ, fam_r = FAM_r,
        σ_birth = SIGMA_BIRTH, σ_resample = SIGMA_RESAMPLE,
        y = Y_ALL, E = E_ALL, D = D, nbrs = NBRS, priors = PRIORS,
        n_samples = n_samples, burn = burn, thin = THIN,
        n_trace = N_TRACE, n_rep = N_REP, n_rep_out = N_REP_OUT,
        outdir = OUTDIR, chain_file = fmtlabel(cid) * ".jld2",
        seed = 42,
    )
end

# ============================================================================
# Part 0 — --pilot: what does the full grid cost?
#
# One short chain per metric family, since cost depends on the metric only
# through how many clusters the prior encourages. Run this before committing to
# the full grid.
# ============================================================================

if IS_PILOT
    pilot = [("geodesic", 0.5), ("hop", 1.0), ("hop", 5.0),
             ("comm_dist", 0.5), ("log_comm", 0.5)]
    cfgs = [make_cfg(m, s; n_samples = N_PILOT, burn = PILOT_BURN,
                     config_id = "pilot_$(m)_s$(fmt_s(s))") for (m, s) in pilot]
    println("\n" * "="^80)
    println("PILOT — $(length(cfgs)) configs × $N_PILOT iterations")
    println("="^80)

    res = collect(pmap(run_config, cfgs))
    rows = DataFrame([r.summary for r in res])
    CSV.write(joinpath(OUTDIR, "summary_pilot.csv"), rows)

    n_full = length(S_GRID) + length(ALT_METRICS) * length(S_GRID_ALT) + length(HOP_GRID) + 1
    n_work = nworkers()
    println()
    @printf("%-14s %9s %9s %9s %9s\n", "metric", "mean K", "time(s)", "s/1k it", "ESS(K)")
    println("-"^60)
    for r in eachrow(rows)
        @printf("%-14s %9.2f %9.1f %9.2f %9.0f\n",
                "$(r.metric) s=$(@sprintf("%g", r.s_decay))", r.mean_K, r.total_time, r.total_time / (N_PILOT / 1000), r.ess_K)
    end
    per1k = mean(rows.total_time) / (N_PILOT / 1000)
    # Cost grows with K, and the pilot runs far too short to reach the
    # stationary K, so treat this as a floor rather than a forecast.
    est = per1k * (N_SAMPLES / 1000) * n_full / n_work
    println()
    @printf("Full grid: %d configs × %d iterations on %d workers\n", n_full, N_SAMPLES, n_work)
    @printf("Projected wall time: at least %.1f h (%.1f s per 1k iterations, per chain)\n",
            est / 3600, per1k)
    println("Cost rises with K, and these pilot chains are too short to reach the")
    println("stationary K, so read that as a lower bound.")
    println("\n  → $(joinpath(OUTDIR, "summary_pilot.csv"))")
    exit(0)
end

# ============================================================================
# The full grid
# ============================================================================

configs = Any[make_cfg("geodesic", s) for s in S_GRID]
for m in ALT_METRICS, s in S_GRID_ALT
    push!(configs, make_cfg(m, s))
end
append!(configs, [make_cfg("hop", s) for s in HOP_GRID])
push!(configs, make_cfg("geodesic", 0.5; infer_s = true))

println("\n" * "="^80)
@printf("DENGUE PRIOR SENSITIVITY — %d configs × %d iterations on %d workers\n",
        length(configs), N_SAMPLES, nworkers())
println("="^80)
for c in configs
    @printf("  %-26s %-11s s=%-8g %s\n",
            c.config_id, c.metric, c.s,
            c.metric == "geodesic" ? @sprintf("decay length %.0f km", c.d_scale / c.s) :
            c.metric == "hop" ? @sprintf("uniform window of %d hop%s", c.hop_radius, c.hop_radius == 1 ? "" : "s") :
            @sprintf("decay length %.2f metric units", 1 / c.s))
end

results = collect(pmap(run_config, configs))

# ============================================================================
# Assemble the flat tables the R Markdown report reads
# ============================================================================

println("\nWriting tables...")
ids = [c.config_id for c in configs]

CSV.write(joinpath(OUTDIR, "config_index.csv"),
    DataFrame(config_id = ids, label = [c.label for c in configs],
              metric = [c.metric for c in configs],
              metric_family = [c.metric_family for c in configs],
              metric_description = [METRIC_DESC[c.metric] for c in configs],
              hop_radius = [c.hop_radius for c in configs],
              s = [c.s for c in configs], kernel_rate = [c.kernel_rate for c in configs],
              infer_s = [c.infer_s for c in configs],
              d_scale = [c.d_scale for c in configs],
              n_samples = [c.n_samples for c in configs],
              burn = [c.burn for c in configs], thin = [c.thin for c in configs],
              chain_file = [c.chain_file for c in configs]))

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

# Pointwise elpd, and every pairwise difference with its standard error.
CSV.write(joinpath(OUTDIR, "pointwise_scores.csv"),
    vcat([DataFrame(config_id = ids[k], geocode = Int.(DF.geocode),
                    elpd_waic_i = r.pointwise.elpd_waic_i, elpd_loo_i = r.pointwise.elpd_loo_i,
                    lcpo_i = r.pointwise.lcpo_i, k_hat = r.pointwise.k_hat)
          for (k, r) in enumerate(results)]...))

let rows = NamedTuple[]
    for a in eachindex(results), b in eachindex(results)
        a == b && continue
        for (crit, f) in (("elpd_loo", :elpd_loo_i), ("elpd_waic", :elpd_waic_i), ("lpml", :lcpo_i))
            d = elpd_difference(getfield(results[a].pointwise, f), getfield(results[b].pointwise, f))
            push!(rows, (config_a = ids[a], config_b = ids[b], criterion = crit,
                         diff = d.diff, se = d.se, z = d.diff / d.se))
        end
    end
    CSV.write(joinpath(OUTDIR, "elpd_differences.csv"), DataFrame(rows))
end

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

# Cross-configuration agreement on the mode partitions: the question of whether
# the clustering is a property of the data or of the prior.
CSV.write(joinpath(OUTDIR, "cross_config_ari.csv"),
    DataFrame(config_a = repeat(ids, inner = length(ids)),
              config_b = repeat(ids, outer = length(ids)),
              ari = vec([ari(results[i].z_mode, results[j].z_mode)
                         for j in eachindex(results), i in eachindex(results)])))

# ============================================================================
# Metric descriptions, for the report's section on what each prior does
# ============================================================================

let
    rows = NamedTuple[]
    for (m, s) in vcat(("geodesic", 0.5), [(a, 0.5) for a in ALT_METRICS],
                       [("hop", h) for h in HOP_GRID])
        D, scale, r = build_D(m, s)
        off = [D[i, j] for i in 1:N for j in 1:N if i != j]
        inwin = r > 0 ? mean(H_HOP[i, j] <= r for i in 1:N for j in 1:N if i != j) : 1.0
        push!(rows, (metric = m, s = s, description = METRIC_DESC[m], hop_radius = r,
                     d_scale = scale, min_off = minimum(off), median_off = median(off),
                     max_off = maximum(off), frac_in_window = inwin,
                     mean_degree = sum(A_ADJ) / N, n_edges = Int(sum(A_ADJ) / 2),
                     graph_diameter = Int(maximum(H_HOP))))
    end
    CSV.write(joinpath(OUTDIR, "metric_summary.csv"), DataFrame(rows))
end

# A sample of pairs, so the report can show how the metrics relate to one
# another without shipping four 399 x 399 matrices.
let
    Random.seed!(7)
    pairs = [(i, j) for i in 1:N for j in (i+1):N]
    sel = length(pairs) <= 20_000 ? pairs : pairs[randperm(length(pairs))[1:20_000]]
    CSV.write(joinpath(OUTDIR, "metric_pairs.csv"),
        DataFrame(i = first.(sel), j = last.(sel),
                  geodesic_km = [D_KM[p...] for p in sel],
                  geodesic    = [D_GEODESIC[1][p...] for p in sel],
                  hop         = [H_HOP[p...] for p in sel],
                  hop_scaled  = [D_HOP[1][p...] for p in sel],
                  comm_dist   = [D_COMMDIST[1][p...] for p in sel],
                  log_comm    = [D_LOGCOMM[1][p...] for p in sel]))
end

jldsave(joinpath(OUTDIR, "run_metadata_sensitivity.jld2");
        uf = UF, year = YEAR, y = Y_ALL, E = E_ALL,
        lat = Float64.(DF.lat), lon = Float64.(DF.lon),
        geocode = Int.(DF.geocode), name = String.(DF.name),
        d_scale_km = D_SCALE, adjacency = A_ADJ, hop = H_HOP,
        log_floor = LOG_FLOOR, config_ids = ids,
        chain_files = [c.chain_file for c in configs])

# ============================================================================
# Console summary
# ============================================================================

println("\n" * "="^112)
println("PRIOR SENSITIVITY — $(length(configs)) configurations")
println("="^112)
@printf("%-26s %8s %8s %8s %9s %9s %8s %8s %9s\n",
        "config", "mean K", "mode K", "sd K", "α post", "s post",
        "acc.D%", "ESS(K)", "WAIC")
println("-"^112)
for (k, r) in enumerate(results)
    sm = r.summary
    @printf("%-26s %8.2f %8d %8.2f %9.3f %9.4f %8.2f %8.0f %9.1f\n",
            sm.config_id, sm.mean_K, sm.mode_K, sm.std_K, sm.alpha_post, sm.s_post,
            sm.acc_death * 100, sm.ess_K, results[k].scores.waic)
end

let loo = [r.scores.elpd_loo for r in results]
    best = argmax(loo)
    println("\nelpd_loo shortfall against the best ($(ids[best])); under ~2 se is no real preference:")
    for k in sortperm(loo; rev = true)
        k == best && continue
        d = elpd_difference(results[best].pointwise.elpd_loo_i, results[k].pointwise.elpd_loo_i)
        @printf("  %-26s shortfall %8.1f  se %6.1f  (%.1f se)\n", ids[k], d.diff, d.se, d.diff / d.se)
    end
end

println("\nReading this table:")
println("  s is a RATE: weight = exp(-d*s), so larger s is a MORE LOCAL prior.")
println("  Except for hop, where s is the window radius: larger s is a LESS local prior.")
println("  At s = 1e-4 the kernel is flat and the ddCRP is a plain CRP; if K there")
println("  differs sharply from K at s = 0.5, the spatial prior is doing real work.")
println("  High cross-config ARI means the clustering is driven by the data, not the prior.")

println("\nAll tables in $OUTDIR")
println("Report: Rscript -e 'rmarkdown::render(\"src/dengue_sensitivity_analysis.rmd\"" *
        (IS_TEST ? ", params = list(results_dir = \"$(normpath(OUTDIR))\")" : "") * ")'")
