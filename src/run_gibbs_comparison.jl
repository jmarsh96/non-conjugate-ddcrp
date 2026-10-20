# Auxiliary Gibbs (Supplement S1) against reversible jump on the same posterior, from four
# shared starts and with equal wall-clock per chain (set by the pilot in pilot_timing.csv).
#
#   julia --project=. src/run_gibbs_comparison.jl --study {skewnormal|poisson|dengue|old_faithful}
#         [--pilot] [--test] [--slurm] [--workers N]

using CSV, DataFrames, Distances, Statistics, StatsBase, Random, JLD2, Printf, Distributions
using Distributed

function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) ? default : ARGS[i+1]
end
const STUDY = arg_value("--study", "")
STUDY in ("skewnormal", "poisson", "dengue", "old_faithful") ||
    error("--study must be one of skewnormal, poisson, dengue, old_faithful")
const USE_SLURM = "--slurm" in ARGS
const IS_TEST = "--test" in ARGS
const FORCE_PILOT = "--pilot" in ARGS

if USE_SLURM
    using SlurmClusterManager
    # The 60s default is too short for a wide allocation.
    addprocs(SlurmManager(launch_timeout = 300.0))
else
    addprocs(parse(Int, arg_value("--workers", "8")))
end

const SRC = @__DIR__
@everywhere begin
    using Pkg; Pkg.activate(joinpath($SRC, ".."))
    using DistanceDependentCRP, CSV, DataFrames, Distances, Statistics, StatsBase,
          Distributions, JLD2, Random, LinearAlgebra
    BLAS.set_num_threads(1)
end
if STUDY == "skewnormal"
    @everywhere include($(joinpath(SRC, "skewnormal_model.jl")))
    @everywhere include($(joinpath(SRC, "skewnormal_common.jl")))
elseif STUDY == "poisson"
    @everywhere include($(joinpath(SRC, "poisson_common.jl")))
elseif STUDY == "dengue"
    @everywhere include($(joinpath(SRC, "negbin_population_model.jl")))
    @everywhere include($(joinpath(SRC, "dengue_common.jl")))
else
    @everywhere include($(joinpath(SRC, "old_faithful_common.jl")))
end

# Settings

const M_POOL = [1, 3, 10, 30]
const STARTS = [:singletons, :one_cluster, :prior_a, :prior_b]
const START_SEEDS = Dict(:singletons => 1001, :one_cluster => 1002, :prior_a => 1003, :prior_b => 1004)

# Wall-clock budget per chain (s); N_MAX bounds memory
const BUDGET = IS_TEST ? Dict("skewnormal" => 4.0, "poisson" => 4.0, "dengue" => 8.0, "old_faithful" => 4.0) :
    Dict("skewnormal" => 600.0, "poisson" => 600.0, "dengue" => 3600.0, "old_faithful" => 900.0)
const N_MAX = IS_TEST ? 1_500 :
    Dict("skewnormal" => 200_000, "poisson" => 220_000, "dengue" => 300_000, "old_faithful" => 250_000)[STUDY]
const N_MIN = IS_TEST ? 100 : 2_000
const N_PILOT = IS_TEST ? 20 : 300
const BURN_FRAC = 0.1
const MAX_STORED = IS_TEST ? 1_000 : 25_000
@everywhere const PK_MAX = 30

const ROOT = joinpath(SRC, "..", "results", IS_TEST ? "gibbs_comparison_test" : "gibbs_comparison")
const OUTDIR = joinpath(ROOT, STUDY)
mkpath(joinpath(OUTDIR, "chains"))

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

function birth_family(fam::Symbol, σ)
    fam === :prior && return PriorProposal()
    fam === :nmm   && return NormalMomentMatch(σ)
    fam === :lnmm  && return LogNormalMomentMatch(σ)
    fam === :ig    && return InverseGammaMomentMatch()
    error("unknown family $fam")
end

# Study specifications

const RESULTS = joinpath(SRC, "..", "results")

function study_spec(study)
    if study == "skewnormal"
        t = CSV.read(joinpath(RESULTS, "simulation_study", "skewnormal", "tuned_sigma.csv"), DataFrame)
        σ = NamedTuple{(:ξ, :ω, :α)}(Tuple(t.σ[findfirst(==(p), t.param)] for p in ("ξ", "ω", "α")))
        return (
            scenarios = [(name = s, y = sim.y, D = sim.D, E = nothing, true_c = sim.c) for (s, sim) in SCENARIOS],
            model = SkewNormalClusters(), priors = PRIORS, ddcrp = DDCRP_PARAMS,
            prop_sds = Dict(:ξ => PROP_SDS.ξ, :ω => PROP_SDS.ω, :α => PROP_SDS.α),
            q_mm = make_birth(:nmm, :lnmm, :nmm, σ), q_mm_name = "ξ:NMM ω:LNMM α:NMM (tuned)",
            q_prior = make_birth(:prior, :prior, :prior, σ),
            rj = [(:mm, UniformLink(), NoUpdate(), "none"), (:mm, PriorLink(), NoUpdate(), "none"),
                  (:prior, UniformLink(), NoUpdate(), "none")],
            collapsed = nothing,
        )
    elseif study == "poisson"
        f = CSV.read(joinpath(RESULTS, "simulation_study", "poisson", "summary_final.csv"), DataFrame)
        best = filter(r -> occursin("[best]", r.label) && r.fdim == "none", f)
        b = best[argmax(best.esjd_K), :]
        q_mm = birth_family(Symbol(b.birth), b.tune_param)
        return (
            scenarios = [(name = s, y = sim.y, D = sim.D, E = nothing, true_c = sim.c) for (s, sim) in SCENARIOS],
            model = PoissonClusterRates(), priors = PRIORS, ddcrp = DDCRP_PARAMS,
            prop_sds = Dict{Symbol,Float64}(),
            q_mm = q_mm, q_mm_name = "$(uppercase(b.birth)) σ=$(b.tune_param) (tuned)",
            q_prior = FixedDistributionProposal(Gamma(PRIORS.λ_a, 1 / PRIORS.λ_b)),
            rj = [(:mm, UniformLink(), NoUpdate(), "none"), (:mm, PriorLink(), NoUpdate(), "none"),
                  (:prior, UniformLink(), NoUpdate(), "none")],
            collapsed = (model = PoissonClusterRatesMarg(), priors = PRIORS_MARG),
        )
    elseif study == "dengue"
        S = load_dengue_study(41, 2024)
        f = CSV.read(joinpath(RESULTS, "real_data", "dengue", "summary_final.csv"), DataFrame)
        w = first(filter(r -> r.family == "negbin", f))
        w.fdim == "none" || error("the dengue comparison assumes a No Update winner")
        mk(fρ, fr) = MixedProposal((ρ = birth_family(fρ, w.σ_birth), r = birth_family(fr, w.σ_birth)))
        return (
            scenarios = [(name = "parana_2024", y = S.y, D = S.D, E = S.E, true_c = nothing)],
            model = NegBinPopulationRates(), priors = S.priors,
            ddcrp = DDCRPParams(1.0, KERNEL_RATE, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01),
            prop_sds = Dict(:ρ => 0.15, :r => 0.30),
            q_mm = mk(Symbol(w.fam_ρ), Symbol(w.fam_r)), q_mm_name = "ρ:$(uppercase(w.fam_ρ)) r:$(uppercase(w.fam_r)) σ=$(w.σ_birth)",
            q_prior = mk(:prior, :prior),
            rj = [(:mm, PriorLink(), NoUpdate(), "none"), (:prior, PriorLink(), NoUpdate(), "none")],
            collapsed = nothing,
        )
    else
        # IGMM falls back to the prior for a Gamma shape, so q_prior is the published proposal
        return (
            scenarios = [(name = "faithful", y = Y_FAITHFUL, D = D_FAITHFUL, E = nothing, true_c = nothing)],
            model = MODEL, priors = PRIORS, ddcrp = DDCRP_PARAMS,
            prop_sds = Dict(:α => PROP_SD_SHAPE),
            q_mm = LogNormalMomentMatch(0.5), q_mm_name = "LNMM σ=0.5",
            q_prior = PriorProposal(),
            rj = [(:prior, UniformLink(), Resample(InverseGammaMomentMatch()), "resample"),
                  (:prior, UniformLink(), NoUpdate(), "none"),
                  (:mm, UniformLink(), NoUpdate(), "none"), (:mm, PriorLink(), NoUpdate(), "none")],
            collapsed = nothing,
        )
    end
end

const SPEC = study_spec(STUDY)

linkname(::UniformLink) = "uniform"
linkname(::PriorLink) = "prior"

function sampler_configs(spec)
    cfgs = NamedTuple[]
    if !isnothing(spec.collapsed)
        push!(cfgs, (label = "Collapsed Gibbs", sampler = "collapsed", q = "none", m = 0,
                     link = "none", fdim = "none", model = spec.collapsed.model,
                     priors = spec.collapsed.priors, proposal = ConjugateProposal(),
                     fdim_prop = NoUpdate(), link_prop = UniformLink()))
    end
    for (q, link, fdim, fdim_name) in spec.rj
        qp = q === :mm ? spec.q_mm : spec.q_prior
        lbl = "RJ q:$(q) link:$(linkname(link))" * (fdim_name == "resample" ? " + Resample" : "")
        push!(cfgs, (label = lbl, sampler = "rj", q = String(q), m = 0, link = linkname(link),
                     fdim = fdim_name, model = spec.model, priors = spec.priors, proposal = qp,
                     fdim_prop = fdim, link_prop = link))
    end
    for q in (:mm, :prior), m in M_POOL
        qp = q === :mm ? spec.q_mm : spec.q_prior
        push!(cfgs, (label = "Gibbs q:$(q) m=$(m)", sampler = "aux_gibbs", q = String(q), m = m,
                     link = "exact", fdim = "none", model = spec.model, priors = spec.priors,
                     proposal = AuxiliaryGibbs(qp, m), fdim_prop = NoUpdate(), link_prop = UniformLink()))
    end
    return cfgs
end

function make_run(c, sc, start; n_samples, burn, thin, save_chain = true)
    merge(c, (
        scenario = sc.name, y = sc.y, D = sc.D, E = sc.E, true_c = sc.true_c,
        ddcrp = SPEC.ddcrp, prop_sds = SPEC.prop_sds, init_proposal = SPEC.q_mm,
        start = start, start_seed = START_SEEDS[start],
        n_samples = n_samples, burn = burn, thin = thin, save_chain = save_chain,
        chain_file = joinpath(OUTDIR, "chains", fmtlabel("$(c.label)_$(sc.name)_$(start)") * ".jld2"),
    ))
end

# Worker side

@everywhere begin
    function make_data(cfg)
        !isnothing(cfg.E) && return CountDataWithPopulation(cfg.y, cfg.E, cfg.D)
        eltype(cfg.y) <: Integer && return CountData(cfg.y, cfg.D)
        return ContinuousData(cfg.y, cfg.D)
    end

    function start_links(start::Symbol, D, ddcrp)
        n = size(D, 1)
        start === :singletons && return collect(1:n)
        start === :one_cluster && return ones(Int, n)
        return simulate_ddcrp(D; α = ddcrp.α, scale = ddcrp.scale, decay_fn = ddcrp.decay_fn)
    end

    function run_chain(cfg, n_samples)
        opts = MCMCOptions(
            n_samples = n_samples, verbose = false, track_diagnostics = true,
            infer_params = Dict{Symbol,Bool}(:α_ddcrp => true, :s_ddcrp => false),
            prop_sds = merge(Dict(:s_ddcrp => 0.3), cfg.prop_sds),
            link_proposal = cfg.link_prop,
        )
        Random.seed!(cfg.start_seed)
        init_c = start_links(cfg.start, cfg.D, cfg.ddcrp)
        return mcmc(cfg.model, make_data(cfg), cfg.ddcrp, cfg.priors, cfg.proposal;
                    fixed_dim_proposal = cfg.fdim_prop, opts = opts,
                    init_c = init_c, init_proposal = cfg.init_proposal)
    end

    thin_samples(s, keep) = typeof(s)((f isa AbstractMatrix ? f[keep, :] : f[keep]
                                       for f in (getfield(s, k) for k in fieldnames(typeof(s))))...)

    function mean_moving_set(c_samples; n_draws = 100)
        rows = unique(round.(Int, range(1, size(c_samples, 1); length = n_draws)))
        mean(mean(length(get_moving_set(i, c_samples[r, :])) for i in axes(c_samples, 2)) for r in rows)
    end

    function pilot_time(cfg)
        run_chain(cfg, 3)                      # compile
        t = @elapsed run_chain(cfg, cfg.n_samples)
        return (label = cfg.label, scenario = cfg.scenario, n_pilot = cfg.n_samples,
                sec_per_iter = t / cfg.n_samples)
    end

    function run_and_save(cfg)
        run_chain(cfg, 3)                      # compile
        total_time = @elapsed begin
            samples, diag = run_chain(cfg, cfg.n_samples)
        end

        keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
        post = thin_samples(samples, keep)
        k_trace = Float64.(calculate_n_clusters(post.c))
        acc = acceptance_rates(diag)
        ess_K = effective_sample_size(k_trace)
        ari = isnothing(cfg.true_c) ? NaN : mean(compute_ari_trace(post.c, cfg.true_c))

        cfg.save_chain && jldsave(cfg.chain_file; samples = post, c = post.c, logpost = post.logpost,
                                  alpha_ddcrp = post.α_ddcrp, burn = 0, thin = 1,
                                  burn_applied = cfg.burn, thin_applied = cfg.thin, label = cfg.label)

        pk = NamedTuple{Tuple(Symbol("pk_$k") for k in 1:PK_MAX)}(Tuple(mean(k_trace .== k) for k in 1:PK_MAX))
        return merge((
            label = cfg.label, sampler = cfg.sampler, q = cfg.q, m = cfg.m, link = cfg.link,
            fdim = cfg.fdim, scenario = cfg.scenario, start = String(cfg.start),
            n_iter = cfg.n_samples, burn = cfg.burn, thin = cfg.thin, n_post = length(k_trace),
            total_time = total_time,
            sec_per_1k = 1000 * total_time / cfg.n_samples,
            mean_K = mean(k_trace), std_K = std(k_trace),
            mean_moving_set = mean_moving_set(post.c),
            ess_K = ess_K, ess_K_per_sec = ess_K / total_time,
            esjd_K = mean(diff(k_trace) .^ 2),
            ess_logpost = effective_sample_size(post.logpost),
            mean_logpost = mean(post.logpost),
            α_post = mean(post.α_ddcrp),
            mean_ari = ari,
            acc_birth = acc.birth, acc_death = acc.death, acc_fixed = acc.fixed, acc_overall = acc.overall,
            chain_file = basename(cfg.chain_file),
        ), pk)
    end
end

# Pilot

const CONFIGS = sampler_configs(SPEC)
const PILOT_FILE = joinpath(OUTDIR, "pilot_timing.csv")

println("Study $(STUDY): $(length(CONFIGS)) samplers × $(length(SPEC.scenarios)) scenario(s) × $(length(STARTS)) starts")
println("  q_mm = $(SPEC.q_mm_name)")

if FORCE_PILOT || !isfile(PILOT_FILE)
    pcfgs = [make_run(c, sc, :prior_a; n_samples = N_PILOT, burn = 0, thin = 1, save_chain = false)
             for c in CONFIGS for sc in SPEC.scenarios]
    println("Pilot: $(length(pcfgs)) runs of $(N_PILOT) iterations")
    pilot = DataFrame(pmap(pilot_time, pcfgs))
    CSV.write(PILOT_FILE, pilot)
end
const PILOT = CSV.read(PILOT_FILE, DataFrame)

function chain_length(label, scenario)
    r = PILOT[(PILOT.label .== label) .& (PILOT.scenario .== scenario), :]
    nrow(r) == 1 || error("no pilot timing for $label / $scenario; rerun with --pilot")
    n = clamp(round(Int, BUDGET[STUDY] / r.sec_per_iter[1]), N_MIN, N_MAX)
    burn = round(Int, BURN_FRAC * n)
    return (n_samples = n, burn = burn, thin = max(1, cld(n - burn, MAX_STORED)))
end

for c in CONFIGS, sc in SPEC.scenarios
    L = chain_length(c.label, sc.name)
    @printf("  %-28s %-12s %8d iterations (burn %d, thin %d)\n", c.label, sc.name, L.n_samples, L.burn, L.thin)
end

# Main run

runs = [make_run(c, sc, st; chain_length(c.label, sc.name)...)
        for c in CONFIGS for sc in SPEC.scenarios for st in STARTS]
println("Running $(length(runs)) chains on $(nworkers()) workers")
rows = pmap(run_and_save, runs)
df = DataFrame(rows)
df.study .= STUDY
df.q_mm_name .= SPEC.q_mm_name
CSV.write(joinpath(OUTDIR, "chains_$(STUDY).csv"), df)

@printf("\n%-28s %-12s %10s %10s %10s %8s\n", "sampler", "scenario", "ESS(K)", "ESS(K)/s", "s/1k it", "mean K")
for g in groupby(df, [:label, :scenario]; sort = false)
    @printf("%-28s %-12s %10.1f %10.3f %10.3f %8.2f\n", g.label[1], g.scenario[1],
            mean(g.ess_K), mean(g.ess_K_per_sec), mean(g.sec_per_1k), mean(g.mean_K))
end
println("\nWrote $(joinpath(OUTDIR, "chains_$(STUDY).csv"))")
