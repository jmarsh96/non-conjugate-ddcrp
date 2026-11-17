# Replicated simulated data sets (Supplement S13): how the sampler ranking of the comparison
# varies over N_REP independent draws of each simulated scenario (Poisson; skew-normal
# overlapping and separated), with the design, priors and tuned proposals of the paper's data
# sets, which are replicate 0. On each data set a reduced set of samplers runs for a fixed number
# of sweeps from two shared starts: the best auxiliary Gibbs and RJ pool configurations of the
# comparison (read from results/gibbs_comparison/summary_<study>.csv), RJ pool with MM, prior
# links and m = 1, RJ as first implemented (MM, uniform links) and, for Poisson, the collapsed
# sampler. Run the studies and run_gibbs_comparison.jl first.
#
#   julia --project=. src/run_replicates.jl [--test] [--slurm] [--workers N]

using CSV, DataFrames, Statistics, Random, JLD2, Printf, Distributions
using Distributed

arg_value(flag, default) = (i = findfirst(==(flag), ARGS); isnothing(i) ? default : ARGS[i+1])
const USE_SLURM = "--slurm" in ARGS
const IS_TEST = "--test" in ARGS

if USE_SLURM
    using SlurmClusterManager
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
    const SRC = $SRC
end
# The two setups share constant names, so each lives in its own module
@everywhere module SN
    using DistanceDependentCRP, Distributions, Distances, Random, Statistics
    include(joinpath(Main.SRC, "skewnormal_model.jl"))
    include(joinpath(Main.SRC, "skewnormal_common.jl"))
end
@everywhere module PO
    using DistanceDependentCRP, Distributions, Distances, Random, Statistics
    include(joinpath(Main.SRC, "poisson_common.jl"))
end

# Settings

const N_REP = IS_TEST ? 2 : 20
const N_ITER = IS_TEST ? 300 : 20_000
const BURN_FRAC = 0.1
const MAX_STORED = IS_TEST ? 200 : 5_000
const STARTS = [:singletons, :prior_a]
const START_SEEDS = Dict(:singletons => 1001, :prior_a => 1003)
const SCENARIOS = [("poisson", "overlapping"), ("skewnormal", "overlapping"), ("skewnormal", "separated")]

const RESULTS = joinpath(SRC, "..", "results")
const OUTDIR = joinpath(RESULTS, IS_TEST ? "replicates_test" : "replicates")
mkpath(joinpath(OUTDIR, "chains"))

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

# Study specifications: models, priors and the tuned proposals, as in run_gibbs_comparison.jl

function birth_family(fam::Symbol, σ)
    fam === :prior && return PriorProposal()
    fam === :nmm   && return NormalMomentMatch(σ)
    fam === :lnmm  && return LogNormalMomentMatch(σ)
    error("unknown family $fam")
end

function study_spec(study)
    if study == "skewnormal"
        t = CSV.read(joinpath(RESULTS, "simulation_study", "skewnormal", "tuned_sigma.csv"), DataFrame)
        σ = NamedTuple{(:ξ, :ω, :α)}(Tuple(t.σ[findfirst(==(p), t.param)] for p in ("ξ", "ω", "α")))
        return (model = SN.SkewNormalClusters(), priors = SN.PRIORS, ddcrp = SN.DDCRP_PARAMS,
                prop_sds = Dict(:ξ => SN.PROP_SDS.ξ, :ω => SN.PROP_SDS.ω, :α => SN.PROP_SDS.α),
                q_mm = SN.make_birth(:nmm, :lnmm, :nmm, σ), q_mm_name = "ξ:NMM ω:LNMM α:NMM (tuned)",
                q_prior = SN.make_birth(:prior, :prior, :prior, σ), collapsed = nothing)
    end
    f = CSV.read(joinpath(RESULTS, "simulation_study", "poisson", "summary_final.csv"), DataFrame)
    best = filter(r -> occursin("[best]", r.label) && r.fdim == "none", f)
    b = best[argmax(best.esjd_K), :]
    return (model = PoissonClusterRates(), priors = PO.PRIORS, ddcrp = PO.DDCRP_PARAMS,
            prop_sds = Dict{Symbol,Float64}(),
            q_mm = birth_family(Symbol(b.birth), b.tune_param), q_mm_name = "$(uppercase(b.birth)) σ=$(b.tune_param) (tuned)",
            q_prior = FixedDistributionProposal(Gamma(PO.PRIORS.λ_a, 1 / PO.PRIORS.λ_b)),
            collapsed = (model = PoissonClusterRatesMarg(), priors = PO.PRIORS_MARG))
end

# Replicate r of a scenario (the paper's data sets use seeds (0, 2) and (1, 3))
function make_data(study, scen, r)
    x_seed, y_seed = 10_000 + r, 20_000 + r
    study == "poisson" && return PO.make_poisson_scenario(; x_seed, y_seed)
    return SN.build_scenario(SN.SCENARIO_DESIGN[scen]...; x_seed, y_seed)
end

# A sampler configuration from its label in the comparison (e.g. "RJ pool q:mm link:prior m=3")
const LINKS = Dict("uniform" => UniformLink(), "prior" => PriorLink(), "metropolised" => MetropolisedPriorLink())
function config(label, role, sp)
    label == "Collapsed Gibbs" && return (label = label, role = role, sampler = "collapsed", q = "none", m = 0,
        link = "none", model = sp.collapsed.model, priors = sp.collapsed.priors,
        proposal = ConjugateProposal(), link_prop = UniformLink())
    mt = match(r"^(Gibbs|RJ pool|RJ) q:(mm|prior)(?: link:(uniform|prior|metropolised))?(?: m=(\d+))?$", label)
    isnothing(mt) && error("cannot parse config label $label")
    q = mt[2] == "mm" ? sp.q_mm : sp.q_prior
    m = isnothing(mt[4]) ? 0 : parse(Int, mt[4])
    kind = mt[1]
    proposal = kind == "Gibbs" ? AuxiliaryGibbs(q, m) : kind == "RJ pool" ? AuxiliaryMH(q, m) : q
    return (label = label, role = role, sampler = kind == "Gibbs" ? "aux_gibbs" : kind == "RJ pool" ? "aux_mh" : "rj",
            q = String(mt[2]), m = m, link = isnothing(mt[3]) ? "uniform" : String(mt[3]),
            model = sp.model, priors = sp.priors, proposal = proposal,
            link_prop = isnothing(mt[3]) ? UniformLink() : LINKS[mt[3]])
end

# The samplers of a scenario: the best of each form in the comparison, the two ladder endpoints
# and the collapsed sampler where it exists. A label that plays two roles is run once.
function sampler_configs(study, scen, sp)
    S = CSV.read(joinpath(RESULTS, "gibbs_comparison", "summary_$(study).csv"), DataFrame)
    d = S[S.scenario .== scen, :]
    best(smp) = (x = d[d.sampler .== smp, :]; x.label[argmax(x.ess_K_per_sec)])
    roles = ["gibbs_best" => best("aux_gibbs"), "pool_best" => best("aux_mh"),
             "pool_m1" => "RJ pool q:mm link:prior m=1", "rj_ul" => "RJ q:mm link:uniform"]
    isnothing(sp.collapsed) || push!(roles, "collapsed" => "Collapsed Gibbs")
    seen = Dict{String,String}(); order = String[]
    for (role, label) in roles
        haskey(seen, label) ? (seen[label] *= "+" * role) : (seen[label] = role; push!(order, label))
    end
    return [config(label, seen[label], sp) for label in order]
end

# Worker side

@everywhere begin
    function start_links(start::Symbol, D, ddcrp)
        start === :singletons && return collect(1:size(D, 1))
        return simulate_ddcrp(D; α = ddcrp.α, scale = ddcrp.scale, decay_fn = ddcrp.decay_fn)
    end

    observed(cfg) = eltype(cfg.y) <: Integer ? CountData(cfg.y, cfg.D) : ContinuousData(cfg.y, cfg.D)

    function run_chain(cfg, n_samples)
        opts = MCMCOptions(n_samples = n_samples, verbose = false, track_diagnostics = true,
                           infer_params = Dict{Symbol,Bool}(:α_ddcrp => true, :s_ddcrp => false),
                           prop_sds = merge(Dict(:s_ddcrp => 0.3), cfg.prop_sds), link_proposal = cfg.link_prop)
        Random.seed!(cfg.start_seed)
        init_c = start_links(cfg.start, cfg.D, cfg.ddcrp)
        return mcmc(cfg.model, observed(cfg), cfg.ddcrp, cfg.priors, cfg.proposal;
                    fixed_dim_proposal = NoUpdate(), opts = opts, init_c = init_c, init_proposal = cfg.init_proposal)
    end

    function run_and_save(cfg)
        run_chain(cfg, 3)                      # compile
        total_time = @elapsed begin
            samples, diag = run_chain(cfg, cfg.n_samples)
        end
        keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
        c = samples.c[keep, :]
        k_trace = Float64.(calculate_n_clusters(c))
        acc = acceptance_rates(diag)
        ess_K = effective_sample_size(k_trace)
        jldsave(cfg.chain_file; c = c, logpost = samples.logpost[keep], alpha_ddcrp = samples.α_ddcrp[keep],
                burn_applied = cfg.burn, thin_applied = cfg.thin, label = cfg.label)
        return (
            study = cfg.study, scenario = cfg.scenario, rep = cfg.rep, label = cfg.label, role = cfg.role,
            sampler = cfg.sampler, q = cfg.q, m = cfg.m, link = cfg.link, start = String(cfg.start),
            n_iter = cfg.n_samples, burn = cfg.burn, thin = cfg.thin, n_post = length(k_trace),
            total_time = total_time, sec_per_1k = 1000 * total_time / cfg.n_samples,
            mean_K = mean(k_trace), std_K = std(k_trace),
            ess_K = ess_K, ess_K_per_1k = 1000 * ess_K / cfg.n_samples, ess_K_per_sec = ess_K / total_time,
            esjd_K = mean(diff(k_trace) .^ 2), mean_ari = mean(compute_ari_trace(c, cfg.true_c)),
            α_post = mean(samples.α_ddcrp[keep]),
            acc_birth = acc.birth, acc_death = acc.death, acc_fixed = acc.fixed, acc_overall = acc.overall,
            chain_file = basename(cfg.chain_file),
        )
    end
end

# Run

burn = round(Int, BURN_FRAC * N_ITER)
thin = max(1, cld(N_ITER - burn, MAX_STORED))
cfgs = NamedTuple[]
for (study, scen) in SCENARIOS
    sp = study_spec(study)
    samplers = sampler_configs(study, scen, sp)
    println("$study / $scen: q_mm = $(sp.q_mm_name)")
    for c in samplers
        println("  $(rpad(c.label, 36)) $(c.role)")
    end
    for r in 1:N_REP
        sim = make_data(study, scen, r)
        for c in samplers, start in STARTS
            push!(cfgs, merge(c, (study = study, scenario = scen, rep = r, y = sim.y, D = sim.D, true_c = sim.c,
                                  ddcrp = sp.ddcrp, prop_sds = sp.prop_sds, init_proposal = sp.q_mm,
                                  start = start, start_seed = START_SEEDS[start],
                                  n_samples = N_ITER, burn = burn, thin = thin,
                                  chain_file = joinpath(OUTDIR, "chains", fmtlabel("$(study)_$(scen)_r$(r)_$(c.label)_$(start)") * ".jld2"))))
        end
    end
end
println("Replicates: $(length(cfgs)) chains of $N_ITER sweeps (burn $burn, thin $thin) on $(nworkers()) workers")
flush(stdout)

# Costlier chains first so the workers stay busy
order = sortperm([(c.study != "skewnormal", c.sampler != "aux_gibbs") for c in cfgs])
rows = pmap(run_and_save, cfgs[order])
chains = DataFrame(rows)
CSV.write(joinpath(OUTDIR, "chains_replicates.csv"), chains)

println("\nMedian over data sets of the mean over starts of ESS(K)/s:")
per_data = combine(groupby(chains, [:study, :scenario, :rep, :label]), :ess_K_per_sec => mean => :ess_K_per_sec)
for g in groupby(combine(groupby(per_data, [:study, :scenario, :label]; sort = true), :ess_K_per_sec => median => :med), [:study, :scenario])
    for r in eachrow(g)
        @printf("  %-10s %-12s %-36s %8.2f\n", r.study, r.scenario, r.label, r.med)
    end
end
println("\nDone. Run src/analyse_replicates.jl for the table and figure.")
