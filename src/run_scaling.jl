# Scaling of the samplers with n and the decay rate s on the Poisson simulation (Supplement S12).
# The data are generated as in poisson_common.jl (K* = 3, rates 1, 4, 7, centres -3, 0, 3) for
# each (n, s) of GRID; n = 150 with s = 0.5 is the Poisson data set of the paper. Four samplers
# run for a fixed number of sweeps from two shared starts: the collapsed sampler, RJ as first
# implemented (MM, uniform link, whole-cluster scoring), RJ pool (MM, prior link, m = 1) and the
# auxiliary Gibbs update (MM, m = 1). Reads the tuned Poisson birth proposal from
# results/simulation_study/poisson/summary_final.csv, so run run_poisson.jl first.
#
#   julia --project=. src/run_scaling.jl [--test] [--slurm] [--workers N]

using CSV, DataFrames, Distances, Statistics, StatsBase, Random, JLD2, Printf, Distributions
using Distributed

function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) ? default : ARGS[i+1]
end
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
end
@everywhere include($(joinpath(SRC, "poisson_common.jl")))

# Settings

const GRID = IS_TEST ? [(150, 0.5), (300, 0.5), (150, 2.0)] :
    [(150, 0.5), (600, 0.5), (2400, 0.5), (600, 0.125), (600, 2.0), (600, 8.0)]
const N_ITER = IS_TEST ? 300 : 20_000
const BURN_FRAC = 0.1
const MAX_STORED = IS_TEST ? 200 : 5_000
const STARTS = [:singletons, :prior_a]
const START_SEEDS = Dict(:singletons => 1001, :prior_a => 1003)
const RATES = [1.0, 4.0, 7.0]

const RESULTS = joinpath(SRC, "..", "results")
const OUTDIR = joinpath(RESULTS, IS_TEST ? "scaling_test" : "scaling")
mkpath(OUTDIR)

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

# Tuned moment-matched birth proposal of the Poisson study, as in run_gibbs_comparison.jl
function tuned_birth()
    f = CSV.read(joinpath(RESULTS, "simulation_study", "poisson", "summary_final.csv"), DataFrame)
    best = filter(r -> occursin("[best]", r.label) && r.fdim == "none", f)
    b = best[argmax(best.esjd_K), :]
    fam = Symbol(b.birth)
    q = fam === :nmm ? NormalMomentMatch(b.tune_param) : LogNormalMomentMatch(b.tune_param)
    return q, "$(uppercase(b.birth)) σ=$(b.tune_param) (tuned)"
end
const Q_MM, Q_MM_NAME = tuned_birth()

# Data for one grid point: the generator of poisson_common.jl with n and s as parameters

function make_scenario(n, s)
    x = make_cluster_x(n, [-3.0, 0.0, 3.0], 1.5; seed = 0)
    c = vcat([fill(1 + (k - 1) * (n ÷ K_TRUE), n ÷ K_TRUE) for k in 1:K_TRUE]...)
    length(c) < n && append!(c, fill(c[end], n - length(c)))
    D = pairwise(Euclidean(), reshape(x, 1, :))
    y = simulate_poisson(c, RATES; seed = 2)
    ddcrp = DDCRPParams(1.0, s, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
    # Effective number of candidate links of an observation: exp of the entropy of its
    # normalised link weights to the other observations, averaged over observations
    W = exp.(-s .* D)
    n_eff = mean(begin
        p = W[i, :]; p[i] = 0; p ./= sum(p)
        exp(-sum(q > 0 ? q * log(q) : 0.0 for q in p))
    end for i in 1:n)
    return (n = n, s = s, y = y, D = D, true_c = c, ddcrp = ddcrp, n_eff_links = n_eff)
end

function sampler_configs()
    [(label = "Collapsed Gibbs", sampler = "collapsed", model = PoissonClusterRatesMarg(),
      priors = PRIORS_MARG, proposal = ConjugateProposal(), link_prop = UniformLink()),
     (label = "RJ q:mm link:uniform", sampler = "rj", model = PoissonClusterRates(),
      priors = PRIORS, proposal = Q_MM, link_prop = UniformLink()),
     (label = "RJ pool q:mm link:prior m=1", sampler = "aux_mh", model = PoissonClusterRates(),
      priors = PRIORS, proposal = AuxiliaryMH(Q_MM, 1), link_prop = PriorLink()),
     (label = "Gibbs q:mm m=1", sampler = "aux_gibbs", model = PoissonClusterRates(),
      priors = PRIORS, proposal = AuxiliaryGibbs(Q_MM, 1), link_prop = UniformLink())]
end

# Worker side

@everywhere begin
    function start_links(start::Symbol, D, ddcrp)
        start === :singletons && return collect(1:size(D, 1))
        return simulate_ddcrp(D; α = ddcrp.α, scale = ddcrp.scale, decay_fn = ddcrp.decay_fn)
    end

    function run_chain(cfg, n_samples)
        opts = MCMCOptions(n_samples = n_samples, verbose = false, track_diagnostics = true,
                           infer_params = Dict{Symbol,Bool}(:α_ddcrp => true, :s_ddcrp => false),
                           prop_sds = Dict(:s_ddcrp => 0.3), link_proposal = cfg.link_prop)
        Random.seed!(cfg.start_seed)
        init_c = start_links(cfg.start, cfg.D, cfg.ddcrp)
        return mcmc(cfg.model, CountData(cfg.y, cfg.D), cfg.ddcrp, cfg.priors, cfg.proposal;
                    fixed_dim_proposal = NoUpdate(), opts = opts, init_c = init_c, init_proposal = cfg.init_proposal)
    end

    function mean_moving_set(c_samples; n_draws = 50)
        rows = unique(round.(Int, range(1, size(c_samples, 1); length = n_draws)))
        mean(mean(length(get_moving_set(i, c_samples[r, :])) for i in axes(c_samples, 2)) for r in rows)
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
            n = cfg.n, s = cfg.s, label = cfg.label, sampler = cfg.sampler, start = String(cfg.start),
            n_iter = cfg.n_samples, burn = cfg.burn, thin = cfg.thin, n_post = length(k_trace),
            n_eff_links = cfg.n_eff_links,
            total_time = total_time, sec_per_1k = 1000 * total_time / cfg.n_samples,
            mean_K = mean(k_trace), std_K = std(k_trace), mean_moving_set = mean_moving_set(c),
            ess_K = ess_K, ess_K_per_1k = 1000 * ess_K / cfg.n_samples, ess_K_per_sec = ess_K / total_time,
            esjd_K = mean(diff(k_trace) .^ 2), mean_ari = mean(compute_ari_trace(c, cfg.true_c)),
            α_post = mean(samples.α_ddcrp[keep]),
            acc_birth = acc.birth, acc_death = acc.death, acc_fixed = acc.fixed, acc_overall = acc.overall,
            chain_file = basename(cfg.chain_file),
        )
    end
end

# Run

mkpath(joinpath(OUTDIR, "chains"))
burn = round(Int, BURN_FRAC * N_ITER)
thin = max(1, cld(N_ITER - burn, MAX_STORED))
cfgs = NamedTuple[]
for (n, s) in GRID
    sc = make_scenario(n, s)
    for c in sampler_configs(), start in STARTS
        push!(cfgs, merge(c, (n = n, s = s, y = sc.y, D = sc.D, true_c = sc.true_c, ddcrp = sc.ddcrp,
                              n_eff_links = sc.n_eff_links, init_proposal = Q_MM,
                              start = start, start_seed = START_SEEDS[start],
                              n_samples = N_ITER, burn = burn, thin = thin,
                              chain_file = joinpath(OUTDIR, "chains", fmtlabel("$(c.label)_n$(n)_s$(s)_$(start)") * ".jld2"))))
    end
end

println("Scaling study: $(length(GRID)) grid points × 4 samplers × $(length(STARTS)) starts, " *
        "$N_ITER sweeps each (burn $burn, thin $thin); birth proposal $Q_MM_NAME")
for (n, s) in GRID
    @printf("  n = %4d  s = %5.3f  effective candidate links %.1f\n", n, s, make_scenario(n, s).n_eff_links)
end
flush(stdout)

# Longest chains first so the workers stay busy
order = sortperm([(-c.n, c.sampler != "aux_gibbs") for c in cfgs])
rows = pmap(run_and_save, cfgs[order])
chains = DataFrame(rows)
CSV.write(joinpath(OUTDIR, "chains_scaling.csv"), chains)

println("\nMeans over starts:")
summ = combine(groupby(chains, [:n, :s, :label, :sampler]; sort = true),
               :sec_per_1k => mean => :sec_per_1k, :mean_K => mean => :mean_K,
               :mean_moving_set => mean => :mean_moving_set, :ess_K_per_1k => mean => :ess_K_per_1k,
               :ess_K_per_sec => mean => :ess_K_per_sec)
for r in eachrow(summ)
    @printf("  n = %4d  s = %5.3f  %-28s  s/1k %7.2f  K %5.2f  |M| %6.1f  ESS(K)/1k %7.1f  ESS(K)/s %8.2f\n",
            r.n, r.s, r.label, r.sec_per_1k, r.mean_K, r.mean_moving_set, r.ess_K_per_1k, r.ess_K_per_sec)
end
println("\nDone. Run src/analyse_scaling.jl for the table.")
