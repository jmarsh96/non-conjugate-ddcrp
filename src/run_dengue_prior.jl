# Sensitivity of the dengue posterior to the empirical-Bayes prior centre λ̄ (Supplement S8).
# Reruns the selected sampler of run_dengue.jl with the Gamma(1.5, 1.5/λ̄') prior on the
# incidence centred at λ̄' = λ̄/2 and 2λ̄, one chain each from the seed of the first final chain,
# and compares each with the four pooled final chains: posterior of K, co-clustering
# probabilities and the posterior mean incidence of each municipality. Run after run_dengue.jl
# and analyse_dengue.jl (needs run_metadata.jld2, the final chains and similarity_matrix.jld2).
# Writes prior_sensitivity.csv and table_dengue_prior.tex, the latter copied into
# docs/ddcrp-paper/tables/ unless --test. --analyse skips the runs and rebuilds the table.
#
#   julia --project=. src/run_dengue_prior.jl [--analyse] [--test] [--slurm] [--workers N]

using DistanceDependentCRP, CSV, DataFrames, Statistics, StatsBase, Random, JLD2, Printf, Distributions
using Distributed

const USE_SLURM = "--slurm" in ARGS
const IS_TEST = "--test" in ARGS
const ANALYSE_ONLY = "--analyse" in ARGS

function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) ? default : parse(Int, ARGS[i+1])
end
const UF = arg_value("--uf", 41)
const YEAR = arg_value("--year", 2024)
const SCALES = [0.5, 2.0]          # λ̄' / λ̄

if !ANALYSE_ONLY
    if USE_SLURM
        using SlurmClusterManager
        addprocs(SlurmManager(launch_timeout = 300.0))
    else
        addprocs(min(parse(Int, string(arg_value("--workers", 2))), length(SCALES)))
    end
end

@everywhere begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using DistanceDependentCRP, Statistics, StatsBase, Distributions, JLD2, Random
    include(joinpath(@__DIR__, "negbin_population_model.jl"))
    include(joinpath(@__DIR__, "dengue_common.jl"))
end

const OUTDIR = joinpath(@__DIR__, "..", "results", "real_data", IS_TEST ? "dengue_test" : "dengue")
const PAPER = joinpath(@__DIR__, "..", "docs", "ddcrp-paper")
const META_FILE = joinpath(OUTDIR, "run_metadata.jld2")
isfile(META_FILE) || error("Missing $META_FILE; run run_dengue.jl$(IS_TEST ? " --test" : "") first")
const META = load(META_FILE)
const STUDY = load_dengue_study(UF, YEAR)
const λ_BAR = STUDY.ρ_bar
const ρ_A = STUDY.ρ_a

# The selected sampler and the length of the final chains, as run_dengue.jl ran them

@everywhere function _birth_family(fam::Symbol, σ)
    fam === :prior && return PriorProposal()
    fam === :nmm   && return NormalMomentMatch(σ)
    fam === :lnmm  && return LogNormalMomentMatch(σ)
    fam === :ig    && return InverseGammaMomentMatch()
    error("unknown family $fam")
end

@everywhere function run_chain(cfg)
    Random.seed!(cfg.seed)
    data = CountDataWithPopulation(cfg.y, cfg.E, cfg.D)
    opts = MCMCOptions(n_samples = cfg.n_samples, verbose = false, track_diagnostics = true,
                       infer_params = Dict{Symbol,Bool}(:α_ddcrp => true, :s_ddcrp => false),
                       prop_sds = Dict{Symbol,Float64}(:ρ => cfg.prop_sds.ρ, :r => cfg.prop_sds.r),
                       link_proposal = PriorLink())
    ddcrp = DDCRPParams(1.0, cfg.kernel_rate, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
    birth = MixedProposal((ρ = _birth_family(cfg.fam_ρ, cfg.σ_birth), r = _birth_family(cfg.fam_r, cfg.σ_birth)))
    total_time = @elapsed begin
        samples, diag = mcmc(NegBinPopulationRates(), data, ddcrp, cfg.priors, birth;
                             fixed_dim_proposal = NoUpdate(), opts = opts)
    end
    keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
    post = NegBinPopulationRatesSamples(samples.c[keep, :], samples.ρ[keep, :], samples.r[keep, :],
                                        samples.logpost[keep], samples.α_ddcrp[keep], samples.s_ddcrp[keep])
    acc = acceptance_rates(diag)
    jldsave(cfg.chain_file; samples = post, burn = 0, thin = 1, burn_applied = cfg.burn,
            thin_applied = cfg.thin, label = cfg.label, scale = cfg.scale)
    k = calculate_n_clusters(post.c)
    return (label = cfg.label, scale = cfg.scale, seed = cfg.seed, total_time = total_time,
            acc_birth = acc.birth, acc_death = acc.death, acc_fixed = acc.fixed,
            ess_K = effective_sample_size(Float64.(k)), chain_file = basename(cfg.chain_file))
end

META["best_resample"] && error("the sensitivity run assumes a No Update winner")
fmtσ(x) = replace(@sprintf("%g", x), "." => "p")
cfgs = [(label = "λ̄ × $(sc)", scale = sc, seed = first(META["final_seeds"]),
         y = STUDY.y, E = STUDY.E, D = STUDY.D, kernel_rate = KERNEL_RATE,
         fam_ρ = Symbol(META["best_fam_rho"]), fam_r = Symbol(META["best_fam_r"]),
         σ_birth = META["sigma_birth"], prop_sds = META["prop_sds"],
         priors = NegBinPopulationRatesPriors(ρ_a = ρ_A, ρ_b = ρ_A / (sc * λ_BAR), r_a = 2.0, r_b = 0.5),
         n_samples = META["n_samples"], burn = META["burn"], thin = META["thin"],
         chain_file = joinpath(OUTDIR, "chains", "prior_sens_x$(fmtσ(sc)).jld2"))
        for sc in SCALES]

const RUN_FILE = joinpath(OUTDIR, "prior_sensitivity_runs.csv")
if !ANALYSE_ONLY
    @printf("Prior sensitivity: %s, %d iterations (burn %d, thin %d), λ̄ = %.0f per 100k\n",
            META["best_label"], META["n_samples"], META["burn"], META["thin"], λ_BAR)
    for c in cfgs
        @printf("  centre %.0f per 100k  (Gamma(%.1f, rate %.2e))\n", c.scale * λ_BAR, ρ_A, c.priors.ρ_b)
    end
    runs = DataFrame(collect(pmap(run_chain, cfgs)))
    CSV.write(RUN_FILE, runs)
end

# Comparison with the pooled final chains of the main analysis

runs = CSV.read(RUN_FILE, DataFrame)
nb = [(l, f) for (l, f) in zip(META["labels"], META["chain_files"]) if !startswith(l, "Poisson")]
main = [load(joinpath(OUTDIR, "chains", f))["samples"] for (_, f) in nb]
sim_main = load(joinpath(OUTDIR, "similarity_matrix.jld2"))["sim"]
k_main = reduce(vcat, [calculate_n_clusters(s.c) for s in main])
inc_main = vec(mean(reduce(vcat, [s.ρ for s in main]), dims = 1))
kmax = 2 * maximum(k_main)
pk(k) = [mean(k .== j) for j in 1:kmax]
tv(p, q) = 0.5 * sum(abs.(p .- q))
upper(A) = [A[i, j] for j in axes(A, 2) for i in 1:(j-1)]

function compare(label, scale, s, time, ess_K)
    k = calculate_n_clusters(s.c)
    inc = vec(mean(s.ρ, dims = 1))
    Δsim = abs.(compute_similarity_matrix(s.c) .- sim_main)
    (label = label, scale = scale, centre = scale * λ_BAR, mean_K = mean(k),
     ci_lo = quantile(k, 0.025), ci_hi = quantile(k, 0.975), tv_K = tv(pk(k), pk(k_main)),
     alpha = mean(s.α_ddcrp), sim_mad = mean(upper(Δsim)), sim_max = maximum(upper(Δsim)),
     inc_min = minimum(inc), inc_max = maximum(inc),
     inc_rel_med = median(abs.(inc .- inc_main) ./ inc_main), inc_rel_max = maximum(abs.(inc .- inc_main) ./ inc_main),
     ess_K = ess_K, total_time = time)
end

rows = [compare("λ̄ (main, chain 1)", 1.0, main[1], NaN, NaN)]
for r in eachrow(runs)
    s = load(joinpath(OUTDIR, "chains", r.chain_file))["samples"]
    push!(rows, compare(r.label, r.scale, s, r.total_time, r.ess_K))
end
res = sort(DataFrame(rows), :scale)
CSV.write(joinpath(OUTDIR, "prior_sensitivity.csv"), res)

fmt(x, d) = isfinite(x) ? Printf.format(Printf.Format("%.$(d)f"), x) : "--"
fmtint(x) = replace(string(round(Int, x)), r"(\d)(?=(\d{3})+$)" => s"\1{,}")
open(joinpath(OUTDIR, "table_dengue_prior.tex"), "w") do io
    println(io, "% Generated by run_dengue_prior.jl; one chain per centre against the four pooled main chains.")
    println(io, "\\begin{tabular}{l r r c r r r r r}\n\\toprule")
    println(io, "Prior centre & per 100k & \$\\bar K\$ & 95\\% interval & TV\$(K)\$ & \$|\\Delta|\$ co-cl. & " *
                "\$\\hat\\lambda\$ range & median \$|\\Delta\\hat\\lambda|/\\hat\\lambda\$ & \$\\bar\\alpha\$ \\\\\n\\midrule")
    for r in eachrow(res)
        name = r.scale == 1.0 ? "\$\\bar\\lambda\$ (main, chain 1)" :
               r.scale < 1 ? "\$\\bar\\lambda / $(round(Int, 1 / r.scale))\$" : "\$$(round(Int, r.scale))\\bar\\lambda\$"
        println(io, "$name & $(fmtint(r.centre)) & $(fmt(r.mean_K, 1)) & [$(round(Int, r.ci_lo)), $(round(Int, r.ci_hi))] & " *
                    "$(fmt(r.tv_K, 3)) & $(fmt(r.sim_mad, 3)) & $(fmtint(r.inc_min)) to $(fmtint(r.inc_max)) & " *
                    "$(fmt(100 * r.inc_rel_med, 1))\\% & $(fmt(r.alpha, 2)) \\\\")
    end
    println(io, "\\bottomrule\n\\end{tabular}")
end
println(read(joinpath(OUTDIR, "table_dengue_prior.tex"), String))

if IS_TEST
    println("--test: assets left in $OUTDIR (docs/ddcrp-paper/ untouched).")
elseif isdir(PAPER)
    cp(joinpath(OUTDIR, "table_dengue_prior.tex"), joinpath(PAPER, "tables", "table_dengue_prior.tex"); force = true)
    println("Table copied to $PAPER/tables/")
end
