# Oracle diagnostic for single-link births (docs/sampler_improvements.md, §1). From states of an
# RJ pool (m = 1, prior links) chain it takes the births the sampler would propose and compares
# their mean acceptance E_q[min(1, r)] with the bound A* = min{1, m(y_M) / p(y_M | ρ_O)} of the
# exact conditional parameter proposal. m(y_M) is by quadrature for one parameter (Old Faithful)
# and by defensive importance sampling otherwise (two independent estimates, weight ESS). It also
# counts the links that separate the high-posterior partitions of each K.
#
#   julia --project=. src/diagnose_births.jl [--test] [--workers N]

using Distributed, CSV, DataFrames, Statistics, Printf

arg_value(flag, default) = (i = findfirst(==(flag), ARGS); isnothing(i) ? default : ARGS[i+1])
const IS_TEST = "--test" in ARGS
addprocs(parse(Int, arg_value("--workers", "5")))

@everywhere begin
    using DistanceDependentCRP, Distributions, Distances, Random, Statistics, StatsBase, LinearAlgebra
    using CSV, DataFrames
    BLAS.set_num_threads(1)
    const DD = DistanceDependentCRP
    const SRC = $(@__DIR__)
    const IS_TEST = $IS_TEST
end
# The two setups share constant names, so each lives in its own module
@everywhere module SN
    using DistanceDependentCRP, Distributions, Distances, Random, Statistics
    include(joinpath(Main.SRC, "skewnormal_model.jl"))
    include(joinpath(Main.SRC, "skewnormal_common.jl"))
end
@everywhere module OF
    using DistanceDependentCRP, Distances, CSV, DataFrames
    include(joinpath(Main.SRC, "old_faithful_common.jl"))
end

const RESULTS = joinpath(@__DIR__, "..", "results")
const OUTDIR = joinpath(RESULTS, IS_TEST ? "sampler_study_test" : "sampler_study")
mkpath(OUTDIR)

function skew_setup(scen)
    t = CSV.read(joinpath(RESULTS, "simulation_study", "skewnormal", "tuned_sigma.csv"), DataFrame)
    σ = NamedTuple{(:ξ, :ω, :α)}(Tuple(t.σ[findfirst(==(p), t.param)] for p in ("ξ", "ω", "α")))
    σ2 = map(x -> 2x, σ)
    sim = SN.make_scenario(scen, 1)
    return (study = "skewnormal", scenario = scen, model = SN.SkewNormalClusters(),
            data = ContinuousData(sim.y, sim.D), ddcrp = SN.DDCRP_PARAMS, priors = SN.PRIORS,
            q = SN.make_birth(:nmm, :lnmm, :nmm, σ),
            q_is = [SN.make_birth(:nmm, :lnmm, :nmm, σ), SN.make_birth(:nmm, :lnmm, :nmm, σ2),
                    SN.make_birth(:prior, :prior, :prior, σ)],
            prop_sds = Dict(:ξ => SN.PROP_SDS.ξ, :ω => SN.PROP_SDS.ω, :α => SN.PROP_SDS.α))
end

function of_setup()
    σ = CSV.read(joinpath(RESULTS, "old_faithful", "tuned_sigma.csv"), DataFrame).σ[1]
    return (study = "old_faithful", scenario = "faithful", model = OF.MODEL,
            data = ContinuousData(OF.Y_FAITHFUL, OF.D_FAITHFUL), ddcrp = OF.DDCRP_PARAMS, priors = OF.PRIORS,
            q = LogNormalMomentMatch(σ), q_is = nothing, prop_sds = Dict(:α => OF.PROP_SD_SHAPE))
end

@everywhere begin
    const N_ITER = IS_TEST ? 1_500 : 20_000
    const BURN = IS_TEST ? 500 : 2_000
    const N_STATES = IS_TEST ? 4 : 40
    const I_PER_STATE = IS_TEST ? 5 : 25
    const N_Q = IS_TEST ? 200 : 2_000
    const N_IS = IS_TEST ? 2_000 : 50_000

    _lse(x) = (m = maximum(x); isfinite(m) ? m + log(sum(exp.(x .- m))) : m)

    # Rebuild a state from a stored draw (per-observation parameter matrices)
    function state_at(S, s, setup)
        st = initialise_state(setup.model, setup.data, setup.ddcrp, setup.priors)
        st.c .= S.c[s, :]
        dicts = cluster_param_dicts(st)
        foreach(empty!, dicts)
        for t in table_vector(st.c), k in keys(dicts)
            dicts[k][sort(t)] = getfield(S, k)[s, t[1]]
        end
        return st
    end

    # log m(y_M): quadrature for one parameter, defensive importance sampling otherwise
    function log_marginal(setup, M, st, rng_seed)
        model, data, priors = setup.model, setup.data, setup.priors
        if length(param_scales(model)) == 1
            ts = range(-12, 12; length = 4001)
            f = [contribution_with(model, M, DD.from_t(model, [t]), st, data, priors) + DD.logjac(model, [t]) for t in ts]
            v = _lse(f) + log(step(ts))
            return v, v, Float64(length(ts))
        end
        est = Float64[]; ess = Float64[]
        Random.seed!(rng_seed)
        for _ in 1:2
            lw = Vector{Float64}(undef, N_IS)
            for r in 1:N_IS
                ρ, _ = sample_birth_params(model, setup.q_is[rand(1:3)], M, st, data, priors)
                lq = _lse([birth_params_logpdf(model, qk, ρ, M, st, data, priors) for qk in setup.q_is]) - log(3)
                lw[r] = contribution_with(model, M, ρ, st, data, priors) - lq
            end
            replace!(x -> isnan(x) ? -Inf : x, lw)
            push!(est, _lse(lw) - log(N_IS))
            w = exp.(lw .- maximum(lw)); push!(ess, sum(w)^2 / sum(abs2, w))
        end
        return est[1], est[2], minimum(ess)
    end

    function diagnose(setup)
        model, data, priors = setup.model, setup.data, setup.priors
        Random.seed!(2024)
        opts = MCMCOptions(n_samples = N_ITER, verbose = false, track_diagnostics = true,
                           infer_params = Dict(:α_ddcrp => true, :s_ddcrp => false),
                           prop_sds = merge(Dict(:s_ddcrp => 0.3), setup.prop_sds), link_proposal = PriorLink())
        S, diag = mcmc(model, data, setup.ddcrp, priors, AuxiliaryMH(setup.q, 1); opts = opts)
        n = size(S.c, 2)
        D = distance_matrix(data)
        rows = NamedTuple[]
        for (si, s) in enumerate(round.(Int, range(BURN + 1, N_ITER; length = N_STATES)))
            st = state_at(S, s, setup)
            L = precompute_log_ddcrp(setup.ddcrp.decay_fn, S.α_ddcrp[s], setup.ddcrp.scale, D)
            K = length(table_vector(st.c))
            dicts = cluster_param_dicts(st)
            for i in shuffle(MersenneTwister(si), 1:n)[1:min(I_PER_STATE, n)]
                O = DD.find_table_for_customer(i, first(dicts))
                M = DD.get_moving_set(i, st.c, O)
                R = DD.sorted_setdiff(O, M)
                isempty(R) && continue
                ρ_O = map(d -> d[O], dicts)
                w = exp.(L[i, :] .- maximum(L[i, :]))
                π_birth = sum(w[M]) / sum(w)
                lp_O = DD.moving_set_loglik(model, M, ρ_O, R, st, data, priors)
                acc = 0.0
                for _ in 1:N_Q
                    ρ, lq = sample_birth_params(model, setup.q, M, st, data, priors)
                    a = contribution_with(model, M, ρ, st, data, priors) - lq - lp_O
                    acc += isnan(a) ? 0.0 : min(1.0, exp(a))
                end
                lm1, lm2, ess = log_marginal(setup, M, st, 1000si + i)
                push!(rows, (study = setup.study, scenario = setup.scenario, state = s, i = i,
                             size_M = length(M), size_R = length(R), K = K, p_birth = π_birth,
                             acc_obs = acc / N_Q, A_star = min(1.0, exp((lm1 + lm2) / 2 - lp_O)),
                             logm_1 = lm1, logm_2 = lm2, is_ess = ess, log_lik_O = lp_O))
            end
        end
        return (moves = DataFrame(rows), structure = structure(S, setup), acc = acceptance_rates(diag))
    end

    # Links separating the Dahl partition of each well-supported K from the others
    function structure(S, setup)
        draws = (BURN + 1):size(S.c, 1)
        Z = [compute_table_assignments(S.c[s, :]) for s in draws]
        Ks = length.(unique.(Z))
        rows = NamedTuple[]
        modes = Tuple{Int, Int}[]                     # (K, draw index)
        for k in sort(unique(Ks))
            idx = findall(==(k), Ks)
            length(idx) / length(Ks) < 0.05 && continue
            sub = idx[round.(Int, range(1, length(idx); length = min(length(idx), 400)))]
            n = length(Z[1])
            P = zeros(n, n)
            for r in sub, i in 1:n, j in 1:n
                P[i, j] += Z[r][i] == Z[r][j]
            end
            P ./= length(sub)
            loss(r) = sum(((Z[r][i] == Z[r][j]) - P[i, j])^2 for i in 1:n, j in 1:n if i < j)
            push!(modes, (k, sub[argmin(loss.(sub))]))
        end
        for (k1, r1) in modes, (k2, r2) in modes
            k1 == k2 && continue
            c1 = S.c[draws[r1], :]; z2 = Z[r2]
            cut = count(i -> z2[i] != z2[c1[i]], eachindex(c1))
            kept = [z2[i] == z2[c1[i]] ? c1[i] : i for i in eachindex(c1)]
            comps = Dict{Int, Set{Int}}()
            for (i, l) in enumerate(compute_table_assignments(kept))
                push!(get!(comps, z2[i], Set{Int}()), l)
            end
            extra = sum(length(v) - 1 for v in values(comps))
            push!(rows, (study = setup.study, scenario = setup.scenario, K_from = k1, K_to = k2,
                         links_cut = cut, links_upper = cut + extra))
        end
        return DataFrame(rows)
    end
end

setups = Any[skew_setup(s) for s in SN.STUDY_SCENARIOS]
push!(setups, of_setup())
println("Diagnosing $(length(setups)) targets on $(nworkers()) workers")
res = pmap(diagnose, setups)

moves = vcat([r.moves for r in res]...)
structure_df = vcat([r.structure for r in res]...; cols = :union)
CSV.write(joinpath(OUTDIR, "diagnose_births_moves.csv"), moves)
CSV.write(joinpath(OUTDIR, "diagnose_births_structure.csv"), structure_df)

bin(m) = m == 1 ? "1" : m <= 3 ? "2-3" : m <= 10 ? "4-10" : m <= 30 ? "11-30" : ">30"
moves.bin = bin.(moves.size_M)
summary = combine(groupby(moves, [:study, :scenario, :bin]),
    nrow => :n_moves, :p_birth => mean => :p_birth,
    [:acc_obs, :p_birth] => ((a, p) -> sum(a .* p) / sum(p)) => :acc_obs,
    [:A_star, :p_birth] => ((a, p) -> sum(a .* p) / sum(p)) => :A_star,
    [:logm_1, :logm_2] => ((a, b) -> maximum(abs.(a .- b))) => :max_logm_gap,
    :is_ess => minimum => :min_is_ess)
summary.ratio = summary.acc_obs ./ summary.A_star
overall = combine(groupby(moves, [:study, :scenario]),
    [:acc_obs, :p_birth] => ((a, p) -> sum(a .* p) / sum(p)) => :acc_obs,
    [:A_star, :p_birth] => ((a, p) -> sum(a .* p) / sum(p)) => :A_star)
overall.bin .= "all"
overall.ratio = overall.acc_obs ./ overall.A_star
summary = vcat(summary, overall; cols = :union)
sort!(summary, [:study, :scenario, :bin])
CSV.write(joinpath(OUTDIR, "diagnose_births_summary.csv"), summary)

println("\nBirth acceptance: observed vs oracle bound A* (weighted by birth-proposal probability)")
for r in eachrow(summary)
    @printf("%-12s %-12s |M| %-6s moves %5s  acc %.4f  A* %.4f  ratio %.2f  logm gap %s\n",
            r.study, r.scenario, r.bin, coalesce(r.n_moves, missing) === missing ? "" : string(r.n_moves),
            r.acc_obs, r.A_star, r.ratio, ismissing(r.max_logm_gap) ? "" : @sprintf("%.3f", r.max_logm_gap))
end
println("\nLinks separating the Dahl partitions of each K (cut = lower bound, upper adds rejoins):")
show(structure_df; allrows = true)
println("\nSampler acceptance rates:")
for (s, r) in zip(setups, res)
    println("  $(s.study) $(s.scenario): ", map(x -> round(x; digits = 4), r.acc))
end
