# Sampler study for the birth/death bottleneck (docs/sampler_improvements.md): the coupled
# birth/death, the informed link proposal and the two-anchor split/merge, each on the published
# reversible jump update and on RJ pool (AuxiliaryMH, m = 1), against auxiliary Gibbs, on simulated
# skew-normal data. Same posterior, four shared starts, equal wall clock per chain from a pilot.
#
#   --stage tune     coupled contrast scales and two-anchor settings by ESJD(K) (tuned_moves.csv)
#   --stage screen   every config on replicate 1 of each scenario
#   --stage confirm  every config (or --configs) on replicates 1-3
#   --stage poisson  every move on the Poisson simulation against collapsed Gibbs (agreement)
#   --stage cost     sec per sweep of every screen config after warm-up, one chain at a time
#                    (run with --workers 1 on an idle core; the analysis rescales per-sweep ESS by it)
#
#   julia --project=. src/run_sampler_study.jl --stage <s> [--configs "..."] [--groups "..."] [--pilot] [--clean] [--test]
#         [--slurm] [--workers N]
#
# --configs "a;b": confirm runs those plus RJ, RJ pool and Gibbs m=3; screen and cost run only those and
# merge their rows into the stage's existing CSV. --groups "pilot" (tune) retunes only the named groups
# (contrast, two_anchor, pilot: the data-informed two-anchor, tilt × p_cross × k). --clean sets
# chain lengths from the clean cost of --stage cost instead of the pilot, so every chain gets the same
# compute however unevenly the machine's cores run it (the analysis rescales by the same cost).

using CSV, DataFrames, Statistics, StatsBase, Random, JLD2, Printf
using Distributed

arg_value(flag, default) = (i = findfirst(==(flag), ARGS); isnothing(i) ? default : ARGS[i+1])
const STAGE = arg_value("--stage", "")
STAGE in ("tune", "screen", "confirm", "poisson", "cost") || error("--stage must be tune, screen, confirm, poisson or cost")
const IS_TEST = "--test" in ARGS
const FORCE_PILOT = "--pilot" in ARGS
const CLEAN = "--clean" in ARGS

if "--slurm" in ARGS
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
# The skew-normal and Poisson setups share constant names, so each lives in its own module
@everywhere module SN
    using DistanceDependentCRP, Distributions, Distances, Random, Statistics
    include(joinpath(Main.SRC, "skewnormal_model.jl"))
    include(joinpath(Main.SRC, "skewnormal_common.jl"))
end
@everywhere module PO
    using DistanceDependentCRP, Distributions, Distances, Random
    include(joinpath(Main.SRC, "poisson_common.jl"))
end

# Settings

const STARTS = [:singletons, :one_cluster, :prior_a, :prior_b]
const START_SEEDS = Dict(:singletons => 1001, :one_cluster => 1002, :prior_a => 1003, :prior_b => 1004)
const REPS = STAGE == "confirm" ? [1, 2, 3] : [1]
const BUDGET = IS_TEST ? 3.0 : get(Dict("screen" => 300.0, "confirm" => 600.0, "poisson" => 150.0), STAGE, 0.0)
const N_MIN = IS_TEST ? 100 : 2_000
const N_PILOT = IS_TEST ? 20 : 200
const BURN_FRAC = 0.1
const MAX_STORED = IS_TEST ? 1_000 : 20_000

const TUNE_SCALES = IS_TEST ? [0.5, 1.0] : [0.25, 0.5, 1.0, 2.0]
const TUNE_TA = IS_TEST ? [(k = 3.0, attempts = 1)] :
    [(k = k, attempts = a) for k in (1.0, 3.0, 10.0) for a in (1, 15)]
const N_TUNE = IS_TEST ? 300 : 20_000
const TUNE_PILOT = IS_TEST ? [(k = 1.0, tilt = 1.0, p_cross = 0.9)] :
    [(k = k, tilt = t, p_cross = p) for k in (1.0, 10.0) for t in (1.0, 3.0) for p in (0.5, 0.9)]
const TUNE_GROUPS = split(arg_value("--groups", "contrast,two_anchor,pilot"), ",")
const TUNE_SCENARIOS = ["overlapping", "aba"]
const TUNE_STARTS = [:prior_a, :one_cluster]
const COUPLED_P = 0.5
const N_COST_WARM = IS_TEST ? 20 : 500
const N_COST = IS_TEST ? 50 : 1_000

const RESULTS = joinpath(SRC, "..", "results")
const ROOT = joinpath(RESULTS, IS_TEST ? "sampler_study_test" : "sampler_study")
const OUTDIR = joinpath(ROOT, STAGE == "tune" ? "tune" : STAGE)
mkpath(joinpath(OUTDIR, "chains"))
const TUNED_FILE = joinpath(ROOT, "tuned_moves.csv")

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

# Write df to file, keeping the rows of file whose `key` is not in df (runs of a subset of configs)
function merge_write(file, df; key = :label)
    if isfile(file)
        old = CSV.read(file, DataFrame)
        df = vcat(old[.!in(Set(df[!, key])).(old[!, key]), :], df; cols = :union)
    end
    CSV.write(file, df)
    return df
end

# Birth proposals: the tuned RJ proposals of each study
function skew_q()
    t = CSV.read(joinpath(RESULTS, "simulation_study", "skewnormal", "tuned_sigma.csv"), DataFrame)
    σ = NamedTuple{(:ξ, :ω, :α)}(Tuple(t.σ[findfirst(==(p), t.param)] for p in ("ξ", "ω", "α")))
    return SN.make_birth(:nmm, :lnmm, :nmm, σ)
end
function poisson_q()
    f = CSV.read(joinpath(RESULTS, "simulation_study", "poisson", "summary_final.csv"), DataFrame)
    b = filter(r -> occursin("[best]", r.label) && r.fdim == "none", f)
    b = b[argmax(b.esjd_K), :]
    return b.birth == "lnmm" ? LogNormalMomentMatch(b.tune_param) : NormalMomentMatch(b.tune_param)
end

const CONTRASTS = Dict("FitT" => s -> FitStudentT(4.0, s), "ZeroT" => s -> ZeroStudentT(4.0, s), "FitN" => s -> FitNormal(s))

# One config: base ∈ (rj, pool, gibbs); coupled contrast name or ""; link; two-anchor or nothing
function config(label, base, q; contrast = "", scale = 1.0, link = PriorLink(), ta = nothing)
    cp = contrast == "" ? nothing : Coupled(CONTRASTS[contrast](scale), COUPLED_P)
    birth = base == "rj" ? (isnothing(cp) ? q : CoupledRJ(q, cp)) :
            base == "pool" ? AuxiliaryMH(q, 1; coupled = cp) :
            base == "gibbs" ? AuxiliaryGibbs(q, 3) : ConjugateProposal()
    return (label = label, base = base, contrast = contrast, scale = scale,
            link_name = link isa InformedLink ? "informed" : link isa PriorLink ? "prior" : "uniform",
            ta_k = isnothing(ta) ? 0.0 : ta.k, ta_attempts = isnothing(ta) ? 0 : ta.attempts,
            birth = birth, link = link, ta = ta)
end

function study_configs(q, tuned; collapsed = false)
    scale_of(name) = tuned.scale[findfirst(==(name), tuned.setting)]
    best = tuned.setting[findfirst(==("best_contrast"), tuned.kind)]
    tr = tuned[tuned.kind .== "two_anchor", :][1, :]
    ta = TwoAnchor(k = tr.scale, attempts = Int(tr.attempts))
    cfgs = [config("RJ", "rj", q), config("RJ pool", "pool", q)]
    for g in ("FitT", "ZeroT", "FitN"), base in ("rj", "pool")
        push!(cfgs, config("$(base == "rj" ? "RJ" : "RJ pool") + coupled $g", base, q; contrast = g, scale = scale_of(g)))
    end
    append!(cfgs, [
        config("RJ + informed", "rj", q; link = InformedLink(0.1, 1.0)),
        config("RJ pool + informed", "pool", q; link = InformedLink(0.1, 1.0)),
        config("RJ + two-anchor", "rj", q; ta = ta),
        config("RJ pool + two-anchor", "pool", q; ta = ta),
        config("RJ + coupled $best + two-anchor", "rj", q; contrast = best, scale = scale_of(best), ta = ta),
        config("RJ pool + coupled $best + two-anchor", "pool", q; contrast = best, scale = scale_of(best), ta = ta),
        config("Gibbs m=3", "gibbs", q; link = UniformLink()),
        config("Gibbs m=3 + two-anchor", "gibbs", q; link = UniformLink(), ta = ta),
    ])
    if "pilot" in tuned.kind
        pr = tuned[tuned.kind .== "pilot", :][1, :]
        tp = TwoAnchor(k = pr.scale, attempts = Int(pr.attempts), pilot = true, tilt = pr.tilt, p_cross = pr.p_cross)
        append!(cfgs, [
            config("RJ + pilot two-anchor", "rj", q; ta = tp),
            config("RJ pool + pilot two-anchor", "pool", q; ta = tp),
            config("Gibbs m=3 + pilot two-anchor", "gibbs", q; link = UniformLink(), ta = tp),
        ])
    end
    collapsed && pushfirst!(cfgs, config("Collapsed Gibbs", "collapsed", q; link = UniformLink()))
    return cfgs
end

# Worker side

@everywhere begin
    function study_objects(study, base)
        study == "skewnormal" && return (model = SN.SkewNormalClusters(), priors = SN.PRIORS, ddcrp = SN.DDCRP_PARAMS,
            prop_sds = Dict(:ξ => SN.PROP_SDS.ξ, :ω => SN.PROP_SDS.ω, :α => SN.PROP_SDS.α))
        base == "collapsed" && return (model = PoissonClusterRatesMarg(), priors = PO.PRIORS_MARG,
                                       ddcrp = PO.DDCRP_PARAMS, prop_sds = Dict{Symbol, Float64}())
        return (model = PoissonClusterRates(), priors = PO.PRIORS, ddcrp = PO.DDCRP_PARAMS, prop_sds = Dict{Symbol, Float64}())
    end

    function make_data(cfg)
        eltype(cfg.y) <: Integer && return CountData(cfg.y, cfg.D)
        return ContinuousData(cfg.y, cfg.D)
    end

    function start_links(start::Symbol, D, ddcrp)
        n = size(D, 1)
        start === :singletons && return collect(1:n)
        start === :one_cluster && return ones(Int, n)
        return simulate_ddcrp(D; α = ddcrp.α, scale = ddcrp.scale, decay_fn = ddcrp.decay_fn)
    end

    function run_chain(cfg, n_samples, thin)
        O = study_objects(cfg.study, cfg.base)
        opts = MCMCOptions(n_samples = n_samples, verbose = false, track_diagnostics = true, thin = thin,
                           infer_params = Dict{Symbol,Bool}(:α_ddcrp => true, :s_ddcrp => false),
                           prop_sds = merge(Dict(:s_ddcrp => 0.3), O.prop_sds),
                           link_proposal = cfg.link, two_anchor = cfg.ta)
        Random.seed!(cfg.start_seed)
        init_c = start_links(cfg.start, cfg.D, O.ddcrp)
        init_q = cfg.base == "collapsed" ? PriorProposal() : cfg.init_q
        return mcmc(O.model, make_data(cfg), O.ddcrp, O.priors, cfg.birth;
                    fixed_dim_proposal = NoUpdate(), opts = opts, init_c = init_c, init_proposal = init_q)
    end

    function pilot_time(cfg)
        run_chain(cfg, 3, 1)
        t = @elapsed run_chain(cfg, cfg.n_samples, 1)
        return (label = cfg.label, scenario = cfg.scenario, rep = cfg.rep, sec_per_iter = t / cfg.n_samples)
    end

    # Same seed, so the longer chain repeats the warm-up and the difference times N_COST warm sweeps
    function cost_time(cfg, n_warm, n)
        run_chain(cfg, 3, 1)
        t0 = @elapsed run_chain(cfg, n_warm, 1)
        t1 = @elapsed run_chain(cfg, n_warm + n, 1)
        return (label = cfg.label, scenario = cfg.scenario, rep = cfg.rep, sec_per_iter = max(t1 - t0, 0.0) / n)
    end

    function move_stats(d)
        t = d.two_anchor
        nv(a, inv) = a - inv
        sa, ma = t.split_attempts, t.merge_attempts
        return (cb_acc = d.coupled_birth_proposes > 0 ? d.coupled_birth_accepts / d.coupled_birth_proposes : NaN,
                cd_acc = d.coupled_death_proposes > 0 ? d.coupled_death_accepts / d.coupled_death_proposes : NaN,
                cb_prop = d.coupled_birth_proposes,
                ta_split_attempts = sa, ta_split_valid = nv(sa, t.split_invalid), ta_split_accepts = t.split_accepts,
                ta_merge_attempts = ma, ta_merge_valid = nv(ma, t.merge_invalid), ta_merge_accepts = t.merge_accepts,
                ta_mean_U = sa + ma > 0 ? t.U_size_sum / (sa + ma) : NaN,
                ta_mean_links_valid = isempty(t.valid_links_changed) ? NaN : mean(t.valid_links_changed),
                ta_split_ge2 = isempty(t.split_accepted_links) ? NaN : mean(t.split_accepted_links .>= 2),
                ta_time = t.time)
    end

    function run_and_save(cfg)
        run_chain(cfg, 3, 1)
        total_time = @elapsed begin
            samples, diag = run_chain(cfg, cfg.n_samples, cfg.thin)
        end
        stored = size(samples.c, 1)
        keep = (cfg.burn + 1):stored
        c = samples.c[keep, :]
        k_trace = Float64.(calculate_n_clusters(c))
        acc = acceptance_rates(diag)
        ess_K = effective_sample_size(k_trace)
        cfg.save_chain && jldsave(cfg.chain_file; c = c, logpost = samples.logpost[keep],
                                  alpha_ddcrp = samples.α_ddcrp[keep], label = cfg.label)
        return merge((
            label = cfg.label, base = cfg.base, contrast = cfg.contrast, scale = cfg.scale,
            link = cfg.link_name, ta_k = cfg.ta_k, ta_attempts = cfg.ta_attempts,
            study = cfg.study, scenario = cfg.scenario, rep = cfg.rep, start = String(cfg.start),
            n_iter = cfg.n_samples, thin = cfg.thin, n_post = length(k_trace), total_time = total_time,
            sec_per_1k = 1000 * total_time / cfg.n_samples, mean_K = mean(k_trace),
            ess_K = ess_K, ess_K_per_sec = ess_K / total_time, esjd_K = mean(diff(k_trace) .^ 2),
            ess_logpost = effective_sample_size(samples.logpost[keep]),
            mean_ari = isnothing(cfg.true_c) ? NaN : mean(compute_ari_trace(c, cfg.true_c)),
            acc_birth = acc.birth, acc_death = acc.death, acc_fixed = acc.fixed, acc_overall = acc.overall,
            chain_file = basename(cfg.chain_file),
        ), move_stats(diag), NamedTuple{Tuple(Symbol("pk_$k") for k in 1:20)}(Tuple(mean(k_trace .== k) for k in 1:20)))
    end
end

# Data sets of the stage
skew_scenario(s, r) = (d = SN.make_scenario(s, r);
    (study = "skewnormal", name = s, rep = r, y = d.y, D = d.D, true_c = d.c))
scenarios() = STAGE == "poisson" ?
    [(study = "poisson", name = "overlapping", rep = 1, y = PO.SIM_OVERLAPPING.y, D = PO.SIM_OVERLAPPING.D,
      true_c = PO.SIM_OVERLAPPING.c)] :
    [skew_scenario(s, r) for s in (STAGE == "tune" ? TUNE_SCENARIOS : SN.STUDY_SCENARIOS) for r in REPS]

function make_run(c, sc, start, q; n_samples, burn = 0, thin = 1, save_chain = true)
    merge(c, (study = sc.study, scenario = sc.name, rep = sc.rep, y = sc.y, D = sc.D, true_c = sc.true_c,
              start = start, start_seed = START_SEEDS[start] + 10 * sc.rep, init_q = q,
              n_samples = n_samples, burn = burn, thin = thin, save_chain = save_chain,
              chain_file = joinpath(OUTDIR, "chains", fmtlabel("$(c.label)_$(sc.name)_r$(sc.rep)_$(start)") * ".jld2")))
end

# Tune stage: contrast scales per family and two-anchor settings on RJ pool, by ESJD(K)

# Groups (--groups): contrast, two_anchor, pilot (the pilot keeps the tuned two-anchor attempts).
# Rows of tuned_moves.csv for groups not run are kept.
function run_tune()
    q = skew_q()
    old = isfile(TUNED_FILE) ? CSV.read(TUNED_FILE, DataFrame) : nothing
    pilot_attempts = isnothing(old) || !("two_anchor" in old.kind) ? 15 :
        Int(old.attempts[findfirst(==("two_anchor"), old.kind)])
    cands = []
    "contrast" in TUNE_GROUPS && append!(cands,
        [(config("pool coupled $g s=$s", "pool", q; contrast = g, scale = s), (group = "contrast", tilt = NaN, p_cross = NaN))
         for g in keys(CONTRASTS) for s in TUNE_SCALES])
    "two_anchor" in TUNE_GROUPS && append!(cands,
        [(config("pool two-anchor k=$(t.k) a=$(t.attempts)", "pool", q; ta = TwoAnchor(k = t.k, attempts = t.attempts)),
          (group = "two_anchor", tilt = NaN, p_cross = NaN)) for t in TUNE_TA])
    "pilot" in TUNE_GROUPS && append!(cands,
        [(config("pool pilot k=$(t.k) tilt=$(t.tilt) pc=$(t.p_cross)", "pool", q;
                 ta = TwoAnchor(k = t.k, attempts = pilot_attempts, pilot = true, tilt = t.tilt, p_cross = t.p_cross)),
          (group = "pilot", tilt = t.tilt, p_cross = t.p_cross)) for t in TUNE_PILOT])
    runs = [make_run(c, sc, st, q; n_samples = N_TUNE, burn = N_TUNE ÷ 10, save_chain = false)
            for (c, _) in cands for sc in scenarios() for st in TUNE_STARTS]
    println("Tuning $(join(TUNE_GROUPS, ", ")): $(length(runs)) chains of $(N_TUNE) iterations")
    tune = DataFrame(pmap(run_and_save, runs))
    meta = Dict(c.label => m for (c, m) in cands)
    tune.group = [meta[l].group for l in tune.label]
    tune.tilt = [meta[l].tilt for l in tune.label]
    tune.p_cross = [meta[l].p_cross for l in tune.label]
    merge_write(joinpath(OUTDIR, "tune_chains.csv"), tune)
    by = combine(groupby(tune, [:label, :group, :contrast, :scale, :ta_k, :ta_attempts, :tilt, :p_cross]),
                 :esjd_K => mean => :esjd_K, :ess_K_per_sec => mean => :ess_K_per_sec, :cb_acc => mean => :cb_acc,
                 [:ta_split_valid, :ta_split_attempts] => ((v, a) -> sum(v) / max(1, sum(a))) => :ta_split_valid_rate,
                 [:ta_split_accepts, :ta_split_valid] => ((v, a) -> sum(v) / max(1, sum(a))) => :ta_split_acc,
                 [:ta_split_accepts, :ta_merge_accepts, :n_iter] => ((s, m, n) -> 1000 * (sum(s) + sum(m)) / sum(n)) => :ta_acc_per_ksweep)
    row(kind, setting, scale, attempts, esjd; tilt = NaN, p_cross = NaN) =
        (kind = kind, setting = setting, scale = scale, attempts = attempts, esjd_K = esjd, tilt = tilt, p_cross = p_cross)
    rows = NamedTuple[]
    if "contrast" in TUNE_GROUPS
        for g in keys(CONTRASTS)
            d = by[by.contrast .== g, :]
            b = d[argmax(d.esjd_K), :]
            push!(rows, row("contrast", g, b.scale, 0, b.esjd_K))
        end
        best = rows[argmax([r.esjd_K for r in rows])].setting
        push!(rows, row("best_contrast", best, NaN, 0, NaN))
    end
    for (grp, kind) in (("two_anchor", "two_anchor"), ("pilot", "pilot"))
        grp in TUNE_GROUPS || continue
        d = by[by.group .== grp, :]
        b = d[argmax(d.esjd_K), :]
        push!(rows, row(kind, kind, b.ta_k, b.ta_attempts, b.esjd_K; tilt = b.tilt, p_cross = b.p_cross))
    end
    new = DataFrame(rows)
    merge_write(TUNED_FILE, new; key = :kind)
    sort!(by, :label)
    for r in eachrow(by)
        @printf("  %-40s ESJD(K) %.5f  ESS(K)/s %7.3f  coupled acc %.4f  valid splits %.3f  split acc %.4f  TA acc/1k %6.2f\n",
                r.label, r.esjd_K, r.ess_K_per_sec, r.cb_acc, r.ta_split_valid_rate, r.ta_split_acc, r.ta_acc_per_ksweep)
    end
    println("\nSelected:"); show(new; allrows = true); println()
end

STAGE == "tune" && (run_tune(); exit())

# Screen, confirm and Poisson stages

isfile(TUNED_FILE) || error("no $TUNED_FILE; run --stage tune first")
const TUNED = CSV.read(TUNED_FILE, DataFrame)
const Q = STAGE == "poisson" ? poisson_q() : skew_q()
CONFIGS = study_configs(Q, TUNED; collapsed = STAGE == "poisson")
# --configs: confirm keeps the three baselines too; screen and cost run only these and merge their rows
const SUBSET = arg_value("--configs", "") != ""
if SUBSET && STAGE in ("screen", "confirm", "cost")
    want = Set(strip.(split(arg_value("--configs", ""), ";")))
    keep_base = STAGE == "confirm" ? ("RJ", "RJ pool", "Gibbs m=3") : ()
    CONFIGS = [c for c in CONFIGS if c.label in want || c.label in keep_base]
    length(CONFIGS) >= length(want) || error("unknown config in --configs; known: $(join([c.label for c in study_configs(Q, TUNED)], "; "))")
end
const write_rows = (file, df) -> SUBSET && STAGE != "confirm" ? merge_write(file, df) : (CSV.write(file, df); df)
const SCENS = scenarios()
const STAGE_STARTS = STAGE == "poisson" ? [:prior_a, :one_cluster] : STARTS

if STAGE == "cost"
    nworkers() == 1 || @warn "cost stage with $(nworkers()) workers: timings share the machine"
    ccfgs = [make_run(c, sc, :prior_a, Q; n_samples = 0, save_chain = false) for c in CONFIGS for sc in SCENS]
    println("Cost: $(length(ccfgs)) configs × data sets, $(N_COST) sweeps after $(N_COST_WARM)")
    cost = DataFrame(pmap(c -> cost_time(c, N_COST_WARM, N_COST), ccfgs))
    write_rows(joinpath(OUTDIR, "cost_timing.csv"), cost)
    for r in eachrow(cost); @printf("  %-36s %-12s %8.2f s/1k\n", r.label, r.scenario, 1000r.sec_per_iter); end
    exit()
end

const PILOT_FILE = CLEAN ? joinpath(ROOT, "cost", "cost_timing.csv") : joinpath(OUTDIR, "pilot_timing.csv")
CLEAN && !isfile(PILOT_FILE) && error("--clean needs $PILOT_FILE; run --stage cost first")
if !CLEAN && (FORCE_PILOT || !isfile(PILOT_FILE))
    pcfgs = [make_run(c, sc, :prior_a, Q; n_samples = N_PILOT, save_chain = false) for c in CONFIGS for sc in SCENS]
    println("Pilot: $(length(pcfgs)) runs of $(N_PILOT) iterations")
    CSV.write(PILOT_FILE, DataFrame(pmap(pilot_time, pcfgs)))
end
const PILOT = CSV.read(PILOT_FILE, DataFrame)

function chain_length(label, scenario, rep)
    r = PILOT[(PILOT.label .== label) .& (PILOT.scenario .== scenario) .& (PILOT.rep .== (CLEAN ? 1 : rep)), :]
    nrow(r) == 1 || error("no pilot timing for $label / $scenario / $rep; rerun with --pilot")
    n = max(N_MIN, round(Int, BUDGET / r.sec_per_iter[1]))
    thin = max(1, cld(n, MAX_STORED))
    return (n_samples = n, thin = thin, burn = round(Int, BURN_FRAC * cld(n, thin)))
end

runs = [make_run(c, sc, st, Q; chain_length(c.label, sc.name, sc.rep)...)
        for c in CONFIGS for sc in SCENS for st in STAGE_STARTS]
println("Stage $(STAGE): $(length(CONFIGS)) configs × $(length(SCENS)) data sets × $(length(STAGE_STARTS)) starts = $(length(runs)) chains on $(nworkers()) workers")
df = DataFrame(pmap(run_and_save, runs))
df.timing .= CLEAN ? "clean" : "pilot"
write_rows(joinpath(OUTDIR, "chains_$(STAGE).csv"), df)

@printf("\n%-36s %-12s %8s %9s %9s %8s %8s %8s\n", "config", "scenario", "s/1k", "ESS(K)/s", "birth acc", "coupled", "TA valid", "mean K")
for g in groupby(df, [:label, :scenario]; sort = false)
    @printf("%-36s %-12s %8.2f %9.3f %9.4f %8.4f %8.3f %8.2f\n", g.label[1], g.scenario[1], mean(g.sec_per_1k),
            mean(g.ess_K_per_sec), mean(g.acc_birth), mean(skipmissing(g.cb_acc)),
            sum(g.ta_split_valid) / max(1, sum(g.ta_split_attempts)), mean(g.mean_K))
end
println("\nWrote $(joinpath(OUTDIR, "chains_$(STAGE).csv"))")
