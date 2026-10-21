# Tables and figure for the sampler study (run_sampler_study.jl) and the birth oracle diagnostic
# (diagnose_births.jl). Per data set the reference is all chains pooled (collapsed Gibbs for the
# Poisson stage). Partition regions are the Dahl partitions of each K with posterior mass ≥ 0.05;
# a draw belongs to the nearest by variation of information, and a region switch is counted
# when a chain enters a different region and stays for at least SWITCH_RUN stored draws.
#
#   julia --project=. src/analyse_sampler_study.jl --stage {screen|confirm|poisson} [--test]

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, StatsPlots, DataFrames, CSV, Statistics, StatsBase, JLD2, Printf, Random

include(joinpath(@__DIR__, "dengue_common.jl"))   # split_rhat, dahl_point_estimate

ENV["GKSwstype"] = "nul"

arg_value(flag, default) = (i = findfirst(==(flag), ARGS); isnothing(i) ? default : ARGS[i+1])
const STAGE = arg_value("--stage", "screen")
const IS_TEST = "--test" in ARGS
const ROOT = joinpath(@__DIR__, "..", "results", IS_TEST ? "sampler_study_test" : "sampler_study")
const DIR = joinpath(ROOT, STAGE)
const N_PAIRS = 500
const PSM_BAND = (0.1, 0.9)
const SWITCH_RUN = 5
const N_REF_DRAWS = 300

load_c(f) = jldopen(io -> io["c"], joinpath(DIR, "chains", f), "r")
labels_of(c) = compute_table_assignments(c)

# effective_sample_size's estimator without its per-lag allocations
function ess(x::AbstractVector{<:Real})
    n = length(x); v = var(x)
    v ≈ 0 && return effective_sample_size(x)
    xc = x .- mean(x)
    function acf(k)
        s = 0.0
        @inbounds @simd for t in 1:(n-k)
            s += xc[t] * xc[t+k]
        end
        return s / ((n - k) * v)
    end
    τ = 1.0
    for k in 1:2:(min(n ÷ 2, 1000) - 1)
        g = acf(k) + acf(k + 1)
        g < 0 && break
        τ += 2g
    end
    return n / τ
end

function binder(z::Vector{Int}, zhat::Vector{Int})
    a = countmap(z); b = countmap(zhat); ab = countmap(collect(zip(z, zhat)))
    return (sum(abs2, values(a)) + sum(abs2, values(b)) - 2 * sum(abs2, values(ab))) / 2
end

function vi(z1::Vector{Int}, z2::Vector{Int})
    n = length(z1)
    h(c) = -sum(v / n * log(v / n) for v in values(c))
    return 2 * h(countmap(collect(zip(z1, z2)))) - h(countmap(z1)) - h(countmap(z2))
end

function label_matrix(c)
    Z = similar(c)
    for s in axes(c, 1)
        Z[s, :] = labels_of(c[s, :])
    end
    return Z
end

function psm(Z)
    n = size(Z, 2); P = zeros(n, n)
    for s in axes(Z, 1)
        z = view(Z, s, :)
        @inbounds for j in 1:n, i in 1:n
            P[i, j] += z[i] == z[j]
        end
    end
    return P ./ size(Z, 1)
end

tv(p, q) = 0.5 * sum(abs.(p .- q))
finite_mean(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
pk_vec(k, kmax) = [mean(k .== j) for j in 1:kmax]

# Reference regions: Dahl partition of each K with mass ≥ 0.05 among pooled draws
function regions(Zs)
    pooled = vcat([Z[round.(Int, range(1, size(Z, 1); length = min(size(Z, 1), N_REF_DRAWS))), :] for Z in Zs]...)
    Ks = [length(unique(view(pooled, s, :))) for s in axes(pooled, 1)]
    refs = Vector{Int}[]
    for k in sort(unique(Ks))
        idx = findall(==(k), Ks)
        length(idx) / length(Ks) < 0.05 && continue
        sub = pooled[idx, :]
        P = psm(sub)
        loss(s) = sum(((sub[s, i] == sub[s, j]) - P[i, j])^2 for j in axes(P, 2) for i in 1:(j-1))
        push!(refs, collect(sub[argmin(loss.(1:size(sub, 1))), :]))
    end
    return refs
end

function switches(Z, refs)
    length(refs) < 2 && return 0
    r = [argmin([vi(collect(view(Z, s, :)), ref) for ref in refs]) for s in axes(Z, 1)]
    runs = [(first(g), length(g)) for g in (r[i:j] for (i, j) in _runs(r))]
    kept = [v for (v, l) in runs if l >= SWITCH_RUN]
    return count(i -> kept[i] != kept[i-1], 2:length(kept))
end
function _runs(r)
    out = Tuple{Int,Int}[]; i = 1
    while i <= length(r)
        j = i
        while j < length(r) && r[j+1] == r[i]; j += 1; end
        push!(out, (i, j)); i = j + 1
    end
    return out
end

# Per data set

df = CSV.read(joinpath(DIR, "chains_$(STAGE).csv"), DataFrame)
out = DataFrame[]
for g in groupby(df, [:scenario, :rep])
    Zs = Dict(f => label_matrix(load_c(f)) for f in g.chain_file)
    ktr = Dict(f => [length(unique(view(Z, s, :))) for s in axes(Z, 1)] for (f, Z) in Zs)
    ref_files = "collapsed" in g.base ? g.chain_file[g.base .== "collapsed"] : g.chain_file
    P_ref = mean(psm(Zs[f]) for f in ref_files)
    kmax = maximum(maximum(v) for v in values(ktr))
    pk_ref = mean(pk_vec(ktr[f], kmax) for f in ref_files)
    best_z = labels_of(load_c(ref_files[1])[1, :]); best_loss = Inf
    for f in ref_files
        z = dahl_point_estimate(load_c(f), P_ref; max_draws = 200)
        l = sum(((z[i] == z[j]) - P_ref[i, j])^2 for j in eachindex(z) for i in 1:(j-1))
        l < best_loss && (best_z = z; best_loss = l)
    end
    refs = regions([Zs[f] for f in g.chain_file])
    n = size(P_ref, 1)
    pairs = [(i, j) for j in 1:n for i in 1:(j-1) if PSM_BAND[1] <= P_ref[i, j] <= PSM_BAND[2]]
    length(pairs) > N_PAIRS && (pairs = pairs[sort(randperm(MersenneTwister(1), length(pairs))[1:N_PAIRS])])
    for r in eachrow(g)
        Z = Zs[r.chain_file]
        ep = filter(isfinite, [ess(Float64.(view(Z, :, i) .== view(Z, :, j))) for (i, j) in pairs])
        b = [binder(collect(view(Z, s, :)), best_z) for s in axes(Z, 1)]
        Δ = abs.(psm(Z) .- P_ref)
        push!(out, DataFrame(chain_file = r.chain_file, scenario = r.scenario, rep = r.rep,
            n_regions = length(refs), region_switches = switches(Z, refs),
            ess_pairs = isempty(ep) ? NaN : median(ep), ess_binder = std(b) > 0 ? ess(b) : NaN,
            psm_max_ref = maximum(Δ[i, j] for j in 1:n for i in 1:(j-1)),
            tv_K_ref = tv(pk_vec(ktr[r.chain_file], kmax), pk_ref)))
    end
end
chains = innerjoin(df, vcat(out...), on = [:chain_file, :scenario, :rep])
chains.ess_pairs_per_sec = chains.ess_pairs ./ chains.total_time
chains.ess_binder_per_sec = chains.ess_binder ./ chains.total_time
chains.switches_per_1000s = 1000 .* chains.region_switches ./ chains.total_time
# Per 1000 sweeps, and rescaled by the clean cost per sweep of --stage cost when it exists (same
# data set, rep 1): robust to chains that shared a machine with uneven cores
for (col, src) in ((:ess_K, :ess_K), (:ess_pairs, :ess_pairs), (:switches, :region_switches))
    chains[!, Symbol(col, :_per_ksweep)] = 1000 .* chains[!, src] ./ chains.n_iter
end
# Only chains sized by --clean are rescaled (the cost was measured on the same machine)
const COST_FILE = joinpath(ROOT, "cost", "cost_timing.csv")
const HAS_COST = isfile(COST_FILE) && "timing" in names(chains) && all(==("clean"), chains.timing)
if HAS_COST
    cost = CSV.read(COST_FILE, DataFrame)
    clean = Dict((r.label, r.scenario) => r.sec_per_iter for r in eachrow(cost))
    chains.clean_sec_per_1k = [1000 * get(clean, (r.label, r.scenario), NaN) for r in eachrow(chains)]
else
    chains.clean_sec_per_1k = chains.sec_per_1k
end
chains.ess_K_per_sec_clean = chains.ess_K_per_ksweep ./ chains.clean_sec_per_1k
chains.ess_pairs_per_sec_clean = chains.ess_pairs_per_ksweep ./ chains.clean_sec_per_1k
chains.switches_per_1000s_clean = 1000 .* chains.switches_per_ksweep ./ chains.clean_sec_per_1k
CSV.write(joinpath(DIR, "chain_metrics_$(STAGE).csv"), chains)

summ = combine(groupby(chains, [:label, :base, :contrast, :link, :ta_k, :scenario, :rep]; sort = false)) do g
    lp = [jldopen(io -> io["logpost"], joinpath(DIR, "chains", f), "r") for f in g.chain_file]
    ks = [Float64.(calculate_n_clusters(load_c(f))) for f in g.chain_file]
    (n_starts = nrow(g), sec_per_1k = mean(g.sec_per_1k), mean_K = mean(g.mean_K),
     ess_K_per_sec = mean(g.ess_K_per_sec), ess_pairs_per_sec = mean(g.ess_pairs_per_sec),
     ess_binder_per_sec = mean(g.ess_binder_per_sec), switches_per_1000s = mean(g.switches_per_1000s),
     clean_sec_per_1k = mean(g.clean_sec_per_1k), ess_K_per_ksweep = mean(g.ess_K_per_ksweep),
     ess_pairs_per_ksweep = mean(g.ess_pairs_per_ksweep), switches_per_ksweep = mean(g.switches_per_ksweep),
     ess_K_per_sec_clean = mean(g.ess_K_per_sec_clean), ess_pairs_per_sec_clean = mean(g.ess_pairs_per_sec_clean),
     switches_per_1000s_clean = mean(g.switches_per_1000s_clean),
     rhat_K = split_rhat(ks), rhat_logpost = split_rhat(lp),
     tv_K_ref = mean(g.tv_K_ref), psm_max_ref = maximum(g.psm_max_ref),
     acc_birth = mean(g.acc_birth), acc_death = mean(g.acc_death), coupled_acc = finite_mean(g.cb_acc),
     ta_valid_split = sum(g.ta_split_valid) / max(1, sum(g.ta_split_attempts)),
     ta_valid_merge = sum(g.ta_merge_valid) / max(1, sum(g.ta_merge_attempts)),
     ta_acc_split = sum(g.ta_split_accepts) / max(1, sum(g.ta_split_valid)),
     ta_acc_merge = sum(g.ta_merge_accepts) / max(1, sum(g.ta_merge_valid)),
     ta_split_ge2 = finite_mean(g.ta_split_ge2),
     ta_time_frac = mean(g.ta_time ./ g.total_time))
end
# Gain over the published RJ on the same data set
for col in (:ess_K_per_sec, :ess_pairs_per_sec, :switches_per_1000s, :ess_K_per_ksweep, :ess_pairs_per_ksweep,
            :ess_K_per_sec_clean, :ess_pairs_per_sec_clean)
    base = Dict((r.scenario, r.rep) => r[col] for r in eachrow(summ) if r.label == "RJ")
    summ[!, Symbol(col, :_vs_RJ)] = [r[col] / get(base, (r.scenario, r.rep), NaN) for r in eachrow(summ)]
end
CSV.write(joinpath(DIR, "summary_$(STAGE).csv"), summ)

# Across replicates
gm(x) = exp(mean(log.(x)))
agg = combine(groupby(summ, [:label, :scenario]; sort = false),
    :ess_K_per_sec_vs_RJ => gm => :gain_K_wall, :ess_pairs_per_sec_vs_RJ => gm => :gain_coclust_wall,
    :ess_K_per_ksweep_vs_RJ => gm => :gain_K_sweep, :ess_pairs_per_ksweep_vs_RJ => gm => :gain_coclust_sweep,
    :ess_K_per_sec_clean_vs_RJ => gm => :gain_K, :ess_pairs_per_sec_clean_vs_RJ => gm => :gain_coclust,
    :switches_per_1000s_clean => mean => :switches_per_1000s, :switches_per_ksweep => mean => :switches_per_ksweep,
    :clean_sec_per_1k => mean => :sec_per_1k, :rhat_K => maximum => :max_rhat_K, :tv_K_ref => maximum => :max_tv_K,
    :acc_birth => mean => :acc_birth, :coupled_acc => mean => :coupled_acc,
    :ta_valid_split => mean => :ta_valid_split, :ta_acc_split => mean => :ta_acc_split)
CSV.write(joinpath(DIR, "summary_by_scenario_$(STAGE).csv"), agg)

fmt(x, d) = isfinite(x) ? Printf.format(Printf.Format("%.$(d)f"), x) : "--"
scens = unique(agg.scenario)
labels = unique(agg.label)
open(joinpath(DIR, "table_sampler_study_$(STAGE).tex"), "w") do io
    println(io, "% Generated by analyse_sampler_study.jl; ESS(K)/s and co-clustering ESS/s relative to the")
    println(io, "% published RJ on the same data (geometric mean over replicates), with s per 1k iterations.")
    HAS_COST && println(io, "% Per-sweep ESS rescaled by the clean cost per sweep of --stage cost.")
    println(io, "\\begin{tabular}{l " * repeat("rrr ", length(scens)) * "}\n\\toprule")
    println(io, " & " * join(["\\multicolumn{3}{c}{$s}" for s in scens], " & ") * " \\\\")
    println(io, "Config & " * join(fill("s/1k & \$K\$ & co-cl.", length(scens)), " & ") * " \\\\\n\\midrule")
    for l in labels
        cells = String[]
        for s in scens
            r = agg[(agg.label .== l) .& (agg.scenario .== s), :]
            nrow(r) == 0 ? append!(cells, ["--", "--", "--"]) :
                append!(cells, [fmt(r.sec_per_1k[1], 2), fmt(r.gain_K[1], 2), fmt(r.gain_coclust[1], 2)])
        end
        println(io, "$l & " * join(cells, " & ") * " \\\\")
    end
    println(io, "\\bottomrule\n\\end{tabular}")
end

# Figure: gain over RJ by config, one panel per scenario
plots = map(scens) do s
    d = agg[agg.scenario .== s, :]
    y = length(labels):-1:1
    p = scatter(d.gain_K, [y[findfirst(==(l), labels)] for l in d.label]; xscale = :log10, label = "ESS(K)/s",
                color = colorant"#2a78d6", markersize = 5, markerstrokewidth = 0, title = s, titlefontsize = 9,
                yticks = (y, s == scens[1] ? labels : fill("", length(labels))), ytickfontsize = 6,
                xtickfontsize = 7, legend = s == scens[end] ? :bottomright : false, legendfontsize = 6,
                xlabel = "gain over RJ", guidefontsize = 7, gridalpha = 0.15)
    scatter!(p, d.gain_coclust, [y[findfirst(==(l), labels)] for l in d.label]; label = "co-clustering ESS/s",
             color = colorant"#eb6834", marker = :diamond, markersize = 5, markerstrokewidth = 0)
    vline!(p, [1.0]; color = :grey60, linestyle = :dash, label = false)
    p
end
fig = plot(plots...; layout = (1, length(scens)), size = (300 * length(scens) + 180, 120 + 16 * length(labels)),
           left_margin = 4StatsPlots.mm, bottom_margin = 5StatsPlots.mm)
savefig(fig, joinpath(DIR, "sampler_study_gain_$(STAGE).png"))

# Oracle diagnostic table, when present
diag_file = joinpath(ROOT, "diagnose_births_summary.csv")
if isfile(diag_file)
    dg = CSV.read(diag_file, DataFrame)
    open(joinpath(ROOT, "table_births_oracle.tex"), "w") do io
        println(io, "% Generated by analyse_sampler_study.jl from diagnose_births_summary.csv: birth acceptance")
        println(io, "% observed vs the bound A* of the exact conditional parameter proposal, weighted by birth probability.")
        println(io, "\\begin{tabular}{ll rrrr}\n\\toprule\nData & \$|M|\$ & moves & observed & \$A^*\$ & ratio \\\\\n\\midrule")
        for r in eachrow(dg)
            println(io, "$(r.scenario) & $(r.bin) & $(ismissing(r.n_moves) ? "" : r.n_moves) & " *
                        "$(fmt(r.acc_obs, 4)) & $(fmt(r.A_star, 4)) & $(fmt(r.ratio, 2)) \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}")
    end
end

println("\nGain over published RJ (geometric mean over replicates)$(HAS_COST ? ", clean cost" : ""); per sweep; wall clock:")
for r in eachrow(agg)
    @printf("  %-36s %-12s K %5.2f co-cl %5.2f | sweep K %5.2f co-cl %5.2f | wall K %5.2f | sw/1k %5.2f  s/1k %6.2f  R̂ %.3f  TV %.3f\n",
            r.label, r.scenario, r.gain_K, r.gain_coclust, r.gain_K_sweep, r.gain_coclust_sweep, r.gain_K_wall,
            r.switches_per_ksweep, r.sec_per_1k, r.max_rhat_K, r.max_tv_K)
end
println("\nWrote $(DIR)")
