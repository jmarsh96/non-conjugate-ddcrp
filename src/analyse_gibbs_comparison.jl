# Tables, numbers and figure for the auxiliary Gibbs vs RJ comparison (RJ pool: Metropolised
# Algorithm 8). The reference is collapsed Gibbs for Poisson and all chains pooled otherwise.
# Copies into docs/ddcrp-paper/ unless --test.
#
#   julia --project=. src/analyse_gibbs_comparison.jl [--test]

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, StatsPlots, DataFrames, CSV, Statistics, StatsBase, JLD2, Printf, Random

include(joinpath(@__DIR__, "dengue_common.jl"))   # split_rhat, dahl_point_estimate

ENV["GKSwstype"] = "nul"   # headless GR

const IS_TEST = "--test" in ARGS
const ROOT = joinpath(@__DIR__, "..", "results", IS_TEST ? "gibbs_comparison_test" : "gibbs_comparison")
const PAPER = joinpath(@__DIR__, "..", "docs", "ddcrp-paper")
const STUDIES = ["skewnormal", "poisson", "dengue", "old_faithful"]
const N_PAIRS = 500
const PSM_BAND = (0.1, 0.9)

# Panel titles and LaTeX macro keys
const PANEL = Dict(("skewnormal", "overlapping") => ("Skew-normal, overlapping", "SkewOverlapping"),
                   ("skewnormal", "separated") => ("Skew-normal, separated", "SkewSeparated"),
                   ("poisson", "overlapping") => ("Poisson", "Poisson"),
                   ("dengue", "parana_2024") => ("Dengue", "Dengue"),
                   ("old_faithful", "faithful") => ("Old Faithful", "Faithful"))

load_c(dir, f) = jldopen(io -> io["c"], joinpath(dir, "chains", f), "r")
labels_of(c) = compute_table_assignments(c)

# Binder loss: pairs on which z and ẑ disagree
function binder(z::Vector{Int}, zhat::Vector{Int})
    a = countmap(z); b = countmap(zhat); ab = countmap(collect(zip(z, zhat)))
    return (sum(abs2, values(a)) + sum(abs2, values(b)) - 2 * sum(abs2, values(ab))) / 2
end

function chain_psm(c)
    n = size(c, 2); P = zeros(n, n)
    for r in axes(c, 1)
        z = labels_of(c[r, :])
        @inbounds for j in 1:n, i in 1:n
            P[i, j] += z[i] == z[j]
        end
    end
    return P ./ size(c, 1)
end

# effective_sample_size's estimator (Geyer initial positive, lags ≤ min(n/2, 1000)) without its
# per-lag allocations, which make 500 pairs × 224 chains take a day
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

tv(p, q) = 0.5 * sum(abs.(p .- q))
pk_vec(k, kmax) = [mean(k .== j) for j in 1:kmax]

# Per-study analysis

function analyse_study(study)
    dir = joinpath(ROOT, study)
    file = joinpath(dir, "chains_$(study).csv")
    isfile(file) || return nothing
    df = CSV.read(file, DataFrame)
    out = DataFrame[]
    traces = Dict{Tuple{String,String},Tuple{Vector{Int},Vector{Float64}}}()
    for scen in unique(df.scenario)
        d = df[df.scenario .== scen, :]
        println("$study / $scen: $(nrow(d)) chains"); flush(stdout)
        psm =Dict{String,Matrix{Float64}}(); ktr = Dict{String,Vector{Int}}()
        for r in eachrow(d)
            c = load_c(dir, r.chain_file)
            psm[r.chain_file] = chain_psm(c)
            ktr[r.chain_file] = calculate_n_clusters(c)
        end
        is_ref = "collapsed" in d.sampler ? d.sampler .== "collapsed" : trues(nrow(d))
        ref_files = d.chain_file[is_ref]
        P_ref = mean(psm[f] for f in ref_files)
        kmax = maximum(maximum(v) for v in values(ktr))
        pk_ref = mean(pk_vec(ktr[f], kmax) for f in ref_files)

        # Reference Dahl partition
        best_z, best_loss = Int[], Inf
        for f in ref_files
            c = load_c(dir, f)
            z = dahl_point_estimate(c, P_ref; max_draws = 200)
            loss = sum(((z[i] == z[j]) - P_ref[i, j])^2 for i in eachindex(z), j in eachindex(z) if i < j)
            loss < best_loss && (best_z = z; best_loss = loss)
        end

        n = size(P_ref, 1)
        pairs = [(i, j) for j in 1:n for i in 1:(j-1) if PSM_BAND[1] <= P_ref[i, j] <= PSM_BAND[2]]
        length(pairs) > N_PAIRS && (pairs = pairs[sort(randperm(MersenneTwister(1), length(pairs))[1:N_PAIRS])])

        for (idx, r) in enumerate(eachrow(d))
            c = load_c(dir, r.chain_file)
            Z = similar(c)
            for s in axes(c, 1)
                Z[s, :] = labels_of(c[s, :])
            end
            ess_pairs = isempty(pairs) ? Float64[] :
                [ess(Float64.(Z[:, i] .== Z[:, j])) for (i, j) in pairs]
            ess_pairs = filter(isfinite, ess_pairs)
            b = [binder(collect(Z[s, :]), best_z) for s in axes(Z, 1)]
            Δ = abs.(psm[r.chain_file] .- P_ref)
            upper = [Δ[i, j] for j in 1:n for i in 1:(j-1)]
            push!(out, DataFrame(
                chain_file = r.chain_file, scenario = scen, n_pairs = length(pairs),
                ess_pairs_med = isempty(ess_pairs) ? NaN : median(ess_pairs),
                ess_pairs_p10 = isempty(ess_pairs) ? NaN : quantile(ess_pairs, 0.1),
                ess_binder = std(b) > 0 ? ess(b) : NaN,
                psm_mad_ref = mean(upper), psm_max_ref = maximum(upper),
                tv_K_ref = tv(pk_vec(ktr[r.chain_file], kmax), pk_ref),
            ))
        end
        for f in d.chain_file
            traces[(scen, f)] = (ktr[f], jldopen(io -> io["logpost"], joinpath(dir, "chains", f), "r"))
        end
    end
    chains = innerjoin(df, vcat(out...), on = [:chain_file, :scenario])
    chains.ess_pairs_per_sec = chains.ess_pairs_med ./ chains.total_time
    chains.ess_binder_per_sec = chains.ess_binder ./ chains.total_time
    CSV.write(joinpath(ROOT, "chain_metrics_$(study).csv"), chains)

    summ = combine(groupby(chains, [:label, :sampler, :q, :m, :link, :fdim, :scenario]; sort = false)) do g
        ks = [traces[(g.scenario[1], f)][1] for f in g.chain_file]
        lps = [traces[(g.scenario[1], f)][2] for f in g.chain_file]
        (n_starts = nrow(g), n_iter = mean(g.n_iter), sec_per_1k = mean(g.sec_per_1k),
         total_time = mean(g.total_time), mean_K = mean(g.mean_K),
         ess_K = mean(g.ess_K), ess_K_per_sec = mean(g.ess_K_per_sec),
         ess_K_per_sec_min = minimum(g.ess_K_per_sec),
         ess_pairs_per_sec = mean(g.ess_pairs_per_sec), ess_binder_per_sec = mean(g.ess_binder_per_sec),
         rhat_K = split_rhat([Float64.(k) for k in ks]), rhat_logpost = split_rhat(lps),
         tv_K_ref = mean(g.tv_K_ref), psm_mad_ref = mean(g.psm_mad_ref), psm_max_ref = maximum(g.psm_max_ref),
         acc_overall = mean(g.acc_overall))
    end
    summ.study .= study
    CSV.write(joinpath(ROOT, "summary_$(study).csv"), summ)
    return summ
end

summaries = filter(!isnothing, [analyse_study(s) for s in STUDIES])
isempty(summaries) && error("no chains_<study>.csv under $ROOT; run run_gibbs_comparison.jl first")
S = vcat(summaries...; cols = :union)

# Best per panel by ESS(K)/s

fmt(x, d) = isfinite(x) ? Printf.format(Printf.Format("%.$(d)f"), x) : "--"
texlabel(l) = replace(l, "q:mm" => "q:MM", "link:" => "link ", "m=" => "\$m=\$")

panels = [(st, sc) for (st, sc) in keys(PANEL) if any((S.study .== st) .& (S.scenario .== sc))]
sort!(panels, by = p -> (findfirst(==(p[1]), STUDIES), p[2]))

best_of(d, samplers, col) = (x = d[in.(d.sampler, Ref(samplers)), :]; nrow(x) == 0 ? nothing : x[argmax(x[!, col]), :])
const RJ_ANY = ("rj", "aux_mh")

macros = Pair{String,String}[]
compact = String[]
for (st, sc) in panels
    d = S[(S.study .== st) .& (S.scenario .== sc), :]
    name, key = PANEL[(st, sc)]
    rj = best_of(d, RJ_ANY, :ess_K_per_sec); gb = best_of(d, ("aux_gibbs",), :ess_K_per_sec)
    rjc = best_of(d, RJ_ANY, :ess_pairs_per_sec); gbc = best_of(d, ("aux_gibbs",), :ess_pairs_per_sec)
    rj0 = best_of(d, ("rj",), :ess_K_per_sec)
    (isnothing(rj) || isnothing(gb) || isnothing(rj0)) && continue
    rK = gb.ess_K_per_sec / rj.ess_K_per_sec
    rC = gbc.ess_pairs_per_sec / rjc.ess_pairs_per_sec
    append!(macros, [
        "gc$(key)BestRJ" => texlabel(rj.label), "gc$(key)BestGibbs" => texlabel(gb.label),
        "gc$(key)RJessKps" => fmt(rj.ess_K_per_sec, 2), "gc$(key)GibbsessKps" => fmt(gb.ess_K_per_sec, 2),
        "gc$(key)RatioK" => fmt(rK, 2), "gc$(key)RatioCoclust" => fmt(rC, 2),
        "gc$(key)CostRatio" => fmt(gb.sec_per_1k / rj.sec_per_1k, 1),
        "gc$(key)RJsecPerK" => fmt(rj.sec_per_1k, 2), "gc$(key)GibbssecPerK" => fmt(gb.sec_per_1k, 2),
        "gc$(key)MaxRhatK" => fmt(maximum(filter(isfinite, d.rhat_K); init = NaN), 3),
        "gc$(key)MaxTVK" => fmt(maximum(d.tv_K_ref), 3),
        "gc$(key)BestRJorig" => texlabel(rj0.label), "gc$(key)RJorigessKps" => fmt(rj0.ess_K_per_sec, 2),
        "gc$(key)RJgainK" => fmt(rj.ess_K_per_sec / rj0.ess_K_per_sec, 2),
    ])
    push!(compact, "$name & $(fmt(rj0.ess_K_per_sec, 2)) & $(texlabel(rj.label)) & $(fmt(rj.sec_per_1k, 2)) & $(fmt(rj.ess_K_per_sec, 2)) & " *
          "$(fmt(rj.ess_pairs_per_sec, 2)) & $(texlabel(gb.label)) & $(fmt(gb.sec_per_1k, 2)) & " *
          "$(fmt(gb.ess_K_per_sec, 2)) & $(fmt(gb.ess_pairs_per_sec, 2)) & $(fmt(rK, 2)) \\\\")
end

# Tables and numbers
open(joinpath(ROOT, "table_gibbs_vs_rj.tex"), "w") do io
    println(io, "% Generated by analyse_gibbs_comparison.jl; best config per sampler by ESS(K)/s, means over starts.")
    println(io, "% orig.: best ESS(K)/s of the reversible jump code as published; Config: best of it and RJ pool.")
    println(io, "\\begin{tabular}{l r lrrr lrrr r}\n\\toprule")
    println(io, " & \\multicolumn{5}{c}{Reversible jump} & \\multicolumn{4}{c}{Auxiliary Gibbs} & \\\\")
    println(io, "\\cmidrule(lr){2-6}\\cmidrule(lr){7-10}")
    println(io, "Study & orig. & Config & s/1k & ESS\$(K)\$/s & co-cl./s & Config & s/1k & ESS\$(K)\$/s & co-cl./s & Gibbs/RJ \\\\\n\\midrule")
    foreach(l -> println(io, l), compact)
    println(io, "\\bottomrule\n\\end{tabular}")
end

open(joinpath(ROOT, "table_gibbs_vs_rj_full.tex"), "w") do io
    println(io, "% Generated by analyse_gibbs_comparison.jl; means over starts.")
    println(io, "\\begin{tabular}{ll rrrrrrr}\n\\toprule")
    println(io, "Study & Sampler & s/1k & ESS\$(K)\$ & ESS\$(K)\$/s & co-cl./s & Binder/s & \$\\hat R(K)\$ & TV\$(K)\$ \\\\\n\\midrule")
    for (pi, (st, sc)) in enumerate(panels)
        d = S[(S.study .== st) .& (S.scenario .== sc), :]
        pi > 1 && println(io, "\\addlinespace")
        for (ri, r) in enumerate(eachrow(d))
            println(io, (ri == 1 ? PANEL[(st, sc)][1] : "") * " & $(texlabel(r.label)) & $(fmt(r.sec_per_1k, 2)) & " *
                    "$(fmt(r.ess_K, 0)) & $(fmt(r.ess_K_per_sec, 2)) & $(fmt(r.ess_pairs_per_sec, 2)) & " *
                    "$(fmt(r.ess_binder_per_sec, 2)) & $(fmt(r.rhat_K, 3)) & $(fmt(r.tv_K_ref, 3)) \\\\")
        end
    end
    println(io, "\\bottomrule\n\\end{tabular}")
end

# RJ tuning ladder: ESS(K)/s as each choice is added, best Gibbs alongside
const LADDER = [
    "untuned" => ["RJ q:prior link:uniform + Resample", "RJ q:prior link:uniform", "RJ q:prior link:prior"],
    "+ tuned birth" => ["RJ q:mm link:uniform"],
    "+ prior link" => ["RJ q:mm link:prior"],
    "+ moving-set scoring" => ["RJ pool q:mm link:prior m=1"],
]
open(joinpath(ROOT, "table_rj_ladder.tex"), "w") do io
    println(io, "% Generated by analyse_gibbs_comparison.jl; ESS(K)/s, means over starts. untuned: the published")
    println(io, "% Old Faithful sampler, else prior birth with uniform links (dengue: prior links).")
    println(io, "\\begin{tabular}{l rrrrr r}\n\\toprule")
    println(io, "Study & " * join(first.(LADDER), " & ") * " & + pool (best) & Gibbs (best) \\\\\n\\midrule")
    for (st, sc) in panels
        d = S[(S.study .== st) .& (S.scenario .== sc), :]
        val(lbls) = (i = findfirst(l -> l in d.label, lbls); isnothing(i) ? NaN : d.ess_K_per_sec[d.label .== lbls[i]][1])
        pool = best_of(d, ("aux_mh",), :ess_K_per_sec); gb = best_of(d, ("aux_gibbs",), :ess_K_per_sec)
        cells = vcat([fmt(val(l), 2) for l in last.(LADDER)],
                     isnothing(pool) ? "--" : fmt(pool.ess_K_per_sec, 2), isnothing(gb) ? "--" : fmt(gb.ess_K_per_sec, 2))
        println(io, PANEL[(st, sc)][1] * " & " * join(cells, " & ") * " \\\\")
    end
    println(io, "\\bottomrule\n\\end{tabular}")
end

# New moves of the sampler study against their base configs: ESS(K)/s and co-clustering ESS/s ratios
const NEW_MOVES = ["RJ q:mm link:informed" => "RJ q:mm link:prior",
                   "RJ pool q:mm link:prior m=1 + two-anchor" => "RJ pool q:mm link:prior m=1",
                   "Gibbs q:mm m=3 + two-anchor" => "Gibbs q:mm m=3"]
if any(in(S.label), first.(NEW_MOVES))
    open(joinpath(ROOT, "table_new_moves.tex"), "w") do io
        println(io, "% Generated by analyse_gibbs_comparison.jl; each new move over its base config:")
        println(io, "% ESS(K)/s ratio / co-clustering ESS/s ratio, means over starts.")
        println(io, "\\begin{tabular}{l " * repeat("r", length(NEW_MOVES)) * "}\n\\toprule")
        println(io, "Study & " * join(["$(texlabel(n)) vs base" for n in first.(NEW_MOVES)], " & ") * " \\\\\n\\midrule")
        for (st, sc) in panels
            d = S[(S.study .== st) .& (S.scenario .== sc), :]
            get1(l, col) = (r = d[d.label .== l, col]; isempty(r) ? NaN : r[1])
            cells = [fmt(get1(n, :ess_K_per_sec) / get1(b, :ess_K_per_sec), 2) * " / " *
                     fmt(get1(n, :ess_pairs_per_sec) / get1(b, :ess_pairs_per_sec), 2) for (n, b) in NEW_MOVES]
            println(io, PANEL[(st, sc)][1] * " & " * join(cells, " & ") * " \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}")
    end
end

open(joinpath(ROOT, "gibbs_comparison_numbers.tex"), "w") do io
    println(io, "% Generated by analyse_gibbs_comparison.jl")
    for (k, v) in macros
        println(io, "\\newcommand{\\$k}{$v}")
    end
end

# Figure

const C_GIBBS = Dict("mm" => colorant"#2a78d6", "prior" => colorant"#eb6834")
const POOL_STYLE = [(("mm", "prior"), colorant"#b0233c", :diamond),
                    (("mm", "metropolised"), colorant"#00938f", :utriangle),
                    (("prior", "metropolised"), colorant"#7a5c00", :dtriangle)]
poolname(q, link) = "RJ pool q:$(q) link:$(link)"
const RJ_STYLE = Dict(
    "RJ q:mm link:uniform" => (colorant"#1baf7a", :dash),
    "RJ q:mm link:prior" => (colorant"#eda100", :dot),
    "RJ q:prior link:uniform" => (colorant"#e87ba4", :dashdot),
    "RJ q:prior link:prior" => (colorant"#008300", :dashdotdot),
    "RJ q:prior link:uniform + Resample" => (colorant"#6250d6", :dash),
    "Collapsed Gibbs" => (colorant"#0b0b0b", :solid),
)

function panel(d, col, title, ylab)
    p = plot(xscale = :log10, yscale = :log10, xticks = ([1, 3, 10, 30], ["1", "3", "10", "30"]),
             title = title, titlefontsize = 9, xlabel = "pool size m", ylabel = ylab,
             guidefontsize = 8, tickfontsize = 7, legend = false,
             yformatter = y -> string(round(y; sigdigits = 2)),
             gridalpha = 0.15, framestyle = :axes)
    for r in eachrow(d[in.(d.sampler, Ref(("rj", "collapsed"))), :])
        haskey(RJ_STYLE, r.label) || continue
        cl, ls = RJ_STYLE[r.label]
        hline!(p, [r[col]]; color = cl, linestyle = ls, lw = 2, label = r.label)
    end
    for q in ("mm", "prior")
        g = sort(d[(d.sampler .== "aux_gibbs") .& (d.q .== q), :], :m)
        nrow(g) == 0 && continue
        plot!(p, g.m, g[!, col]; color = C_GIBBS[q], lw = 2, marker = q == "mm" ? :circle : :rect,
              markersize = 5, markerstrokecolor = :white, markerstrokewidth = 1.5, label = "Gibbs q:$(q)")
    end
    for ((q, link), cl, mk) in POOL_STYLE
        g = sort(d[(d.sampler .== "aux_mh") .& (d.q .== q) .& (d.link .== link), :], :m)
        nrow(g) == 0 && continue
        plot!(p, g.m, g[!, col]; color = cl, lw = 2, linestyle = :dash, marker = mk, markersize = 5,
              markerstrokecolor = :white, markerstrokewidth = 1.5, label = poolname(q, link))
    end
    return p
end

plots_K = [panel(S[(S.study .== st) .& (S.scenario .== sc), :], :ess_K_per_sec, PANEL[(st, sc)][1],
                 i == 1 ? "ESS(K) / s" : "") for (i, (st, sc)) in enumerate(panels)]
plots_C = [panel(S[(S.study .== st) .& (S.scenario .== sc), :], :ess_pairs_per_sec, "",
                 i == 1 ? "co-clustering ESS / s" : "") for (i, (st, sc)) in enumerate(panels)]

leg = plot(framestyle = :none, legend = :left, legendfontsize = 7,
           foreground_color_legend = nothing, background_color_legend = nothing)
for q in ("mm", "prior")
    plot!(leg, [NaN], [NaN]; color = C_GIBBS[q], lw = 2, marker = q == "mm" ? :circle : :rect,
          markersize = 5, markerstrokecolor = :white, label = "Gibbs q:$(q)")
end
for ((q, link), cl, mk) in POOL_STYLE
    any((S.sampler .== "aux_mh") .& (S.q .== q) .& (S.link .== link)) || continue
    plot!(leg, [NaN], [NaN]; color = cl, lw = 2, linestyle = :dash, marker = mk, markersize = 5,
          markerstrokecolor = :white, label = poolname(q, link))
end
for (lbl, (cl, ls)) in sort(collect(RJ_STYLE); by = first)
    lbl in S.label && plot!(leg, [NaN], [NaN]; color = cl, linestyle = ls, lw = 2, label = lbl)
end

np = length(panels)
fig = plot(plots_K..., leg, plots_C..., plot(framestyle = :none);
           layout = grid(2, np + 1; widths = vcat(fill(0.82 / np, np), 0.18)),
           size = (290 * np + 260, 580), left_margin = 9StatsPlots.mm, bottom_margin = 5StatsPlots.mm)
savefig(fig, joinpath(ROOT, "gibbs_vs_rj_ess.png"))

println("\nBest per panel (ESS(K)/s):")
for (k, v) in macros
    occursin(r"Best|Ratio", k) && @printf("  %-28s %s\n", k, v)
end

# Paper copies

if IS_TEST
    println("\n--test: assets left in $ROOT (docs/ddcrp-paper/ untouched).")
elseif !isdir(PAPER)
    println("\nNo paper directory at $PAPER (docs/ is git-ignored): assets left in $ROOT.")
else
    for f in ("table_gibbs_vs_rj.tex", "table_gibbs_vs_rj_full.tex", "table_rj_ladder.tex", "gibbs_comparison_numbers.tex")
        cp(joinpath(ROOT, f), joinpath(PAPER, "tables", f); force = true)
    end
    cp(joinpath(ROOT, "gibbs_vs_rj_ess.png"), joinpath(PAPER, "figures", "gibbs_vs_rj_ess.png"); force = true)
    println("\nPaper assets copied to $PAPER/{figures,tables}/")
end
