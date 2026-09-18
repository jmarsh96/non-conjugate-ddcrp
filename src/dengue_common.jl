# ============================================================================
# dengue_common.jl
#
# Helpers shared by the dengue scripts (`run_dengue.jl`, `analyse_dengue.jl`,
# `dengue_sensitivity.jl`), so the contiguity graph, the hop-window prior and
# the posterior summaries cannot drift apart between the headline analysis and
# the prior-sensitivity study.
#
# Pure functions only: include it inside an `@everywhere` block when workers
# need it.
# ============================================================================

using CSV, DataFrames, Statistics, StatsBase, Distributions
using DistanceDependentCRP

# ============================================================================
# The contiguity graph and the hop-window prior
# ============================================================================

"""
    load_adjacency(path, geocodes) -> Matrix{Float64}

Symmetric 0/1 adjacency matrix from the edge list written by
`src/prepare_dengue_adjacency.R`, indexed in the order of `geocodes`. Edges to
municipalities not in `geocodes` are ignored.
"""
function load_adjacency(path::AbstractString, geocodes::AbstractVector{<:Integer})
    idx = Dict(Int(g) => i for (i, g) in enumerate(geocodes))
    n = length(geocodes)
    e = CSV.read(path, DataFrame)
    A = zeros(Float64, n, n)
    for row in eachrow(e)
        i = get(idx, Int(row.geocode_i), 0); j = get(idx, Int(row.geocode_j), 0)
        (i == 0 || j == 0) && continue
        A[i, j] = 1.0; A[j, i] = 1.0
    end
    return A
end

neighbour_lists(A::AbstractMatrix) = [findall(!iszero, view(A, i, :)) for i in axes(A, 1)]

"""
    hop_distances(nbrs) -> Matrix{Float64}

All-pairs shortest-path distance in edges, by breadth-first search from every
node. `Inf` marks unreachable pairs, which should not occur: the adjacency
script bridges any disconnected component before writing the edge list.
"""
function hop_distances(nbrs::Vector{Vector{Int}})
    n = length(nbrs)
    H = fill(Inf, n, n)
    queue = Vector{Int}(undef, n)
    for src in 1:n
        H[src, src] = 0.0
        head = 1; tail = 1; queue[1] = src
        while head <= tail
            u = queue[head]; head += 1
            for v in nbrs[u]
                if isinf(H[src, v])
                    H[src, v] = H[src, u] + 1.0
                    tail += 1; queue[tail] = v
                end
            end
        end
    end
    return H
end

# TRUNCATION WITHOUT -Inf. `precompute_log_ddcrp` takes log(f(d)), so a link
# weight of exactly zero puts -Inf into the log-posterior and NaN into the MH
# ratio. The hop window therefore uses a soft floor: every pair within r hops
# sits at distance 0 (weight 1, the same for all of them) and every pair beyond
# it at LOG_FLOOR, which under kernel rate 1 is e^-30 times weaker. The window
# is effectively hard while every log-weight stays finite.
const LOG_FLOOR = 30.0

"""
    hop_window_D(H, r; log_floor = LOG_FLOOR) -> Matrix{Float64}

Distance matrix for the hop-window ddCRP prior of radius `r`: 0 within `r`
hops and `log_floor` beyond. Use with kernel rate 1, so `r` is the only thing
the prior has to say.
"""
function hop_window_D(H::AbstractMatrix, r::Integer; log_floor::Real = LOG_FLOOR)
    r >= 1 || error("hop window radius must be a positive integer, got $r")
    return [H[i, j] <= r ? 0.0 : Float64(log_floor) for i in axes(H, 1), j in axes(H, 2)]
end

# ============================================================================
# Point estimates and agreement
# ============================================================================

"""
    dahl_point_estimate(c_samples, sim; max_draws = 2000) -> Vector{Int}

Least-squares (Dahl 2006) point estimate: the sampled partition whose own
co-clustering matrix is closest to the posterior similarity matrix.
"""
function dahl_point_estimate(c_samples::Matrix{Int}, sim::Matrix{Float64};
                             max_draws::Int = 2000)
    S = size(c_samples, 1)
    cand = S <= max_draws ? (1:S) : round.(Int, range(1, S; length = max_draws))
    best, best_loss = first(cand), Inf
    for s in cand
        z = compute_table_assignments(c_samples[s, :])
        loss = 0.0
        @inbounds for i in 1:length(z), j in (i+1):length(z)
            loss += ((z[i] == z[j] ? 1.0 : 0.0) - sim[i, j])^2
        end
        loss < best_loss && (best_loss = loss; best = s)
    end
    return compute_table_assignments(c_samples[best, :])
end

"""
    mode_point_estimate(c_samples, logpost) -> Vector{Int}

MAP-at-mode partition: the highest log-posterior draw among those whose cluster
count equals the posterior mode of K. Preferred over Dahl for the maps, which
over-fragments when the co-clustering signal is weak; this summary's K matches
the K that gets reported.
"""
function mode_point_estimate(c_samples::Matrix{Int}, logpost::AbstractVector)
    z_all = [compute_table_assignments(c_samples[i, :]) for i in axes(c_samples, 1)]
    Kt = [maximum(z) for z in z_all]
    cand = findall(==(mode(Kt)), Kt)
    return z_all[cand[argmax(logpost[cand])]]
end

"""
    ari(a, b) -> Float64

Adjusted Rand index between two label vectors.
"""
function ari(a::Vector{Int}, b::Vector{Int})
    n = length(a)
    tab = Dict{Tuple{Int,Int},Int}()
    for i in 1:n
        tab[(a[i], b[i])] = get(tab, (a[i], b[i]), 0) + 1
    end
    ai = countmap(a); bj = countmap(b)
    c2(x) = x * (x - 1) / 2
    sij = sum(c2(v) for v in values(tab))
    sa  = sum(c2(v) for v in values(ai))
    sb  = sum(c2(v) for v in values(bj))
    exp_idx = sa * sb / c2(n)
    max_idx = (sa + sb) / 2
    return (sij - exp_idx) / (max_idx - exp_idx)
end

# ============================================================================
# Posterior predictive discrepancy statistics
#
# The first four probe what a Poisson model gets wrong: the spread of the
# counts, the extreme upper tail, and how many municipalities sit at very low
# incidence. Moran's I on the contiguity graph asks whether the fitted
# clustering reproduces the spatial autocorrelation in the data.
# ============================================================================

"""
    moran_I(v, nbrs) -> Float64

Moran's I of `v` over the neighbour lists `nbrs`, with binary contiguity
weights.
"""
function moran_I(v::AbstractVector{<:Real}, nbrs::Vector{Vector{Int}})
    n = length(v)
    dev = v .- mean(v)
    num = 0.0; W = 0
    @inbounds for i in 1:n, j in nbrs[i]
        num += dev[i] * dev[j]
        W += 1
    end
    den = sum(abs2, dev)
    return (den == 0 || W == 0) ? NaN : (n / W) * (num / den)
end

function ppc_statistics(E, nbrs)
    loginc(v) = log.((v .+ 0.5) ./ E)
    return [
        "dispersion index Var/Mean"    => v -> var(v) / mean(v),
        "maximum count"                => v -> float(maximum(v)),
        "n municipalities < 100/100k"  => v -> float(count(i -> v[i] / E[i] < 100, eachindex(v))),
        "90th pct of incidence"        => v -> quantile(v ./ E, 0.9),
        "Moran's I of log incidence"   => v -> moran_I(loginc(v), nbrs),
    ]
end

# ============================================================================
# Between-chain convergence
# ============================================================================

"""
    split_rhat(chains) -> Float64

Rank-normalised split-R̂ (Vehtari et al. 2021) over a vector of equal-length
draw vectors. Each chain is split in half, the pooled draws are rank-normalised
so the statistic is robust to heavy tails, and the usual between/within
variance ratio is formed. Values above about 1.01 indicate that the chains have
not converged to a common distribution.
"""
function split_rhat(chains::AbstractVector{<:AbstractVector{<:Real}})
    length(chains) >= 2 || return NaN
    L = minimum(length.(chains)) ÷ 2
    L >= 2 || return NaN
    pooled = vcat([c[1:2L] for c in chains]...)
    # Rank-normalise: ranks to normal scores via the Blom transform.
    z = quantile.(Normal(), (tiedrank(pooled) .- 3/8) ./ (length(pooled) + 1/4))
    parts = Vector{Vector{Float64}}()
    off = 0
    for _ in chains
        push!(parts, z[off .+ (1:L)])
        push!(parts, z[off .+ (L+1:2L)])
        off += 2L
    end
    W = mean(var.(parts))
    B = L * var(mean.(parts))
    W > 0 || return NaN
    return sqrt(((L - 1) / L * W + B / L) / W)
end
