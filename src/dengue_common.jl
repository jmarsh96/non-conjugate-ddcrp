# Graph, point-estimate, PPC and convergence helpers shared by the run and analyse scripts.

using CSV, DataFrames, Statistics, StatsBase, Distributions
using DistanceDependentCRP

# Symmetric 0/1 adjacency matrix from the edge list of prepare_dengue_adjacency.R, in `geocodes` order.
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

# All-pairs shortest-path distance in edges (breadth-first search from every node).
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

# Soft floor: a link weight of exactly zero would put -Inf into the log-posterior.
const LOG_FLOOR = 30.0

# Hop-window distance matrix: 0 within `r` hops, `log_floor` beyond. Use with kernel rate 1.
function hop_window_D(H::AbstractMatrix, r::Integer; log_floor::Real = LOG_FLOOR)
    r >= 1 || error("hop window radius must be a positive integer, got $r")
    return [H[i, j] <= r ? 0.0 : Float64(log_floor) for i in axes(H, 1), j in axes(H, 2)]
end

# Least-squares (Dahl 2006) partition: the draw closest to the posterior similarity matrix.
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

# Moran's I of `v` with binary contiguity weights.
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

# Rank-normalised split-R̂ (Vehtari et al. 2021) over equal-length draw vectors.
function split_rhat(chains::AbstractVector{<:AbstractVector{<:Real}})
    length(chains) >= 2 || return NaN
    L = minimum(length.(chains)) ÷ 2
    L >= 2 || return NaN
    pooled = vcat([c[1:2L] for c in chains]...)
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
