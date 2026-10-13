# Builds the municipality-level dengue dataset for one Brazilian state and year:
# cases y (InfoDengue alertcity API, weeks 1-52), population, and centroid lat/lon.
# Cached in data/cleaned/; pass --force to refetch.
#
#   julia --project=. src/prepare_dengue_data.jl --uf 41 --year 2024 [--force]

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CSV, DataFrames, Downloads, Printf, Statistics

# Download cache in data/raw/; outputs in data/cleaned/.
const RAW_DIR  = joinpath(@__DIR__, "..", "data", "raw")
const DATA_DIR = joinpath(@__DIR__, "..", "data", "cleaned")

const MUNICIPIOS_URL =
    "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"

# IBGE state names, keyed by the two-digit UF code.
const UF_NAMES = Dict(
    11 => "Rondônia",   12 => "Acre",       13 => "Amazonas",  14 => "Roraima",
    15 => "Pará",       16 => "Amapá",      17 => "Tocantins", 21 => "Maranhão",
    22 => "Piauí",      23 => "Ceará",      24 => "Rio Grande do Norte",
    25 => "Paraíba",    26 => "Pernambuco", 27 => "Alagoas",   28 => "Sergipe",
    29 => "Bahia",      31 => "Minas Gerais", 32 => "Espírito Santo",
    33 => "Rio de Janeiro", 35 => "São Paulo", 41 => "Paraná",
    42 => "Santa Catarina", 43 => "Rio Grande do Sul", 50 => "Mato Grosso do Sul",
    51 => "Mato Grosso", 52 => "Goiás",     53 => "Distrito Federal",
)

# Argument parsing

function parse_args(args)
    uf, year, force = 41, 2024, false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--uf"
            uf = parse(Int, args[i+1]); i += 2
        elseif a == "--year"
            year = parse(Int, args[i+1]); i += 2
        elseif a == "--force"
            force = true; i += 1
        else
            error("unrecognised argument: $a")
        end
    end
    return (uf = uf, year = year, force = force)
end

# Download helpers

# GET `url` and parse it as CSV, with retries; an empty DataFrame if every attempt fails.
function fetch_csv(url::AbstractString; retries::Int = 3)
    for attempt in 1:retries
        try
            io = IOBuffer()
            Downloads.download(url, io; timeout = 60)
            return CSV.read(take!(io), DataFrame)
        catch err
            attempt == retries && @warn "request failed after $retries attempts" url err
            sleep(2.0 * attempt)
        end
    end
    return DataFrame()
end

# Centroids and names (`geocode, name, lat, lon`) for every municipality in state `uf`.
function municipality_table(uf::Int; force::Bool = false)
    path = joinpath(RAW_DIR, "municipios_brasileiros.csv")
    if force || !isfile(path)
        mkpath(RAW_DIR)
        println("Downloading municipality centroids...")
        Downloads.download(MUNICIPIOS_URL, path)
    end
    df = CSV.read(path, DataFrame)
    sub = df[df.codigo_uf .== uf, :]
    return DataFrame(
        geocode = sub.codigo_ibge,
        name    = sub.nome,
        lat     = Float64.(sub.latitude),
        lon     = Float64.(sub.longitude),
    )
end

# Total notified cases over weeks 1-52 of `year` and the municipality population, or `nothing`.
function fetch_dengue(geocode::Integer, year::Int)
    url = "https://info.dengue.mat.br/api/alertcity?" *
          "geocode=$(geocode)&disease=dengue&format=csv" *
          "&ew_start=1&ew_end=52&ey_start=$(year)&ey_end=$(year)"
    df = fetch_csv(url)
    (nrow(df) == 0 || !hasproperty(df, :casos) || !hasproperty(df, :pop)) && return nothing

    cases = sum(skipmissing(df.casos))
    pops  = collect(skipmissing(df.pop))
    isempty(pops) && return nothing
    return (cases = Int(round(cases)), population = Float64(maximum(pops)))
end

# Boundary geometry (for the R/sf maps)

# Cache the IBGE municipal boundary mesh for state `uf` as GeoJSON.
function fetch_boundaries(uf::Int; force::Bool = false)
    path = joinpath(DATA_DIR, "malha_uf$(uf).geojson")
    if !force && isfile(path)
        println("Boundary mesh already cached: $path")
        return path
    end
    url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/$(uf)?" *
          "formato=application/vnd.geo+json&intrarregiao=municipio&qualidade=minima"
    println("Downloading municipal boundary mesh for UF $uf...")
    try
        Downloads.download(url, path; timeout = 120)
        println("  saved $path ($(round(filesize(path) / 1024, digits=1)) KB)")
    catch err
        @warn "boundary download failed — maps will fall back to centroid points" err
    end
    return path
end

# Main

function main(args)
    opts = parse_args(args)
    uf, year = opts.uf, opts.year
    mkpath(DATA_DIR)

    outfile = joinpath(DATA_DIR, "dengue_uf$(uf)_$(year).csv")
    if isfile(outfile) && !opts.force
        df = CSV.read(outfile, DataFrame)
        println("Dataset already cached: $outfile ($(nrow(df)) municipalities)")
        println("Pass --force to refetch.")
        fetch_boundaries(uf)
        summarise(df, uf, year)
        return
    end

    state = get(UF_NAMES, uf, "UF $uf")
    muni = municipality_table(uf)
    println("Fetching $year dengue counts for $(nrow(muni)) municipalities in $state...")

    geocodes = Int[]; names = String[]; cases = Int[]
    pops = Float64[]; lats = Float64[]; lons = Float64[]
    n_failed = 0

    for (k, row) in enumerate(eachrow(muni))
        res = fetch_dengue(row.geocode, year)
        if isnothing(res)
            n_failed += 1
        else
            push!(geocodes, row.geocode); push!(names, row.name)
            push!(cases, res.cases);      push!(pops, res.population)
            push!(lats, row.lat);         push!(lons, row.lon)
        end
        if k % 25 == 0 || k == nrow(muni)
            @printf("  %4d / %4d  (%d failed)\n", k, nrow(muni), n_failed)
            flush(stdout)
        end
        sleep(0.1)   # be polite to the API
    end

    df = DataFrame(geocode = geocodes, name = names, y = cases,
                   pop = pops, lat = lats, lon = lons)

    # Municipalities with no population cannot carry an exposure offset.
    n_before = nrow(df)
    df = df[df.pop .> 0, :]
    n_dropped = n_before - nrow(df)
    n_dropped > 0 && println("Dropped $n_dropped municipalities with non-positive population.")

    CSV.write(outfile, df)
    println("Wrote $outfile ($(nrow(df)) municipalities, $n_failed API failures)")

    fetch_boundaries(uf)
    summarise(df, uf, year)
end

function summarise(df::DataFrame, uf::Int, year::Int)
    inc = df.y ./ (df.pop ./ 1e5)
    println("\n--- $(get(UF_NAMES, uf, "UF $uf")), $year ---")
    @printf("  municipalities : %d\n", nrow(df))
    @printf("  total cases    : %d\n", sum(df.y))
    @printf("  population     : %.2f million\n", sum(df.pop) / 1e6)
    @printf("  cases          : min %d, median %.0f, max %d\n",
            minimum(df.y), median(df.y), maximum(df.y))
    @printf("  incidence/100k : min %.1f, median %.1f, max %.1f\n",
            minimum(inc), median(inc), maximum(inc))
    @printf("  zero-case municipalities : %d\n", count(==(0), df.y))
    μ = mean(df.y)
    @printf("  marginal Var/Mean of counts : %.1f  (Poisson ⇒ 1)\n", var(df.y) / μ)
end

main(ARGS)
