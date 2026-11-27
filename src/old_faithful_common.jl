# Old Faithful setup shared by run_old_faithful.jl and run_gibbs_comparison.jl.

const MODEL = GammaClusterShapeMarg()

const DDCRP_PARAMS = DDCRPParams(1.0, 0.2, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
const PRIORS = GammaClusterShapeMargPriors(α_a=2.0, α_b=0.5, β_a=2.0, β_b=0.5)
const PROP_SD_SHAPE = 0.5

const FAITHFUL = DataFrame(CSV.File(joinpath(@__DIR__, "..", "data", "cleaned", "faithful.csv")))
const Y_FAITHFUL = Float64.(FAITHFUL.eruptions)
const W_FAITHFUL = Float64.(FAITHFUL.waiting)
const D_FAITHFUL = pairwise(Euclidean(), W_FAITHFUL)
