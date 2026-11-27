# Shared figure style: the Okabe-Ito colourblind-safe palette and publication defaults.
# include(joinpath(@__DIR__, "plot_style.jl")) after `using StatsPlots`.

const OI_BLUE = colorant"#0072B2"
const OI_ORANGE = colorant"#E69F00"
const OI_GREEN = colorant"#009E73"
const OI_VERMILLION = colorant"#D55E00"
const OI_PURPLE = colorant"#CC79A7"
const OI_SKY = colorant"#56B4E9"
const OI_BLACK = colorant"#000000"
const OI_GREY = colorant"#999999"

# Categorical order: six hues, then the neutrals for a seventh and eighth series.
const CATEGORICAL = [OI_BLUE, OI_ORANGE, OI_GREEN, OI_VERMILLION, OI_PURPLE, OI_SKY, OI_BLACK, OI_GREY]

# Sequential ramp for a magnitude on lines and arrows: light grey to black.
const GREY_RAMP = cgrad([colorant"#D0D0D0", OI_BLACK])

default(palette = CATEGORICAL, dpi = 300, gridalpha = 0.15, gridlinewidth = 0.5,
        foreground_color_legend = nothing, background_color_legend = nothing)
