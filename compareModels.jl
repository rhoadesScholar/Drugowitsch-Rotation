include("runBatchCompareSim.jl")
include("linspecer.jl")
using Printf

function compareSims(A::Array{Float64,2}, As::Array{Array{Float64,2},1}, C::Array{Float64,2}, muPrior::Array{Float64,1}, endT::Number, dt::Number,
                    sigmas::Array{Float64,1}, simNames::Array{String,1})
    set_zero_subnormals(true)
    variationNum = length(As)
    colors = linspecer(variationNum)
    static = StaticWorld(A, C, muPrior, endT, dt)
    # figure()
    for v = 1:variationNum
        println(string("Variation #", v))
        plotOpts = PlotOpts(simNames[v], colors[v,:])
        simOpts = SimOpts(sigmas, 1000)
        @time runBatchSim(plotOpts, static, simOpts, As)
    end
    legend()

    return
end

function compareSims(As::Array{Array{Float64,2},1}, C::Array{Float64,2}, muPrior::Array{Float64,1}, sigmas::Array{Float64,1}, simNames::Array{String,1})
    set_zero_subnormals(true)
    variationNum = length(As)
    colors = linspecer(variationNum)
    static = StaticWorld(A, C, muPrior)
    # figure()
    for v = 1:variationNum
        println(string("Variation #", v))
        plotOpts = PlotOpts(simNames[v], colors[v,:])
        simOpts = SimOpts(sigmas, 1000)
        @time runBatchSim(plotOpts, static, simOpts, As)
    end
    legend()

    return
end
