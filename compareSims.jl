include("runBatchSim.jl")
include("linspecer.jl")
using Printf

# dt = .1
# A = [1 0 0 dt;
#      0 1 dt -dt;
#      0 0 1 0;
#      0 0 0 1]
# C = [0. 1. 0 0;
#      0 0 0 1]
# muPrior = [0. 0; 10. 0; 0. 1; 1. 0]
# endT = 100.
# sigmas = [1., 100., 0., 1.]
#
# static = StaticWorld(A, C, muPrior, endT, dt)
#
# flexSigValues = [10, 1, .1]
# flexSigInd = 4
# flexVarName = "sigmaVest"

function compareSims(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, endT::Number, dt::Number,
                    sigmas::Array{Float64,1}, flexSigValues::Array{Float64,1}, flexSigInd::Int64, flexVarName::String)
    static = StaticWorld(A, C, muPrior, endT, dt)
    compareSims(static, sigmas, flexSigValues, flexSigInd, flexVarName)
    return
end

function compareSims(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, sigmas::Array{Float64,1},
                    flexSigValues::Array{Float64,1}, flexSigInd::Int64, flexVarName::String)
    static = StaticWorld(A, C, muPrior)
    compareSims(static, sigmas, flexSigValues, flexSigInd, flexVarName)
    return
end

function compareSims(static::StaticWorld, sigmas::Array{Float64,1}, flexSigValues::Array{Float64,1}, flexSigInd::Int64, flexVarName::String)
    set_zero_subnormals(true)
    variationNum = length(flexSigValues)
    colors = linspecer(variationNum)
    # figure()
    for v = 1:variationNum
        println(string("Variation #", v))
        plotOpts = PlotOpts(@sprintf("%s = %s", flexVarName, string.(flexSigValues[v])), colors[v,:])
        setindex!(sigmas, flexSigValues[v], flexSigInd)
        simOpts = SimOpts(sigmas, 1000)
        @time runBatchSim(plotOpts, static, simOpts)
    end
    legend()
    return
end
