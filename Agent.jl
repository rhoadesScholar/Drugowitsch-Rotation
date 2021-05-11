include("Kalman.jl")
# using Main.Kalman
include("SimWorld.jl")

struct Agent{KMs<:Array{KalmanModel,1}, FN<:Function, I<:Int}#, SWs<:Array{SimWorld,1}}
    models::KMs
    choice::FN
    dims::I

    # worlds::SWs
end

function getMSE(MusLL::Array{Array{Float64,2},1}, dims::Int, Vars::Array{Array{Float64,2},1})
    MVar = reshape(vcat([mean(reshape(diag(data), dims, :), dims=1) for data in Vars]...), :, size(Vars,1))
    MSEs = [mean(reshape(data[1:end-1,:], dims, :, size(data,2)), dims=1) for data in MusLL]
    MSE = dropdims(mean(vcat(MSEs...), dims=1), dims=1)

    return MSE, MVar
end
#
# function plotMSE(MSE::Array{Float64,2}, mVars::Array{Float64,2}, allT::Array{Float64,1}, opts::PlotOpts)
#
#     mVars = hcat([diag(M) for M in kworld.Vars]...)
#
#     subplot(2, 2, 1)
#     plot(allT, MSE[1,:], color=opts.color[:], linewidth=opts.width, label=opts.label)
#     plot(allT, mVars[1,:], color=opts.color[:], linewidth=opts.width, linestyle=":")
#     # plotshade(RMSE[1,:], eVars[1,:], allT, opts)
#     # fill_between(allT, RMSE[1,:]+mVars[1,:], RMSE[1,:]-mVars[1,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
#     xlabel("time")
#     ylabel("mean square error")
#     title("Position MSE")
#
#     if size(MSE,1) > 3
#         subplot(2, 2, 2)
#         plot(allT, MSE[2,:], color=opts.color[:], linewidth=opts.width, label=opts.label)
#         plot(allT, mVars[2,:], color=opts.color[:], linewidth=opts.width, linestyle=":")
#         # plotshade(RMSE[2,:], eVars[2,:], allT, opts)
#         # fill_between(allT, RMSE[2,:]+mVars[2,:], RMSE[2,:]-mVars[2,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
#         xlabel("time")
#         ylabel("mean square error")
#         title("Object Distance MSE")
#     end
#
#     subplot(2, 2, 3)
#     plot(allT, MSE[end-1,:], color=opts.color[:], linewidth=opts.width, label=opts.label)
#     plot(allT, mVars[end-1,:], color=opts.color[:], linewidth=opts.width, linestyle=":")
#     # plotshade(RMSE[end,:], eVars[end,:], allT, opts)
#     # fill_between(allT, RMSE[end,:]+mVars[end,:], RMSE[end,:]-mVars[end,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
#     xlabel("time")
#     ylabel("mean square error")
#     title("Velocity MSE")
#
#     subplot(2, 2, 4)
#     plot(allT, MSE[end,:], color=opts.color[:], linewidth=opts.width, label=opts.label)
#     # plotshade(RMSE[end,:], eVars[end,:], allT, opts)
#     # fill_between(allT, RMSE[end,:]+mVars[end,:], RMSE[end,:]-mVars[end,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
#     xlabel("time")
#     ylabel("Log Likelihood")
#     title("Overall Model Performance")
#
#     return
# end
