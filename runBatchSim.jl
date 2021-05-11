using PyPlot
using Distributions
using Random
using LinearAlgebra
using Statistics
using Distributed

struct StaticWorld{A2<:Array{Float64,2}, F<:Float64, I<:Int64, A1<:Array{Float64,1}, AA<:Array{Array{Float64,2},1}, FN<:Function}
    #Sim Priors [position, object distance, object motion, velocity]
    A::A2
    C::A2
    muPrior::A2
    endT::F
    dt::F
    endI::I
    allT::A1
    allAs::AA
    Zs::FN
end
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, endT::Float64, dt::Float64) =
    StaticWorld(A, C, muPrior, endT, dt, Integer(ceil(endT/dt)))
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}) =
    StaticWorld(A, C, muPrior, 500., .5, 1000)
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, endT::Float64, dt::Float64, endI::Int) =
    StaticWorld(A, C, muPrior, endT, dt, endI,
                Array{Float64,1}(0:dt:endI*dt),
                [A^t for t in 0:endI], (A::Array{Float64,2},Z::Array{Float64,2})->A*Z)

struct SimOpts{A<:Array{Float64,1}, F<:Float64, I<:Int64}
    sigmas::A
    a::F
    N::I
end
SimOpts(num::Int64) = SimOpts(ones(num,1))
SimOpts(static::StaticWorld) = SimOpts(ones(size(static.A,1)))
SimOpts(sigmas::Array{Float64,1}) = SimOpts(sigmas, 1., 500)
SimOpts(sigmas::Array{Float64,1}, N::Int) = SimOpts(sigmas, 1., N)
SimOpts(sigmas::Array{Float64,1}, a::Float64) = SimOpts(sigmas, a, 500)

struct PlotOpts{S<:String, A<:Array{Float64,1}, F<:Float64}
    label::S
    color::A
    width::F
    alpha::F
end
PlotOpts(label::String) = PlotOpts(label, [0,0,1], 2., .2)
PlotOpts(label::String, color::Array{Float64,1}) = PlotOpts(label, color, 2., .2)

function getKalman(static::StaticWorld, initVar::Array{Float64,2}, sigmaIn::Array{Float64,2})
    Vars = Array{Array{Float64,2}, 1}(undef, static.endI+1)
    Ks = Array{Array{Float64,2}, 1}(undef, static.endI+1)
    Vars[1] = initVar
    #Get filter
    for i = 2:static.endI+1
        Ks[i] = (static.A*Vars[i-1]*static.A'*static.C')/(static.C*static.A*Vars[i-1]*static.A'*static.C' .+ sigmaIn)
        Vars[i] = (I - Ks[i]*static.C)*static.A*Vars[i-1]*static.A'
    end
    return Ks, Vars
end

struct FullWorld{S<:StaticWorld, F<:Float64, A<:Array{Float64,2}, AA<:Array{Array{Float64,2},1}, FN<:Function, FN2<:Function}
    static::S
    a::F
    sigmaIn::A
    Ys::FN
    Ks::AA
    Vars::AA
    Z::FN2
end
FullWorld(static::StaticWorld, simOpts::SimOpts) =
    FullWorld(static::StaticWorld, simOpts.a,
    diagm(simOpts.a*simOpts.sigmas), (static.C*diagm(simOpts.sigmas)*static.C')/static.dt)
FullWorld(static::StaticWorld, a::Float64, initVar::Array{Float64,2}, sigmaIn::Array{Float64,2}) =
    FullWorld(static, a, sigmaIn,
    (Z::Array{Float64,2})->rand!(MvNormal(zeros(size(sigmaIn,1)), sigmaIn), similar(static.C*static.muPrior)) + static.C*Z,
    getKalman(static, initVar, sigmaIn)...,
    ()->(static.muPrior + sqrt.(diag(initVar)).*randn(size(static.muPrior))))

function runSim(kworld::FullWorld)
    # Zs = [A*Z for A in kworld.static.allAs]
    #Zs = A->A*Z
    # Z  = kworld.Z()
    # Zs = kworld.static.Zs(Z)
    Z = collect(Iterators.repeated(kworld.Z(),kworld.static.endI+1));
    Zs = kworld.static.Zs.(kworld.static.allAs, Z);
    Ys = kworld.Ys.(Zs);#[kworld.static.C*z .+ rand!(kworld.noise, similar(kworld.static.C*Z[1])) for z in Zs]

    Mus = Array{Array{Float64,2},1}(undef, kworld.static.endI+1)
    Mus[1] = kworld.static.muPrior
    for i = 2:kworld.static.endI+1
        Mus[i] = kworld.static.A*Mus[i-1] + kworld.Ks[i]*(Ys[i] - kworld.static.C*kworld.static.A*Mus[i-1]);
    end
    return hcat(se.(Mus-Zs)...)
end

function se(M::Array{Float64,2})
    return dropdims(mean(M.^2,dims=2), dims=2)#SQRT CHANGES METRIC FROM VARIANCE (JAN MEETING 2/3/20), note:summing across dimensions sums together variances (changed to averaging)
end

function plotshade(y::Array{Float64,1}, err::Array{Float64,1}, x::Array{Float64,1}, opts::PlotOpts)
    f = fill_between(x[1:length(y)], y+err, y-err,color=opts.color, alpha=opts.alpha, linestyle="-.", hatch="/")
    p = plot(x[1:length(y)],y,color=opts.color[:],linewidth = opts.width, label=opts.label) ## change color | linewidth to adjust mean line()
end

function plotMSE(MSE::Array{Float64,2}, mVars::Array{Float64,2}, allT::Array{Float64,1}, opts::PlotOpts)

    subplot(2, 2, 1)
    plot(allT, MSE[1,:], color=opts.color[:], linewidth=opts.width, label=opts.label)
    plot(allT, mVars[1,:], color=opts.color[:], linewidth=opts.width, linestyle=":")
    # plotshade(RMSE[1,:], eVars[1,:], allT, opts)
    # fill_between(allT, RMSE[1,:]+mVars[1,:], RMSE[1,:]-mVars[1,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
    xlabel("time")
    ylabel("mean square error")
    title("Position MSE")

    if size(MSE,1) > 2
        subplot(2, 2, 2)
        plot(allT, MSE[2,:], color=opts.color[:], linewidth=opts.width, label=opts.label)
        plot(allT, mVars[2,:], color=opts.color[:], linewidth=opts.width, linestyle=":")
        # plotshade(RMSE[2,:], eVars[2,:], allT, opts)
        # fill_between(allT, RMSE[2,:]+mVars[2,:], RMSE[2,:]-mVars[2,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
        xlabel("time")
        ylabel("mean square error")
        title("Object Distance MSE")
    end

    subplot(2, 2, 3)
    plot(allT, MSE[4,:], color=opts.color[:], linewidth=opts.width, label=opts.label)
    plot(allT, mVars[4,:], color=opts.color[:], linewidth=opts.width, linestyle=":")
    # plotshade(RMSE[end,:], eVars[end,:], allT, opts)
    # fill_between(allT, RMSE[end,:]+mVars[end,:], RMSE[end,:]-mVars[end,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
    xlabel("time")
    ylabel("mean square error")
    title("Velocity MSE")

    return
end

function runBatchSim(plotOpts::PlotOpts, static::StaticWorld, simOpts::SimOpts)
    kworld = FullWorld(static, simOpts)

    ses = Array{Array{Float64,2},1}(undef, simOpts.N)
    # pmap(n->ses[n] = runSim(kworld), 1:simOpts.N)
    @simd for n = 1:simOpts.N
       ses[n] = runSim(kworld)
    end
    MSE = mean(ses)
    # eVars = var(rses; mean=RMSE)
    mVars = hcat([diag(M) for M in kworld.Vars]...)#SQRT CHANGES METRIC FROM VARIANCE (JAN MEETING 2/3/20), note:summing across dimensions sums together variances
    plotMSE(MSE, mVars, kworld.static.allT, plotOpts)

    return ses
end
