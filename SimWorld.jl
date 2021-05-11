using LinearAlgebra
using Distributions
using Random

struct SimWorld{A2<:Array{Float64,2}, F<:Float64, I<:Int64, A1<:Array{Float64,1}, AA<:Array{Array{Float64,2},1}}
    A::A2
    C::A2
    muInit::A1
    emitVar::A1
    endT::F
    dt::F
    endI::I
    allT::A1
    allAs::AA
end

function getStates(SW::SimWorld)
    Z = (SW.muInit .+ sqrt.(SW.emitVar).*randn(size(SW.muInit)))
    Zs = [A*Z for A in SW.allAs]
    Ys = [SW.C*z .+ rand!(MvNormal(zeros(size(SW.C*SW.C',1)), SW.C*diagm(SW.emitVar)*SW.C'), similar(SW.C*SW.muInit)) for z in Zs];
    return Zs::Array{Array{Float64,1},1}, Ys::Array{Array{Float64,1},1}
end

SimWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, emitVar::Array{Float64,1}, endT::Float64=1000., dt::Float64=.1) =
    SimWorld(A, C, muPrior, emitVar, endT, dt, Integer(ceil(endT/dt)))
SimWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, emitVar::Array{Float64,1}, endT::Float64, dt::Float64, endI::Int) =
    SimWorld(A, C, muPrior, emitVar, endT, dt, endI,
                Array{Float64,1}(0:dt:endI*dt),
                [A^t for t in 0:endI])
