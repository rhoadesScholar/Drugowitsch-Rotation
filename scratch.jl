include("Agent.jl")
include("linspecer.jl")
using Distributed

dt = .1;
A1 = [1. 0 0 0 0 0 dt 0;
      0 1 0 0 0 0 0 dt;
      0 0 1 0 dt 0 -dt 0;
      0 0 0 1 0 dt 0 -dt;
      0 0 0 0 1 0 0 0;
      0 0 0 0 0 1 0 0;
      0 0 0 0 0 0 1 0;
      0 0 0 0 0 0 0 1];

 A2 = [1. 0 0 0 0 0 dt 0;
       0 1 0 0 0 0 0 dt;
       0 0 1 0 0 0 -dt 0;
       0 0 0 1 0 0 0 -dt;
       0 0 0 0 1 0 0 0;
       0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 1 0;
       0 0 0 0 0 0 0 1];

As = Array{Array{Float64,2},1}(undef,2);
As[1] = A1;
As[2] = A2;

C = [0. 0 1 0 0 0 0 0;
     0. 0 0 1 0 0 0 0;
     0. 0 0 0 0 0 1 0;
     0. 0 0 0 0 0 0 1];
muPrior = [0., 0, 10., 0, 0., 1, 1., 0];
endT = 100.;
initVar = [1., 1, 100., 100, 1., 1, 1., 1];
emitVar = [1., 1, 100., 100, 1., 1, 1., 1];
a = 1.;

SWs = empty!(Array{SimWorld,1}(undef,1));
for A in As
      push!(SWs, SimWorld(A, C, muPrior, emitVar, endT, dt))
end

KMs = (Array{KalmanModel,1}(undef,length(As)));
for i in 1:length(As)
      KMs[i] = KalmanModel(As[i], C, muPrior, initVar, a, Integer(ceil(endT/dt))+1);
end

N=100
MusLL = Array{Array{Float64,2},3}(undef, length(SWs), length(KMs), N)
for s in 1:length(SWs), i in 1:N
      Zs, Ys = getStates(SWs[s])
      for k in 1:length(KMs)
            # MusLL[s,k,i] = KMs[k].getMusLL(Zs, Ys)
            MusLL[s,k,i] = runSim(Zs, Ys, KMs[k])
            if i==N
                  println(s,":",k,":",i,"->",MusLL[s,k,i][end-1])
            end
      end
end

findmax!(maxval, ind, A)



___________________________
static = StaticWorld(A, C, muPrior, endT, dt);

flexSigValues = [10, 1, .1]
flexSigInd = 4
flexVarName = "sigmaVest"

v=1

simNames = Array{String, 1}(undef,2)
simNames[1] = "Moving"
simNames[2] = "Still"

set_zero_subnormals(true)
variationNum = length(simNames)
colors = linspecer(variationNum)

simOpts = SimOpts(sigmas, 1000)
plotOpts = PlotOpts(simNames[v], colors[v,:])
kworld = FullWorld(static, simOpts);
