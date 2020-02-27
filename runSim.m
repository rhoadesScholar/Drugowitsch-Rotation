function [MusLL, Vars] = runSim(As, C, muPrior, initVar, options)%labels, dims, emitVar, muInit, endT, N, dt, a, epsilon)
    arguments
        As cell
        C = [0. 0 1 0 0 0 0 0;
            0. 0 0 1 0 0 0 0;
            0. 0 0 0 0 0 1 0;
            0. 0 0 0 0 0 0 1];
        muPrior = {[0., 0, 10., 10, 0., 1, 1., 0]}';%[posX, posY, objDistX, objDistY, objVelX, objVelY, selfVelX, selfVelY]
        initVar = {[1., 1, 5., 5, 10., 10, 100., 100]}';%[posX, posY, objDistX, objDistY, objVelX, objVelY, selfVelX, selfVelY]
        options.labels = {'Moving', 'Still', ''; 'Moving', 'Still', 'Combined'};
        options.dims = 2;
        options.emitVar = initVar;%[memoryX, memoryY, visionX, visionY, objInertiaX, objInertiaY, vestibularX, vestibularY]
        options.muInit = muPrior;
        options.endT = 100.;
        options.N = 100;
        options.dt = .1;
        options.a = 1.;
        options.epsilon = .1;    
        options.agnt = false;
        options.noiseVary = true;
    end
    
    arg = fieldnames(options);
    for v = 1:length(arg)
       setThis(arg{v}, options.(arg{v}));
    end
    
    for i = 1:length(As)
          KMs(i) = KalmanModel(As{i}, C, muPrior{i}, initVar{i}, a, ceil(endT/dt)+1);
          SWs(i) = SimWorld(As{i}, C, muInit{i}, emitVar{i}, endT, dt);
    end
    if agnt && length(KMs) > 1
        Bond007 = Agent(KMs, epsilon, SWs);
        agnt = true;
    else
        agnt = false;
    end
    
    Vars = cat(4, KMs.Vars);

    MusLL = NaN(length(SWs), length(KMs)+agnt, length(muPrior{1})+1, ceil(endT/dt)+1);
    for s = 1:length(SWs)
        SEs = NaN(length(KMs)+agnt, N, length(muPrior{1})+1, ceil(endT/dt)+1);
        Mus = NaN(length(KMs), length(muPrior{1})+1, ceil(endT/dt)+1);
        for i = 1:N
            [Zs, Ys] = SWs(s).getStates();
            for k = 1:length(KMs)
                if ~noiseVary || k == s
                    [SEs(k,i,:,:), ~, Mus(k,:,:)] =  KMs(k).runSim(Zs, Ys);
                end
            end

            if agnt
                SEs(k+1,i,1:end-1,:) = Bond007.getMetaMus(Mus, Zs);        
            end

        end
        MusLL(s,:,:,:) =  squeeze(nanmean(SEs, 2));
    end
    
    if noiseVary
        temp = NaN(1,size(MusLL,2),size(MusLL,3),size(MusLL,4));
        for i = 1:size(MusLL,1)
            temp(1,i,:,:) = MusLL(i,i,:,:);
        end
        MusLL= temp;
    end
    
    plotMSE(MusLL, dims, Vars, SWs(1).allT, labels)
    return
end

function setThis(var, val)
    assignin('caller', var, val);
end