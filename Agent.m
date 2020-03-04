classdef Agent < handle
    properties
        KMs
        epsilon
        SWs
        CostFun
        Sigmas
        Sigma
        A
        C
        E
        Emahal
        LLike
        opts
    end
    
    methods
        function obj = Agent(varargin)
            props = properties(obj);
            for v = varargin
                obj.(props{1}) = v{1};
                props = props(2:end);
            end
            
            Vars = cat(4, obj.KMs.Vars);
            sigs = @(sigmas) reshape(cell2mat(arrayfun(@(i) nearestSPD(sigmas(:,:,i)), 1:size(sigmas,3), 'UniformOutput', false)), size(sigmas,1), size(sigmas,2), []);
            obj.Sigmas = @(t) sigs(squeeze(Vars(:,:,t,:)));
            obj.Sigma = @(t, k) sigs(squeeze(Vars(:,:,t,k)));
            
            As = {obj.KMs.A};
            obj.A = @(p) nansum(reshape(cell2mat(arrayfun(@(k) As{k}*p(k), 1:length(obj.KMs), 'UniformOutput', false)), size(As{1},1), size(As{1},2), []), 3);          
            Cs = {obj.KMs.C};
            obj.C = @(p) nansum(reshape(cell2mat(arrayfun(@(k) Cs{k}*p(k), 1:length(obj.KMs), 'UniformOutput', false)), size(Cs{1},1), size(Cs{1},2), []), 3);
            
            obj.LLike = @(Y, Mu, Cov) -(logdet(Cov) + log(2*pi)*numel(Y) + ((Y-Mu)'*(Cov\(Y-Mu))))/2;%using log1p and expm1 are hackss, boooo
            obj.E = @(X, p) obj.epsilon*(obj.A(p)*X')';
            obj.Emahal = @(X, p) obj.epsilon * (nansum(reshape(cell2mat(cellfun(@(a, pr) pr*a, cellfun(@(c, a) c*a*c', Cs, As, 'UniformOutput', false), ...
                                                num2cell(p, 2)', 'UniformOutput', false)), size(obj.KMs(1).C,1), [], length(obj.KMs)),3)...
                                                * X')';
            
            if ~any(contains(fields(obj), 'CostFun')) || isempty(obj.CostFun)
                obj.CostFun = 'MSE';
            end
            obj.opts = optimoptions('patternsearch', 'Display','off', 'MaxTime', .0001, 'UseVectorized', true);
        end
        
        function [SEs, metaMus, metaVars] = getMetaMus(obj, Mus, Zs, Ys)
            ps = softmax(obj.getLogOdds(Mus));        
            metaVars = NaN([size(obj.KMs(1).A), size(Ys,2)]);
            LEvid = NaN(1,size(Ys,2));
            
            switch obj.CostFun
                case 'MSE'
                    x = @(t) ps(:,t)'*squeeze(Mus(:,1:end-1,t));
                    
                    metaMus = arrayfun(@(t) x(t), 1:size(Mus,3), 'UniformOutput', false);
                    metaMus = reshape([metaMus{:}], [], size(metaMus,2));
                    
                case 'Abrupt'                     
                    dist = @(t) gmdistribution(Mus(:,1:end-1,t), obj.Sigmas(t), ps(:,t));
            
                    cdfFunc = @(X, t) diff(cdf(dist(t), [X-obj.E(X,ps(:,t)); X+obj.E(X,ps(:,t))]));            
                    lb = squeeze(max(Mus(:,1:end-1,:),[],1));
                    ub = squeeze(min(Mus(:,1:end-1,:),[],1));            
                    x = @(t) patternsearch(@(X) cdfFunc(X,t), ps(:,t)'*dist(t).mu, [], [], [], [], lb(:,t), ub(:,t), [], obj.opts);
                    
                    metaMus = arrayfun(@(t) x(t), 1:size(Mus,3), 'UniformOutput', false);
                    metaMus = reshape([metaMus{:}], [], size(metaMus,2));
                    
                case 'Mahal'                        
                    %Initialize
                    metaMus = obj.KMs(1).blankMus(1:end-1,:);
                    tempDist = gmdistribution([obj.KMs.muPrior]', obj.Sigmas(1), ps(:,1));
                    lb = squeeze(max(Mus(:,1:end-1,:),[],1));
                    ub = squeeze(min(Mus(:,1:end-1,:),[],1));
%                     cdfFunc = @(X, t) diff(cdf(tempDist, [X-obj.E(X,ps(:,1)); X+obj.E(X,ps(:,1))])); 
                    cdfFunc = @(X) arrayfun(@(m) diff(cdf(tempDist, [X(m,:)-obj.E(X(m,:),ps(:,1)); X(m,:)+obj.E(X(m,:),ps(:,1))])), 1:size(X,1));   
                    metaMus(:,1) = patternsearch(@(X) cdfFunc(X,1), ps(:,1)'*tempDist.mu, [], [], [], [], lb(:,1), ub(:,1), [], obj.opts);                   
                    Ydists = cell(1, size(Ys,2));
                    Ydists{1} = gmdistribution(reshape(cell2mat(arrayfun(@(k) obj.KMs(k).C * obj.KMs(k).A * obj.KMs(k).muPrior, 1:length(obj.KMs), 'UniformOutput', false)), [], length(obj.KMs))',...
                                    reshape(cell2mat(arrayfun(@(k) obj.KMs(k).C * obj.Sigma(1, k) * obj.KMs(k).C', 1:length(obj.KMs), 'UniformOutput', false)), size(obj.KMs(1).C,1), [], length(obj.KMs)),...
                                    ps(:,1));
                    mahalT = Ydists{1}.mahal(Ys(:,1)');
%                     pMahalT = (exp(-mahalT) ./ nansum(exp(-mahalT)))';
                    pMahalT = 1 - (mahalT./nansum(mahalT))';
                    if any(pMahalT <= 0)
                            pMahalT = pMahalT -  min(pMahalT) + eps;
                    end
                    pMahalTm1 = pMahalT;
                    
                    thisSig = @(sigs, p) nansum(reshape(cell2mat(arrayfun(@(k) p(k)*sigs(:,:,k), 1:length(p), 'UniformOutput', false)), size(obj.KMs(1).A,1), size(obj.KMs(1).A,2), []), 3);
                    metaVars(:,:,1) = thisSig(cat(3,obj.KMs.initVar), pMahalT);
                    LEvid(1) = log(Ydists{1}.pdf(Ys(:,1)'));
                    %End Initialize
                                        
                    P = @(lastVar, A) A*lastVar*A';
                    K = @(lastVar, k) (P(lastVar, obj.KMs(k).A) * obj.KMs(k).C') / ...
                                      (obj.KMs(k).C*P(lastVar, obj.KMs(k).A)*obj.KMs(k).C' + obj.KMs(k).C*obj.KMs(k).initVar*obj.KMs(k).C');
                    nextVars = @(lastVar) reshape(cell2mat(arrayfun(@(k) nearestSPD((eye(size(obj.KMs(k).C,2)) - K(lastVar, k)*obj.KMs(k).C) * P(lastVar, obj.KMs(k).A)), ...
                                                1:length(obj.KMs), 'UniformOutput', false)), size(obj.KMs(1).A,1), size(obj.KMs(1).A,2), length(obj.KMs));                    
                    
                    nextMus = @(lastMus, lastVar, Y) reshape(cell2mat(arrayfun(@(k) obj.KMs(k).A * lastMus + K(lastVar, k)*(Y - obj.KMs(k).C*obj.KMs(k).A*lastMus), ...
                                                1:length(obj.KMs), 'UniformOutput', false)), [], length(obj.KMs))';
                                            
                    eYvars = @(sigs) reshape(cell2mat(arrayfun(@(k) nearestSPD(obj.KMs(k).C * sigs(:,:,k) * obj.KMs(k).C'), 1:length(obj.KMs), 'UniformOutput', false)), size(obj.KMs(1).C,1), [], length(obj.KMs));
                    eYs = @(mus) reshape(cell2mat(arrayfun(@(k) obj.KMs(k).C * mus(:,k), 1:length(obj.KMs), 'UniformOutput', false)), [], length(obj.KMs))';
                                        
                    for i = 2:size(Ys,2)
                        
                        tempVars = nextVars(metaVars(:,:,i-1));
                        tempMus = nextMus(metaMus(:,i-1), metaVars(:,:,i-1), Ys(:,i));
                        
                        tempYvars = eYvars(tempVars);
                        
                        Ydists{i} = gmdistribution(eYs(tempMus'), eYvars(tempVars), pMahalTm1);
                        pMahalTm1 = pMahalT;
                        mahalT = Ydists{i}.mahal(Ys(:,i)');
%                         pMahalT = (exp(-mahalT) ./ nansum(exp(-mahalT)))';
                        pMahalT = 1 - (mahalT./nansum(mahalT))';
                        if any(pMahalT <= 0)
                                pMahalT = pMahalT -  min(pMahalT) + eps;
                        end
                        
                        tempDist = gmdistribution(tempMus, tempVars, pMahalT);%OBJ.SIGMAS NEEDS TO BE FIXED TO BE THE ACTUAL VARIANCE
%                         cdfFunc = @(X) diff(cdf(tempDist, [X-obj.E(X,pMahalT); X+obj.E(X,pMahalT)]));   
                        cdfFunc = @(X) arrayfun(@(m) diff(cdf(tempDist, [X(m,:)-obj.E(X(m,:),pMahalT); X(m,:)+obj.E(X(m,:),pMahalT)])), 1:size(X,1));   
                        metaMus(:,i) = patternsearch(@(X) cdfFunc(X), pMahalT'*tempDist.mu, [], [], [], [], lb(:,i), ub(:,i), [], obj.opts);
                        metaVars(:,:,i) = thisSig(tempVars, pMahalT);
                        LEvid(i) = LEvid(i-1) + log(Ydists{1}.pdf(Ys(:,1)'));
                    end
            end                      
%                     metaVars = arrayfun(@(t) Sig(t), 1:size(Mus,3), 'UniformOutput', false);
%                     metaVars = reshape([metaVars{:}], size(Sig(1),1), size(Sig(1),2), []);
%             metaVars = cumCov(metaMus');            
%                 LEvid = cumsum(arrayfun(@(t) obj.LLike(Ys(:,t), obj.C(ps(:,t))*metaMus(:,t), obj.C(ps(:,t))*metaVars(:,:,t)*obj.C(ps(:,t))'), 1:size(Mus,3)));
            
            SEs = cat(1,(metaMus - Zs).^2, LEvid);
            
            return
        end
       
        function s = logcumsumexp(~, x, w, dim)
            % Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
            switch nargin 
                case 2
                    % Determine which dimension sum will use
                    dim = find(size(x)~=1,1);
                    if isempty(dim), dim = 1; end
                    w = ones(1, size(x,dim));
                    
                case 3
                    if isscalar(w)
                        dim = w;
                        w = ones(1, size(x,dim));
                    else
                        dim = find(size(x)~=1,1);
                        if isempty(dim), dim = 1; end
                    end
                    
                case 4
                    if isempty(w)
                        w = ones(1, size(x,dim));
                    end
            end

            % subtract the largest in each dim
            y = max(x,[],dim);
            x = bsxfun(@minus,x,y);
            s = y + log(cumsum(w.*exp(x),dim));
            i = find(~isfinite(y));
            if ~isempty(i)
                s(i) = y(i);
            end
            return
        end
       
        function logOdds = getLogOdds(~, MusLL)
            odds = @(a) [diff(a), diff(fliplr(a))];
            if length(size(MusLL)) == 4
                temp = squeeze(MusLL(:,1:end-1,end,:));
                logOdds = NaN(size(temp));
                for i = 1:size(temp,1)
                    logOdds(i,:,:) = reshape(cell2mat(arrayfun(@(j) odds(squeeze(temp(i,:,j))), 1:size(temp,3), 'UniformOutput', false)), 1, size(temp,2), size(temp,3));
                end
            else
                temp = squeeze(MusLL(:,end,:));
                logOdds = reshape(cell2mat(arrayfun(@(j) odds(squeeze(temp(:,j))), 1:size(temp,2), 'UniformOutput', false)), size(temp,1), size(temp,2));
            end

        end
    end
end