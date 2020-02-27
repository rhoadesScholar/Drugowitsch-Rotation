classdef Agent < handle
    properties
        KMs
        epsilon
        SWs
        Sigmas
        Sigma
        A
        C
        E
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
            
            obj.LLike = @(Y, Mu, Cov) -(logdet(Cov) + log(2*pi)*size(obj.C, 1) + ((Y-Mu)'*(Cov\(Y-Mu))))/2;%using log1p and expm1 are hackss, boooo
            obj.E = @(X, p) obj.epsilon*(obj.A(p)*X')';
            
            obj.opts = optimoptions('patternsearch', 'Display','off', 'MaxTime', .0001);%, 'UseVectorized', true
        end
        
        function [SEs, metaMus] = getMetaMus(~, Mus, Zs)
            ps = getLogOdds(Mus);%softmax(squeeze(Mus(:,end,:)));       
%             dists = arrayfun(@(t) gmdistribution(Mus(:,1:end-1,t), obj.Sigmas(t), ps(:,t)), 1:size(Mus,3), 'UniformOutput', false);
%             dist = @(t) gmdistribution(Mus(:,1:end-1,t), obj.Sigmas(t), ps(:,t));
            
%             Sig = @(t) nansum(reshape(cell2mat(arrayfun(@(k) obj.Sigma(t, k)*ps(k,t), 1:length(obj.KMs), 'UniformOutput', false)), size(Zs,1), size(Zs,1), []), 3);
%             metaVars = arrayfun(@(t) Sig(t), 1:size(Mus,3), 'UniformOutput', false);
%             metaVars = reshape([metaVars{:}], size(Sig(1),1), size(Sig(1),2), []);
%             
%             cdfFunc = @(X, t) diff(cdf(dist(t), [X-obj.E(X,ps(:,t)); X+obj.E(X,ps(:,t))]));            
%             lb = squeeze(max(Mus(:,1:end-1,:),[],1));
%             ub = squeeze(min(Mus(:,1:end-1,:),[],1));            
%             x = @(t) patternsearch(@(X) cdfFunc(X,t), ps(:,t)'*dist(t).mu, [], [], [], [], lb(:,t), ub(:,t), [], obj.opts);
            x = @(t) ps(:,t)'*squeeze(Mus(:,1:end-1,t));
            metaMus = arrayfun(@(t) x(t), 1:size(Mus,3), 'UniformOutput', false);
            metaMus = reshape([metaMus{:}], [], size(metaMus,2));
            SEs = (metaMus - Zs).^2;
            
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
        
    end
end