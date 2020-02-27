classdef KalmanModel < handle
    properties
        A
        C
        muPrior
        initVar
        a
        totalI
        Ks
        Vars
        blankMus
    end
    
    methods
        function obj = KalmanModel(varargin)
            props = properties(obj);
            for v = varargin
                obj.(props{1}) = v{1};
                props = props(2:end);
            end
            
            if size(obj.initVar,2) ~= size(obj.initVar,1)
               obj.initVar = diag(obj.initVar);
            end
            obj.Vars = NaN([size(obj.initVar), obj.totalI]);
            obj.Ks = NaN([fliplr(size(obj.C)), obj.totalI]);
            obj.Vars(:,:, 1) = obj.initVar;
%           Get filter
            for i = 2:obj.totalI
                P = obj.A*obj.Vars(:,:,i-1)*obj.A';
                obj.Ks(:,:,i) = (P*obj.C')/(obj.C*P*obj.C' + obj.C*obj.initVar*obj.C');
                obj.Vars(:,:,i) = (eye(size(obj.C,2)) - obj.Ks(:,:,i)*obj.C)*P;
            end
            
            obj.blankMus = NaN(length(obj.muPrior)+1, obj.totalI);
        end
        
        function [SEs, eYs, Mus] = runSim(obj, Zs, Ys)
            LLike = @(Y, Mu, Cov) -(logdet(Cov) + log(2*pi)*numel(Y) + ((Y-Mu)'*(Cov\(Y-Mu))))/2;%using log1p and expm1 are hackss, boooo
            Mus = obj.blankMus;
            Mus(:, 1) = [obj.muPrior; 0];
            for i = 2:obj.totalI
                Mus(1:end-1,i) = obj.A*Mus(1:end-1,i-1) + obj.Ks(:,:,i)*(Ys(:,i) - obj.C*obj.A*Mus(1:end-1,i-1));
                Mus(end,i) = Mus(end,i-1) + LLike(Ys(:,i), obj.C*Mus(1:end-1,i), obj.C*obj.Vars(:,:,i)*obj.C');%obj.C*obj.A*obj.Vars(:,:,i-1)*obj.A'*obj.C'+ obj.C*obj.initVar*obj.C');
            end
            SEs = [(real(Mus(1:end-1,:)) - Zs).^2; Mus(end,:)];
            eYs = obj.C*real(Mus(1:end-1,:));
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
        
        function s = logsumexp(~, x, w, dim)
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
            s = y + log(sum(w.*exp(x),dim));
            i = find(~isfinite(y));
            if ~isempty(i)
                s(i) = y(i);
            end
            return
        end
        
    end
    
    
end