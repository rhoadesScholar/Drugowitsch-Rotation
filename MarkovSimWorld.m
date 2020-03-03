classdef MarkovSimWorld < handle
    properties
        As
        C
        muInit
        initVar
        emitVar
        endT
        dt
        TRANS
        EMIS
        endI
        allT
        allAs        
        blankZs
        blankYs
    end
    
    methods
        function obj = MarkovSimWorld(varargin)
            props = properties(obj);
            for v = varargin
                obj.(props{1}) = v{1};
                props = props(2:end);
            end
            
            if ~exist('obj.emitVar', 'var') || isempty(obj.emitVar)
                obj.emitVar = obj.initVar;
            end
            
            obj.endI = ceil(obj.endT/obj.dt);
            obj.allT = 0:obj.dt:obj.endI*obj.dt;
            
            obj.allAs = @() obj.getAllAs();
            
            obj.blankZs = NaN(length(obj.muInit{1}), obj.endI+1);
            obj.blankYs = NaN(min(size(obj.C)), obj.endI+1);
        end
        
        function [Zs, Ys] = getStates(obj)
            [theseAs, seq] = obj.allAs();
            Z = (obj.muInit{seq(1)} + sqrt(obj.initVar{seq(1)}).*randn(size(obj.muInit{seq(1)})));
            randos = cell2mat(cellfun(@(Var) mvnrnd(zeros(min(size(obj.C)),1), obj.C*diag(Var)*obj.C'), obj.emitVar(seq), 'UniformOutput', false))';
            Zs = obj.blankZs;
            Ys = obj.blankYs;
            for i = 1:size(theseAs,3)
                Zs(:,i) = theseAs(:,:,i)*Z;
                Ys(:,i) = obj.C*Zs(:,i) + randos(:,i);
            end
        end
        
        function [allAs, seq] = getAllAs(obj)
            seq = hmmgenerate(obj.endI+1, obj.TRANS, obj.EMIS);
            allAs = cumMtimes(obj.As(seq));
        end
    end
    
end