classdef SimWorld < handle
    properties
        A
        C
        muInit
        emitVar
        endT
        dt
        endI
        allT
        allAs        
        blankZs
        blankYs
    end
    
    methods
        function obj = SimWorld(varargin)
            props = properties(obj);
            for v = varargin
                obj.(props{1}) = v{1};
                props = props(2:end);
            end
            
            obj.endI = ceil(obj.endT/obj.dt);
            obj.allT = 0:obj.dt:obj.endI*obj.dt;
            
            obj.allAs = NaN([size(obj.A), obj.endI+1]);
            for i = 0:obj.endI
                obj.allAs(:,:,i+1) = obj.A^i;
            end
            
            obj.blankZs = NaN(length(obj.muInit), obj.endI+1);
            obj.blankYs = NaN(min(size(obj.C)), obj.endI+1);
        end
        
        function [Zs, Ys] = getStates(obj)
            Z = (obj.muInit + sqrt(obj.emitVar).*randn(size(obj.muInit)));
            randos = mvnrnd(zeros(size(obj.C*obj.C',1),1), obj.C*diag(obj.emitVar)*obj.C', obj.endI+1)';
            Zs = obj.blankZs;
            Ys = obj.blankYs;
            for i = 1:size(obj.allAs,3)
                Zs(:,i) = obj.allAs(:,:,i)*Z;
                Ys(:,i) = obj.C*Zs(:,i) + randos(:,i);
            end
        end
    end
    
end