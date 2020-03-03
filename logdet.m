function [y,U] = logdet(A)
    %LOGDET  Logarithm of determinant for positive-definite matrix
    % logdet(A) returns log(det(A)) where A is positive-definite.
    % This is faster and more stable than using log(det(A)).
    % Note that logdet does not check if A is positive-definite.
    % If A is not positive-definite, the result will not be the same as log(det(A)).

    % Written by Tom Minka
    % (c) Microsoft Corporation. All rights reserved.

%     if any(diag(A) == 0, 'all')%HACK: nudge zero elements from the diagonal
%         temp = diag(A);
%         temp(temp==0) = eps;
%         A = diag(temp) + triu(A,1) + tril(A,-1);
%     end
    
    
    [U, flag] = chol(A);
    if flag~=0
        [A, success] = nearestSPD(A);
        [U, flag] = chol(A);
        if flag~=0 || ~success
            y = log(det(A));
            return
        end
    end
    y = 2*nansum(log(diag(U)));
    return
end
