function [Ahat, success] = nearestSPD(A)
% nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
% usage: Ahat = nearestSPD(A)
%
% From Higham: "The nearest symmetric positive semidefinite matrix in the
% Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
% where H is the symmetric polar factor of B=(A + A')/2."
%
% http://www.sciencedirect.com/science/article/pii/0024379588902236
%
% arguments: (input)
%  A - square matrix, which will be converted to the nearest Symmetric
%    Positive Definite Matrix.
%
% Arguments: (output)
%  Ahat - The matrix chosen as the nearest SPD matrix to A.
success = true;

if nargin ~= 1
  error('Exactly one argument must be provided.')
end

% test for a square matrix A
[r,c] = size(A);
if r ~= c
  error('A must be a square matrix.')
elseif (r == 1) && (A <= 0)
  % A was scalar and non-positive, so just return eps
  Ahat = eps;
  return
end

[~,p] = cholcov(A);
if p == 0
    % A is already SPD
    Ahat = A;
    success = true;
    return
end

% symmetrize A into B
B = (A + A')/2;

% Compute the symmetric polar factor of B. Call it H.
% FALSE CLAIM: "Clearly H is itself SPD." Verify/fix
[~,Sigma,V] = svd(B);
H = V*Sigma*V';

if ~issymmetric(H)
    H = (H + H')/2;
end
try
    chol(H);
catch
    success = false;
%     if ~all(eps > H, 'all')
%         H = nearestSPD(H);
%     end
end

% get Ahat in the above formula
Ahat = (B+H)/2;

% ensure symmetry
Ahat = (Ahat + Ahat')/2;

if any(diag(Ahat) == 0, 'all')%HACK: nudge zero elements from the diagonal
    temp = diag(Ahat);
    temp(temp==0) = eps;
    Ahat = diag(temp) + triu(Ahat,1) + tril(Ahat,-1);
end

% test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
p = 1;
k = 0;
while p ~= 0 && success
  [~,p] = chol(Ahat);
  k = k + 1;
  if p ~= 0
    % Ahat failed the chol test. It must have been just a hair off,
    % due to floating point trash, so it is simplest now just to
    % tweak by adding a tiny multiple of an identity matrix.
    mineig = min(eig(Ahat));
    Ahat = Ahat + (-mineig*k.^2 + eps(mineig))*eye(size(A));
    
    if all(A < eps, 'all')
        success = false;
    end
  end
end