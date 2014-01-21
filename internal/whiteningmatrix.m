% Returns a matrix w and vector mu that whitens a ny-by-nx HOG.
%
% Whitening:
%   y = w * (x - mu)
% Coloring:
%   x = c * y + mu;
function [w, mu, c] = whiteningmatrix(ny, nx),

[sig, mu] = hogcovariance(ny, nx);
[v,d] = eig(sig);

sqr = sqrt(d);

% compute whitening matrix:
% since d is diagonal, we can efficiently invert 
w = diag(1./diag(sqr)) * v';

% compute coloring matrix:
c = v * sqr;
