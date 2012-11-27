% 
% Usage:   x =mexConjGrad(A,b,x0,tol,itermax)
%
% Name: mexConjGrad
%
% Description: Conjugate gradient algorithm, sometimes faster than the 
%    equivalent Matlab function pcg. In order to solve Ax=b;
%
% Inputs: A:  double square n x n matrix. HAS TO BE POSITIVE DEFINITE
%         b:  double vector of length n.
%         x0: double vector of length n. (optional) initial guess.
%         tol: (optional) tolerance.
%         itermax: (optional) maximum number of iterations.
%
% Output: x: double vector of length n.
%
% Author: Julien Mairal, 2009


