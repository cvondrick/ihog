% 
% Usage:   AAt =mexCalcAAt(A);
%
% Name: mexCalcAAt
%
% Description: Compute efficiently AAt = A*A', when A is sparse 
%   and has a lot more columns than rows. In some cases, it is
%   up to 20 times faster than the equivalent Matlab expression
%   AAt=A*A';
%
% Inputs: A:  double sparse m x n matrix   
%
% Output: AAt: double m x m matrix 
%
% Author: Julien Mairal, 2009


