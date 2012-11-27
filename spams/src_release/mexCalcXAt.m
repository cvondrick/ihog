% 
% Usage:   XAt =mexCalcXAt(X,A);
%
% Name: mexCalcXAt
%
% Description: Compute efficiently XAt = X*A', when A is sparse and has a 
%   lot more columns than rows. In some cases, it is up to 20 times 
%   faster than the equivalent Matlab expression XAt=X*A';
%
% Inputs: X:  double m x n matrix
%         A:  double sparse p x n matrix   
%
% Output: XAt: double m x p matrix 
%
% Author: Julien Mairal, 2009


