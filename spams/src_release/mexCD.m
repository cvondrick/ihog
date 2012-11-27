% 
% Usage:   A=mexCD(X,D,A0,param);
%
% Name: mexCD
%
% Description: mexCD addresses l1-decomposition problem with a 
%     coordinate descent type of approach.
%     It is optimized for solving a large number of small or medium-sized 
%     decomposition problem (and not for a single large one).
%     It first computes the Gram matrix D'D.
%     This method is particularly well adapted when there is low 
%     correlation between the dictionary elements and when one can benefit 
%     from a warm restart.
%     It aims at addressing the two following problems
%     for all columns x of X, it computes a column alpha of A such that
%       2) when param.mode=1
%         min_{alpha} ||alpha||_1 s.t. ||x-Dalpha||_2^2 <= lambda
%         For this constraint setting, the method solves a sequence of 
%         penalized problems (corresponding to param.mode=2) and looks
%         for the corresponding Lagrange multplier with a simple but
%         efficient heuristic.
%       3) when param.mode=2
%         min_{alpha} 0.5||x-Dalpha||_2^2 + lambda||alpha||_1 
%
% Inputs: X:  double m x n matrix   (input signals)
%               m is the signal size
%               n is the number of signals to decompose
%         D:  double m x p matrix   (dictionary)
%               p is the number of elements in the dictionary
%               All the columns of D should have unit-norm !
%         A0:  double sparse p x n matrix   (initial guess)
%         param: struct
%            param.lambda  (parameter)
%            param.mode (optional, see above, by default 2)
%            param.itermax (maximum number of iterations)
%            param.tol (tolerance parameter)
%            param.numThreads (optional, number of threads for exploiting
%            multi-core / multi-cpus. By default, it takes the value -1,
%            which automatically selects all the available CPUs/cores).
%
% Output: A: double sparse p x n matrix (output coefficients)
%
% Note: this function admits a few experimental usages, which have not
%     been extensively tested:
%         - single precision setting (even though the output alpha 
%           is double precision)
%
% Author: Julien Mairal, 2009


