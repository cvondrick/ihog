% 
% Usage:   alpha=mexSOMP(X,D,list_groups,param);
%
% Name: mexSOMP
%     (this function has not been intensively tested).
%
% Description: mexSOMP is an efficient implementation of a
%     Simultaneous Orthogonal Matching Pursuit algorithm. It is optimized
%     for solving a large number of small or medium-sized 
%     decomposition problem (and not for a single large one).
%     It first computes the Gram matrix D'D and then perform
%     a Cholesky-based OMP of the input signals in parallel.
%     It aims at addressing the following NP-hard problem
%
%     X is a matrix structured in groups of signals, which we denote
%     by X=[X_1,...,X_n]
%     
%     for all matrices X_i of X, 
%         min_{A_i} ||A_i||_{0,infty}  s.t  ||X_i-D A_i||_2^2 <= eps*n_i
%         where n_i is the number of columns of X_i
%
%         or
%
%         min_{A_i} ||X_i-D A_i||_2^2  s.t. ||A_i||_{0,infty} <= L
%         
%
% Inputs: X:  double m x N matrix   (input signals)
%            m is the signal size
%            N is the total number of signals 
%         D:  double m x p matrix   (dictionary)
%            p is the number of elements in the dictionary
%            All the columns of D should have unit-norm !
%         list_groups : int32 vector containing the indices (starting at 0)
%            of the first elements of each groups.
%         param: struct
%            param.L (maximum number of elements in each decomposition)
%            param.eps (threshold on the squared l2-norm of the residual
%            param.numThreads (optional, number of threads for exploiting
%            multi-core / multi-cpus. By default, it takes the value -1,
%            which automatically selects all the available CPUs/cores).
%
% Output: alpha: double sparse p x N matrix (output coefficients)
%
% Note: this function admits a few experimental usages, which have not
%     been extensively tested:
%      - single precision setting (even though the output alpha is double 
%        precision)
%
% Author: Julien Mairal, 2010


