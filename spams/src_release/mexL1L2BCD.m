% 
% Usage:   alpha=mexL1L2BCD(X,D,alpha0,list_groups,param);
%
% Name: mexL1L2BCD
%     (this function has not been intensively tested).
%
% Description: mexL1L2BCD is a solver for a 
%     Simultaneous signal decomposition formulation based on block 
%     coordinate descent.
%
%     X is a matrix structured in groups of signals, which we denote
%     by X=[X_1,...,X_n]
%     
%     if param.mode=2, it solves
%         for all matrices X_i of X, 
%         min_{A_i} 0.5||X_i-D A_i||_2^2 + lambda/sqrt(n_i)||A_i||_{1,2}  
%         where n_i is the number of columns of X_i
%     if param.mode=1, it solves
%         min_{A_i} ||A_i||_{1,2} s.t. ||X_i-D A_i||_2^2  <= n_i lambda
%         
%
% Inputs: X:  double m x N matrix   (input signals)
%            m is the signal size
%            N is the total number of signals 
%         D:  double m x p matrix   (dictionary)
%            p is the number of elements in the dictionary
%         alpha0: double dense p x N matrix (initial solution)
%         list_groups : int32 vector containing the indices (starting at 0)
%            of the first elements of each groups.
%         param: struct
%            param.lambda (regularization parameter)
%            param.mode (see above, by default 2)
%            param.itermax (maximum number of iterations, by default 100)
%            param.tol (tolerance parameter, by default 0.001)
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


