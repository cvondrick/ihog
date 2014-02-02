% 
% Usage: [W]=mexRidgeRegression(Y,X,W0,param);
%
% Name: mexRidgeRegression
%
% Description: mexFistaFlat solves sparse regularized problems.
%         X is a design matrix of size m x p
%         X=[x^1,...,x^n]', where the x_i's are the rows of X
%         Y=[y^1,...,y^n] is a matrix of size m x n
%         It implements a conjugate gradient solver for ridge regression
%         
% Inputs: Y:  double dense m x n matrix
%         X:  double dense or sparse m x p matrix   
%         W0:  double dense p x n matrix or p x Nn matrix (for multi-logistic loss)
%              initial guess
%         param: struct
%            param.lambda (regularization parameter)
%            param.numThreads (optional, number of threads for exploiting
%                multi-core / multi-cpus. By default, it takes the value -1,
%                which automatically selects all the available CPUs/cores).
%            param.itermax (optional, maximum number of iterations, 100 by default)
%            param.tol (optional, tolerance for stopping criteration, which is a relative duality gap
%                if it is available, or a relative change of parameters).
%
% Output:  W:  double dense p x n matrix 

% Author: Julien Mairal, 2013


