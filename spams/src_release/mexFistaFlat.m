% 
% Usage: [W [optim]]=mexFistaFlat(Y,X,W0,param);
%
% Name: mexFistaFlat
%
% Description: mexFistaFlat solves sparse regularized problems.
%         X is a design matrix of size m x p
%         X=[x^1,...,x^n]', where the x_i's are the rows of X
%         Y=[y^1,...,y^n] is a matrix of size m x n
%         It implements the algorithms FISTA, ISTA and subgradient descent.
%         
%           - if param.loss='square' and param.regul is a regularization function for vectors,
%             the entries of Y are real-valued,  W = [w^1,...,w^n] is a matrix of size p x n
%             For all column y of Y, it computes a column w of W such that
%               w = argmin 0.5||y- X w||_2^2 + lambda psi(w)
%
%           - if param.loss='square' and param.regul is a regularization function for matrices
%             the entries of Y are real-valued,  W is a matrix of size p x n. 
%             It computes the matrix W such that
%               W = argmin 0.5||Y- X W||_F^2 + lambda psi(W)
%            
%           - param.loss='square-missing' : same as param.loss='square', but handles missing data
%             represented by NaN (not a number) in the matrix Y
%
%           - if param.loss='logistic' and param.regul is a regularization function for vectors,
%             the entries of Y are either -1 or +1, W = [w^1,...,w^n] is a matrix of size p x n
%             For all column y of Y, it computes a column w of W such that
%               w = argmin (1/m)sum_{j=1}^m log(1+e^(-y_j x^j' w)) + lambda psi(w),
%             where x^j is the j-th row of X.
%
%           - if param.loss='logistic' and param.regul is a regularization function for matrices
%             the entries of Y are either -1 or +1, W is a matrix of size p x n
%               W = argmin sum_{i=1}^n(1/m)sum_{j=1}^m log(1+e^(-y^i_j x^j' w^i)) + lambda psi(W)
%
%           - if param.loss='multi-logistic' and param.regul is a regularization function for vectors,
%             the entries of Y are in {0,1,...,N} where N is the total number of classes
%             W = [W^1,...,W^n] is a matrix of size p x Nn, each submatrix W^i is of size p x N
%             for all submatrix WW of W, and column y of Y, it computes
%               WW = argmin (1/m)sum_{j=1}^m log(sum_{j=1}^r e^(x^j'(ww^j-ww^{y_j}))) + lambda sum_{j=1}^N psi(ww^j),
%             where ww^j is the j-th column of WW.
%
%           - if param.loss='multi-logistic' and param.regul is a regularization function for matrices,
%             the entries of Y are in {0,1,...,N} where N is the total number of classes
%             W is a matrix of size p x N, it computes
%               W = argmin (1/m)sum_{j=1}^m log(sum_{j=1}^r e^(x^j'(w^j-w^{y_j}))) + lambda psi(W)
%             where ww^j is the j-th column of WW.
%
%           - param.loss='cur' : useful to perform sparse CUR matrix decompositions, 
%               W = argmin 0.5||Y-X*W*X||_F^2 + lambda psi(W)
%
%
%         The function psi are those used by mexProximalFlat (see documentation)
%
%         This function can also handle intercepts (last row of W is not regularized),
%         and/or non-negativity constraints on W, and sparse matrices for X
%
% Inputs: Y:  double dense m x n matrix
%         X:  double dense or sparse m x p matrix   
%         W0:  double dense p x n matrix or p x Nn matrix (for multi-logistic loss)
%              initial guess
%         param: struct
%            param.loss (choice of loss, see above)
%            param.regul (choice of regularization, see function mexProximalFlat)
%            param.lambda (regularization parameter)
%            param.lambda2 (optional, regularization parameter, 0 by default)
%            param.lambda3 (optional, regularization parameter, 0 by default)
%            param.verbose (optional, verbosity level, false by default)
%            param.pos (optional, adds positivity constraints on the
%                coefficients, false by default)
%            param.transpose (optional, transpose the matrix in the regularization function)
%            param.size_group (optional, for regularization functions assuming a group
%                 structure)
%            param.groups (int32, optional, for regularization functions assuming a group
%                 structure, see mexProximalFlat)
%            param.numThreads (optional, number of threads for exploiting
%                multi-core / multi-cpus. By default, it takes the value -1,
%                which automatically selects all the available CPUs/cores).
%            param.max_it (optional, maximum number of iterations, 100 by default)
%            param.it0 (optional, frequency for computing duality gap, every 10 iterations by default)
%            param.tol (optional, tolerance for stopping criteration, which is a relative duality gap
%                if it is available, or a relative change of parameters).
%            param.gamma (optional, multiplier for increasing the parameter L in fista, 1.5 by default)
%            param.L0 (optional, initial parameter L in fista, 0.1 by default, should be small enough)
%            param.fixed_step (deactive the line search for L in fista and use param.L0 instead)
%            param.compute_gram (optional, pre-compute X^TX, false by default).
%            param.intercept (optional, do not regularize last row of W, false by default).
%            param.ista (optional, use ista instead of fista, false by default).
%            param.subgrad (optional, if not param.ista, use subradient descent instead of fista, false by default).
%            param.a, param.b (optional, if param.subgrad, the gradient step is a/(t+b)
%            also similar options as mexProximalFlat
%
%            the function also implements the ADMM algorithm via an option param.admm=true. It is not documented
%            and you need to look at the source code to use it.
%
% Output:  W:  double dense p x n matrix or p x Nn matrix (for multi-logistic loss)
%          optim: optional, double dense 4 x n matrix.
%              first row: values of the objective functions.
%              third row: values of the relative duality gap (if available)
%              fourth row: number of iterations
%
% Author: Julien Mairal, 2010


