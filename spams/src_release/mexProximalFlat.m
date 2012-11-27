% 
% Usage:  [V [val_regularizer]]=mexProximalFlat(U,param);
%
% Name: mexProximalFlat
%
% Description: mexProximalFlat computes proximal operators. Depending
%         on the value of param.regul, it computes 
%
%         Given an input matrix U=[u^1,\ldots,u^n], it computes a matrix 
%         V=[v^1,\ldots,v^n] such that
%         if one chooses a regularization functions on vectors, it computes
%         for each column u of U, a column v of V solving
%         if param.regul='l0'
%             argmin 0.5||u-v||_2^2 + lambda||v||_0
%         if param.regul='l1'
%             argmin 0.5||u-v||_2^2 + lambda||v||_1
%         if param.regul='l2'
%             argmin 0.5||u-v||_2^2 + 0.5lambda||v||_2^2
%         if param.regul='elastic-net'
%             argmin 0.5||u-v||_2^2 + lambda||v||_1 + lambda_2||v||_2^2
%         if param.regul='fused-lasso'
%             argmin 0.5||u-v||_2^2 + lambda FL(v) + ...
%                               ...  lambda_2||v||_1 + lambda_3||v||_2^2
%         if param.regul='linf'
%             argmin 0.5||u-v||_2^2 + lambda||v||_inf
%         if param.regul='l1-constraint'
%             argmin 0.5||u-v||_2^2 s.t. ||v||_1 <= lambda
%         if param.regul='l2-not-squared'
%             argmin 0.5||u-v||_2^2 + lambda||v||_2
%         if param.regul='group-lasso-l2'  
%             argmin 0.5||u-v||_2^2 + lambda sum_g ||v_g||_2 
%             where the groups are either defined by param.groups or by param.size_group,
%         if param.regul='group-lasso-linf'
%             argmin 0.5||u-v||_2^2 + lambda sum_g ||v_g||_inf
%         if param.regul='sparse-group-lasso-l2'  
%             argmin 0.5||u-v||_2^2 + lambda sum_g ||v_g||_2 + lambda_2 ||v||_1
%             where the groups are either defined by param.groups or by param.size_group,
%         if param.regul='sparse-group-lasso-linf'
%             argmin 0.5||u-v||_2^2 + lambda sum_g ||v_g||_inf + lambda_2 ||v||_1
%         if param.regul='trace-norm-vec' 
%             argmin 0.5||u-v||_2^2 + lambda ||mat(v)||_* 
%            where mat(v) has param.size_group rows
%
%         if one chooses a regularization function on matrices
%         if param.regul='l1l2',  V= 
%             argmin 0.5||U-V||_F^2 + lambda||V||_{1/2}
%         if param.regul='l1linf',  V= 
%             argmin 0.5||U-V||_F^2 + lambda||V||_{1/inf}
%         if param.regul='l1l2+l1',  V= 
%             argmin 0.5||U-V||_F^2 + lambda||V||_{1/2} + lambda_2||V||_{1/1}
%         if param.regul='l1linf+l1',  V= 
%             argmin 0.5||U-V||_F^2 + lambda||V||_{1/inf} + lambda_2||V||_{1/1}
%         if param.regul='l1linf+row-column',  V= 
%             argmin 0.5||U-V||_F^2 + lambda||V||_{1/inf} + lambda_2||V'||_{1/inf}
%         if param.regul='trace-norm',  V= 
%             argmin 0.5||U-V||_F^2 + lambda||V||_*
%         if param.regul='rank',  V= 
%             argmin 0.5||U-V||_F^2 + lambda rank(V)
%         if param.regul='none',  V= 
%             argmin 0.5||U-V||_F^2 
%         
%         for all these regularizations, it is possible to enforce non-negativity constraints
%         with the option param.pos, and to prevent the last row of U to be regularized, with
%         the option param.intercept
%
% Inputs: U:  double m x n matrix   (input signals)
%               m is the signal size
%         param: struct
%               param.lambda  (regularization parameter)
%               param.regul (choice of regularization, see above)
%               param.lambda2  (optional, regularization parameter)
%               param.lambda3  (optional, regularization parameter)
%               param.verbose (optional, verbosity level, false by default)
%               param.intercept (optional, last row of U is not regularized,
%                 false by default)
%               param.transpose (optional, transpose the matrix in the regularization function)
%               param.size_group (optional, for regularization functions assuming a group
%                 structure). It is a scalar. When param.groups is not specified, it assumes
%                 that the groups are the sets of consecutive elements of size param.size_group
%               param.groups (int32, optional, for regularization functions assuming a group
%                 structure. It is an int32 vector of size m containing the group indices of the
%                 variables (first group is 1).
%               param.pos (optional, adds positivity constraints on the
%                 coefficients, false by default)
%               param.numThreads (optional, number of threads for exploiting
%                 multi-core / multi-cpus. By default, it takes the value -1,
%                 which automatically selects all the available CPUs/cores).
%
% Output: V: double m x n matrix (output coefficients)
%         val_regularizer: double 1 x n vector (value of the regularization
%         term at the optimum).
%
% Author: Julien Mairal, 2010


