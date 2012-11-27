% 
% Usage:   [V [val_regularizer]]=mexProximalGraph(U,graph,param);
%
% Name: mexProximalGraph
%
% Description: mexProximalGraph computes a proximal operator. Depending
%         on the value of param.regul, it computes 
%
%         Given an input matrix U=[u^1,\ldots,u^n], and a set of groups G,
%         it computes a matrix V=[v^1,\ldots,v^n] such that
%
%         if param.regul='graph'
%         for every column u of U, it computes a column v of V solving
%             argmin 0.5||u-v||_2^2 + lambda\sum_{g \in G} \eta_g||v_g||_inf
%
%         if param.regul='graph+ridge'
%         for every column u of U, it computes a column v of V solving
%             argmin 0.5||u-v||_2^2 + lambda\sum_{g \in G} \eta_g||v_g||_inf + lambda_2||v||_2^2
%
%
%         if param.regul='multi-task-graph'
%            V=argmin 0.5||U-V||_F^2 + lambda \sum_{i=1}^n\sum_{g \in G} \eta_g||v^i_g||_inf + ...
%                                                lambda_2 \sum_{g \in G} \eta_g max_{j in g}||V_j||_{inf}
%         
%         it can also be used with any regularization addressed by mexProximalFlat
%
%         for all these regularizations, it is possible to enforce non-negativity constraints
%         with the option param.pos, and to prevent the last row of U to be regularized, with
%         the option param.intercept
%
% Inputs: U:  double p x n matrix   (input signals)
%               m is the signal size
%         graph: struct
%               with three fields, eta_g, groups, and groups_var
%
%               The first fields sets the weights for every group
%                  graph.eta_g            double N vector 
%  
%               The next field sets inclusion relations between groups 
%               (but not between groups and variables):
%                  graph.groups           sparse (double or boolean) N x N matrix  
%                  the (i,j) entry is non-zero if and only if i is different than j and 
%                  gi is included in gj.
%               
%               The next field sets inclusion relations between groups and variables
%                  graph.groups_var       sparse (double or boolean) p x N matrix
%                  the (i,j) entry is non-zero if and only if the variable i is included 
%                  in gj, but not in any children of gj.
%
%               examples are given in test_ProximalGraph.m
%
%         param: struct
%               param.lambda  (regularization parameter)
%               param.regul (choice of regularization, see above)
%               param.lambda2  (optional, regularization parameter)
%               param.lambda3  (optional, regularization parameter)
%               param.verbose (optional, verbosity level, false by default)
%               param.intercept (optional, last row of U is not regularized,
%                 false by default)
%               param.pos (optional, adds positivity constraints on the
%                 coefficients, false by default)
%               param.numThreads (optional, number of threads for exploiting
%                 multi-core / multi-cpus. By default, it takes the value -1,
%                 which automatically selects all the available CPUs/cores).
%
% Output: V: double p x n matrix (output coefficients)
%         val_regularizer: double 1 x n vector (value of the regularization
%         term at the optimum).
%
% Author: Julien Mairal, 2010


