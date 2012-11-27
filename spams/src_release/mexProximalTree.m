% 
% Usage:  [V [val_regularizer]]=mexProximalTree(U,tree,param);
%
% Name: mexProximalTree
%
% Description: mexProximalTree computes a proximal operator. Depending
%         on the value of param.regul, it computes 
%
%         Given an input matrix U=[u^1,\ldots,u^n], and a tree-structured set of groups T,
%         it returns a matrix V=[v^1,\ldots,v^n]:
%         
%         when the regularization function is for vectors,
%         for every column u of U, it compute a column v of V solving
%         if param.regul='tree-l0'
%             argmin 0.5||u-v||_2^2 + lambda \sum_{g \in T} \delta^g(v)
%         if param.regul='tree-l2'
%           for all i, v^i = 
%             argmin 0.5||u-v||_2^2 + lambda\sum_{g \in T} \eta_g||v_g||_2
%         if param.regul='tree-linf'
%           for all i, v^i = 
%             argmin 0.5||u-v||_2^2 + lambda\sum_{g \in T} \eta_g||v_g||_inf
%
%         when the regularization function is for matrices:
%         if param.regul='multi-task-tree'
%            V=argmin 0.5||U-V||_F^2 + lambda \sum_{i=1}^n\sum_{g \in T} \eta_g||v^i_g||_inf + ...
%                                                lambda_2 \sum_{g \in T} \eta_g max_{j in g}||V_j||_{inf}
%         
%         it can also be used with any non-tree-structured regularization addressed by mexProximalFlat
%
%         for all these regularizations, it is possible to enforce non-negativity constraints
%         with the option param.pos, and to prevent the last row of U to be regularized, with
%         the option param.intercept
%
% Inputs: U:  double m x n matrix   (input signals)
%               m is the signal size
%         tree: struct
%               with four fields, eta_g, groups, own_variables and N_own_variables.
%
%               The tree structure requires a particular organization of groups and variables
%                  * Let us denote by N = |T|, the number of groups.
%                    the groups should be ordered T={g1,g2,\ldots,gN} such that if gi is included
%                    in gj, then j <= i. g1 should be the group at the root of the tree 
%                    and contains every variable.
%                  * Every group is a set of  contiguous indices for instance 
%                    gi={3,4,5} or gi={4,5,6,7} or gi={4}, but not {3,5};
%                  * We define root(gi) as the indices of the variables that are in gi,
%                    but not in its descendants. For instance for
%                    T={ g1={1,2,3,4},g2={2,3},g3={4} }, then, root(g1)={1}, 
%                    root(g2)={2,3}, root(g3)={4},
%                    We assume that for all i, root(gi) is a set of contigous variables
%                  * We assume that the smallest of root(gi) is also the smallest index of gi.
%
%                  For instance, 
%                    T={ g1={1,2,3,4},g2={2,3},g3={4} }, is a valid set of groups.
%                    but we can not have
%                    T={ g1={1,2,3,4},g2={1,2},g3={3} }, since root(g1)={4} and 4 is not the
%                    smallest element in g1.
%
%               We do not lose generality with these assumptions since they can be fullfilled for any
%               tree-structured set of groups after a permutation of variables and a correct ordering of the
%               groups.
%               see more examples in test_ProximalTree.m of valid tree-structured sets of groups.
%               
%               The first fields sets the weights for every group
%                  tree.eta_g            double N vector 
%  
%               The next field sets inclusion relations between groups 
%               (but not between groups and variables):
%                  tree.groups           sparse (double or boolean) N x N matrix  
%                  the (i,j) entry is non-zero if and only if i is different than j and 
%                  gi is included in gj.
%                  the first column corresponds to the group at the root of the tree.
%
%               The next field define the smallest index of each group gi, 
%               which is also the smallest index of root(gi)
%               tree.own_variables    int32 N vector
%
%               The next field define for each group gi, the size of root(gi)
%               tree.N_own_variables  int32 N vector 
%
%               examples are given in test_ProximalTree.m
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
%               param.transpose (optional, transpose the matrix in the regularization function)
%               param.size_group (optional, for regularization functions assuming a group
%                 structure). It is a scalar. When param.groups is not specified, it assumes
%                 that the groups are the sets of consecutive elements of size param.size_group
%               param.numThreads (optional, number of threads for exploiting
%                 multi-core / multi-cpus. By default, it takes the value -1,
%                 which automatically selects all the available CPUs/cores).
%
% Output: V: double m x n matrix (output coefficients)
%         val_regularizer: double 1 x n vector (value of the regularization
%         term at the optimum).
%
%
% Author: Julien Mairal, 2010


