% 
% Usage:   [permutations groups nbvars] =mexTreeOfGroupStruct(gstruct)
%
% Name: mexTreeOfGroupStruct
%
% Description: converts a group structure into the tree structure
%    used by mexProximalTree, mexFistaTree or mexStructTrainDL
%
% Inputs: gstruct: the structure of groups as a cell array, one element per node
%     an element is itself a 4 lements cell array:
%       nodeid (>= 0), weight (double), array of vars associated to the node,
%       array of children (nodeis's)
% Output: permutations: permutation vector that must be applied to the result of the
%               programm using the tree. Empty if no permutation is needed.
%      tree: struct (see documentation of mexProximalTree)
%      nbvars : number of variables in the tree
%
% Author: Jean-Paul CHIEZE, 2012
