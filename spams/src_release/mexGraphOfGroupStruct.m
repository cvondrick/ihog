% 
% Usage:   groups =mexGraphOfGroupStruct(gstruct)
%
% Name: mexGraphOfGroupStruct
%
% Description: converts a group structure into the graph structure
%    used by mexProximalGraph, mexFistaGraph or mexStructTrainDL
%
% Inputs: gstruct: the structure of groups as a cell array, one element per node
%     an element is itself a 4 elements cell array:
%       nodeid (>= 0), weight (double), array of vars associated to the node,
%       array of children (nodeis's)
% Output: graph: struct (see documentation of mexProximalGraph)
%
% Author: Jean-Paul CHIEZE, 2012
