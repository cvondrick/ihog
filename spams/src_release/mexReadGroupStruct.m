% 
% Usage:   gstruct =mexReadGroupStruct(file)
%
% Name: mexReadGroupStruct
%
% Description: reads a text file describing "simply" the structure of groups
%    of variables needed by mexProximalGraph, mexProximalTree, mexFistaGraph,
%    mexFistaTree and mexStructTrainDL and builds the corresponding group structure.
%    weight is a float
%    variables-list : a space separated list of integers, maybe empty,
%        but '[' and '] must be present. Numbers in the range (0 - Nv-1)
%    children-list : a space separated list of node-id's
%        If the list is empty, '->' may be omitted.
%    The data must obey some rules : 
%        - A group contains the variables of the corresponding node and of the whole subtree.
%        - Variables attached to a node are those that are not int the subtree.
%        - If the data destination is a Graph, there may be several independant trees,
%           and a varibale may appear in several trees.
%    If the destination is a Tree, there must be only one tree, the root node
%        must have id == 0 and each variable must appear only once.
%
% Inputs: file:  the file name
%
% Output: groups: cell array, one element for each node
%                an element is itsel a 4 elements cell array:
%	         nodeid (int >= 0), weight (double), array of vars of the node,
%                array of children (nodeid's)
%
% Author: Jean-Paul CHIEZE, 2012
