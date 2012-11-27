% 
% Usage:   DAG=mexRemoveCycleGraph(G);
%
% Name: mexRemoveCycleGraph
%
% Description: mexRemoveCycleGraph heuristically removes
%       arcs along cycles in the graph G to obtain a DAG.
%       the arcs of G can have weights. The heuristic will
%       remove in priority arcs with the smallest weights.
%
%       for a graph G with |V| nodes and |E| arcs,
%       G is a double sparse adjacency matrix of size |V|x|V|,
%       with |E| non-zero entries. The non-zero entries correspond
%       to the weights of the arcs.
%
%       DAG is a also double sparse adjacency matrix of size |V|x|V|,
%       but the graph is acyclic.
%
%       Note that another heuristic to obtain a DAG from a general 
%       graph consists of ordering the vertices.
%
% Inputs: G:  double sparse |V| x |V| matrix
%
% Output: DAG:  double sparse |V| x |V| matrix
%
% Author: Julien Mairal, 2012


