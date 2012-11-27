% 
% Usage:   num=mexCountConnexComponents(G,N);
%
% Name: mexCountConnexComponents
%
% Description: mexCountConnexComponents counts the number of connected
%       components of the subgraph of G corresponding to set of nodes
%       in N. In other words, the subgraph of G by removing from G all
%       the nodes which are not in N.
%
%       for a graph G with |V| nodes and |E| arcs,
%       G is a double sparse adjacency matrix of size |V|x|V|,
%       with |E| non-zero entries.
%       (see example in test_CountConnexComponents.m)
%       N is a dense vector of size |V|. if  N[i] is non-zero,
%       it means that the node i is selected.
%
%
% Inputs: G:  double sparse |V| x |V| matrix (full graph)
%         N:  double full |V| vector.
%
% Output: num: number of connected components.
%
% Author: Julien Mairal, 2012


