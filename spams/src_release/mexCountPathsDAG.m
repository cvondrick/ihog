% 
% Usage:   num=mexCountPathsDAG(G);
%
% Name: mexCountPathsDAG
%
% Description: mexCountPathsDAG counts the number of paths 
%       in a DAG.
%
%       for a graph G with |V| nodes and |E| arcs,
%       G is a double sparse adjacency matrix of size |V|x|V|,
%       with |E| non-zero entries.
%       (see example in test_CountPathsDAG.m
%
%
% Inputs: G:  double sparse |V| x |V| matrix (full graph)
%
% Output: num: number of paths
%
% Author: Julien Mairal, 2012


