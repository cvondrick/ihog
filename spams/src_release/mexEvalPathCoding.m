% 
% Usage:   [val [paths]]=mexEvalPathCoding(U,DAG,param);
%
% Name: mexEvalPathCoding
%
% Description: mexEvalPathCoding evaluate the path coding penalies 
%         of http://arxiv.org/abs/1204.4539 and provides a path 
%         decomposition of a vector W.
%
%         Given an input matrix U=[u^1,\ldots,u^n], 
%
%
% Inputs: U:  double p x n matrix   (input signals)
%               m is the signal size
%         DAG:  struct
%               with three fields, weights, start_weights, stop_weights
%         for a graph with |V| nodes and |E| arcs,
%         DAG.weights: sparse double |V| x |V| matrix. Adjacency
%               matrix. The non-zero entries represent costs on arcs
%               linking two nodes.
%         DAG.start_weights: dense double |V| vector. Represent the costs
%               of starting a path from a specific node.
%         DAG.stop_weights: dense double |V| vector. Represent the costs
%               of ending a path at a specific node.
%
%         if param.regul='graph-path-l0', non-convex penalty
%         if param.regul='graph-path-conv', convex penalty
%
%         param: struct
%               param.regul (choice of regularization, see above)
%               param.verbose (optional, verbosity level, false by default)
%               param.precision (optional, by default a very large integer.
%                 It returns approximate proximal operator by choosing a small integer,
%                 for example, 100 or 1000.
%               param.numThreads (optional, number of threads for exploiting
%                 multi-core / multi-cpus. By default, it takes the value -1,
%                 which automatically selects all the available CPUs/cores).
%
% Output: V: double 1 x n vector (values of the objective function)
%         paths: optional, double sparse p x k matrix. selected paths for the 
%                first column of U
%
% Author: Julien Mairal, 2012


