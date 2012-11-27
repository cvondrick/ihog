% 
% Usage:  V=mexSparseProject(U,param);
%
% Name: mexSparseProject
%
% Description: mexSparseProject solves various optimization 
%     problems, including projections on a few convex sets.
%     It aims at addressing the following problems
%     for all columns u of U in parallel
%       1) when param.mode=1 (projection on the l1-ball)
%           min_v ||u-v||_2^2  s.t.  ||v||_1 <= thrs
%       2) when param.mode=2
%           min_v ||u-v||_2^2  s.t. ||v||_2^2 + lamuda1||v||_1 <= thrs
%       3) when param.mode=3
%           min_v ||u-v||_2^2  s.t  ||v||_1 + 0.5lamuda1||v||_2^2 <= thrs 
%       4) when param.mode=4
%           min_v 0.5||u-v||_2^2 + lamuda1||v||_1  s.t  ||v||_2^2 <= thrs
%       5) when param.mode=5
%           min_v 0.5||u-v||_2^2 + lamuda1||v||_1 +lamuda2 FL(v) + ... 
%                                                   0.5lamuda_3 ||v||_2^2
%          where FL denotes a "fused lasso" regularization term.
%       6) when param.mode=6
%          min_v ||u-v||_2^2 s.t lamuda1||v||_1 +lamuda2 FL(v) + ...
%                                             0.5lamuda3||v||_2^2 <= thrs
%           
%        When param.pos=true and param.mode <= 4,
%        it solves the previous problems with positivity constraints 
%
% Inputs: U:  double m x n matrix   (input signals)
%               m is the signal size
%               n is the number of signals to project
%         param: struct
%           param.thrs (parameter)
%           param.lambda1 (parameter)
%           param.lambda2 (parameter)
%           param.lambda3 (parameter)
%           param.mode (see above)
%           param.pos (optional, false by default)
%           param.numThreads (optional, number of threads for exploiting
%             multi-core / multi-cpus. By default, it takes the value -1,
%             which automatically selects all the available CPUs/cores).
%
% Output: V: double m x n matrix (output matrix)
%
% Note: this function admits a few experimental usages, which have not
%     been extensively tested:
%         - single precision setting 
%
% Author: Julien Mairal, 2009


