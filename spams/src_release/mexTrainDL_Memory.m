% 
% Usage:   [D]=mexTrainDL(X,param);
%
% Name: mexTrainDL_Memory
%
% Description: mexTrainDL_Memory is an efficient but memory consuming 
%     variant of the dictionary learning technique presented in
%
%     "Online Learning for Matrix Factorization and Sparse Coding"
%     by Julien Mairal, Francis Bach, Jean Ponce and Guillermo Sapiro
%     arXiv:0908.0050
%     
%     "Online Dictionary Learning for Sparse Coding"      
%     by Julien Mairal, Francis Bach, Jean Ponce and Guillermo Sapiro
%     ICML 2009.
%
%     Contrary to the approaches above, the algorithm here 
%        does require to store all the coefficients from all the training
%        signals. For this reason this variant can not be used with large
%        training sets, but is more efficient than the regular online
%        approach for training sets of reasonable size.
%
%     It addresses the dictionary learning problems
%        1) if param.mode=1
%     min_{D in C} (1/n) sum_{i=1}^n  ||alpha_i||_1  s.t.  ...
%                                         ||x_i-Dalpha_i||_2^2 <= lambda
%        2) if param.mode=2
%     min_{D in C} (1/n) sum_{i=1}^n (1/2)||x_i-Dalpha_i||_2^2 + ... 
%                                                      lambda||alpha_i||_1  
%
%     C is a convex set verifying
%        1) if param.modeD=0
%           C={  D in Real^{m x p}  s.t.  forall j,  ||d_j||_2^2 <= 1 }
%        1) if param.modeD=1
%           C={  D in Real^{m x p}  s.t.  forall j,  ||d_j||_2^2 + ... 
%                                                  gamma1||d_j||_1 <= 1 }
%        1) if param.modeD=2
%           C={  D in Real^{m x p}  s.t.  forall j,  ||d_j||_2^2 + ... 
%                                  gamma1||d_j||_1 + gamma2 FL(d_j) <= 1 }
%
%     Potentially, n can be very large with this algorithm.
%
% Inputs: X:  double m x n matrix   (input signals)
%               m is the signal size
%               n is the number of signals to decompose
%         param: struct
%            param.D: (optional) double m x p matrix   (dictionary)
%              p is the number of elements in the dictionary
%              When D is not provided, the dictionary is initialized 
%              with random elements from the training set.
%           param.K (size of the dictionary, optional is param.D is provided)
%           param.lambda  (parameter)
%           param.iter (number of iterations).  If a negative number is 
%              provided it will perform the computation during the
%              corresponding number of seconds. For instance param.iter=-5
%              learns the dictionary during 5 seconds.
%            param.mode (optional, see above, by default 2) 
%            param.modeD (optional, see above, by default 0)
%            param.posD (optional, adds positivity constraints on the 
%              dictionary, false by default, not compatible with 
%              param.modeD=2)
%            param.gamma1 (optional parameter for param.modeD >= 1)
%            param.gamma2 (optional parameter for param.modeD = 2)
%            param.batchsize (optional, size of the minibatch, by default 
%              512)
%            param.iter_updateD (optional, number of BCD iterations for the dictionary 
%                update step, by default 1)
%            param.modeParam (optimization mode).
%              1) if param.modeParam=0, the optimization uses the 
%                 parameter free strategy of the ICML paper
%              2) if param.modeParam=1, the optimization uses the 
%                 parameters rho as in arXiv:0908.0050
%              3) if param.modeParam=2, the optimization uses exponential 
%                 decay weights with updates of the form 
%                 A_{t} <- rho A_{t-1} + alpha_t alpha_t^T
%            param.rho (optional) tuning parameter (see paper arXiv:0908.0050)
%            param.t0 (optional) tuning parameter (see paper arXiv:0908.0050)
%            param.clean (optional, true by default. prunes 
%              automatically the dictionary from unused elements).
%            param.numThreads (optional, number of threads for exploiting
%              multi-core / multi-cpus. By default, it takes the value -1,
%              which automatically selects all the available CPUs/cores).
%
% Output: 
%         param.D: double m x p matrix   (dictionary)
%
% Note: this function admits a few experimental usages, which have not
%     been extensively tested:
%         - single precision setting (even though the output alpha is double 
%           precision)
%
% Author: Julien Mairal, 2009


