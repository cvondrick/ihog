% 
% Usage:   [D [model]]=mexTrainDL(X,param[,model]);
%          model is optional
%
% Name: mexTrainDL
%
% Description: mexTrainDL is an efficient implementation of the
%     dictionary learning technique presented in
%
%     "Online Learning for Matrix Factorization and Sparse Coding"
%     by Julien Mairal, Francis Bach, Jean Ponce and Guillermo Sapiro
%     arXiv:0908.0050
%     
%     "Online Dictionary Learning for Sparse Coding"      
%     by Julien Mairal, Francis Bach, Jean Ponce and Guillermo Sapiro
%     ICML 2009.
%
%     Note that if you use param.mode=1 or 2, if the training set has a
%     reasonable size and you have enough memory on your computer, you 
%     should use mexTrainDL_Memory instead.
% 
%
%     It addresses the dictionary learning problems
%        1) if param.mode=0
%     min_{D in C} (1/n) sum_{i=1}^n (1/2)||x_i-Dalpha_i||_2^2  s.t. ...
%                                                  ||alpha_i||_1 <= lambda
%        2) if param.mode=1
%     min_{D in C} (1/n) sum_{i=1}^n  ||alpha_i||_1  s.t.  ...
%                                           ||x_i-Dalpha_i||_2^2 <= lambda
%        3) if param.mode=2
%     min_{D in C} (1/n) sum_{i=1}^n (1/2)||x_i-Dalpha_i||_2^2 + ... 
%                                  lambda||alpha_i||_1 + lambda_2||alpha_i||_2^2
%        4) if param.mode=3, the sparse coding is done with OMP
%     min_{D in C} (1/n) sum_{i=1}^n (1/2)||x_i-Dalpha_i||_2^2  s.t. ... 
%                                                  ||alpha_i||_0 <= lambda
%        5) if param.mode=4, the sparse coding is done with OMP
%     min_{D in C} (1/n) sum_{i=1}^n  ||alpha_i||_0  s.t.  ...
%                                           ||x_i-Dalpha_i||_2^2 <= lambda
%        6) if param.mode=5, the sparse coding is done with OMP
%     min_{D in C} (1/n) sum_{i=1}^n 0.5||x_i-Dalpha_i||_2^2 +lambda||alpha_i||_0  
%                                           
%
%%     C is a convex set verifying
%        1) if param.modeD=0
%           C={  D in Real^{m x p}  s.t.  forall j,  ||d_j||_2^2 <= 1 }
%        2) if param.modeD=1
%           C={  D in Real^{m x p}  s.t.  forall j,  ||d_j||_2^2 + ... 
%                                                  gamma1||d_j||_1 <= 1 }
%        3) if param.modeD=2
%           C={  D in Real^{m x p}  s.t.  forall j,  ||d_j||_2^2 + ... 
%                                  gamma1||d_j||_1 + gamma2 FL(d_j) <= 1 }
%        4) if param.modeD=3
%           C={  D in Real^{m x p}  s.t.  forall j,  (1-gamma1)||d_j||_2^2 + ... 
%                                  gamma1||d_j||_1 <= 1 }

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
%           param.lambda2  (optional, by default 0)
%           param.iter (number of iterations).  If a negative number is 
%              provided it will perform the computation during the
%              corresponding number of seconds. For instance param.iter=-5
%              learns the dictionary during 5 seconds.
%           param.mode (optional, see above, by default 2) 
%           param.posAlpha (optional, adds positivity constraints on the
%             coefficients, false by default, not compatible with 
%             param.mode =3,4)
%           param.modeD (optional, see above, by default 0)
%           param.posD (optional, adds positivity constraints on the 
%             dictionary, false by default, not compatible with 
%             param.modeD=2)
%           param.gamma1 (optional parameter for param.modeD >= 1)
%           param.gamma2 (optional parameter for param.modeD = 2)
%           param.batchsize (optional, size of the minibatch, by default 
%              512)
%           param.iter_updateD (optional, number of BCD iterations for the dictionary
%              update step, by default 1)
%           param.modeParam (optimization mode).
%              1) if param.modeParam=0, the optimization uses the 
%                 parameter free strategy of the ICML paper
%              2) if param.modeParam=1, the optimization uses the 
%                 parameters rho as in arXiv:0908.0050
%              3) if param.modeParam=2, the optimization uses exponential 
%                 decay weights with updates of the form 
%                 A_{t} <- rho A_{t-1} + alpha_t alpha_t^T
%           param.rho (optional) tuning parameter (see paper arXiv:0908.0050)
%           param.t0 (optional) tuning parameter (see paper arXiv:0908.0050)
%           param.clean (optional, true by default. prunes 
%              automatically the dictionary from unused elements).
%           param.verbose (optional, true by default, increase verbosity)
%           param.numThreads (optional, number of threads for exploiting
%              multi-core / multi-cpus. By default, it takes the value -1,
%              which automatically selects all the available CPUs/cores).
%
% Output: 
%         param.D: double m x p matrix   (dictionary)
%
% Note: this function admits a few experimental usages, which have not
%     been extensively tested:
%         - single precision setting 
%
% Author: Julien Mairal, 2009


