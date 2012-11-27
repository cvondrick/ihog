% 
% Usage:   [U [,V]]=nmf(X,param);
%
% Name: nmf
%
% Description: mexTrainDL is an efficient implementation of the
%     non-negative matrix factorization technique presented in 
%
%     "Online Learning for Matrix Factorization and Sparse Coding"
%     by Julien Mairal, Francis Bach, Jean Ponce and Guillermo Sapiro
%     arXiv:0908.0050
%     
%     "Online Dictionary Learning for Sparse Coding"      
%     by Julien Mairal, Francis Bach, Jean Ponce and Guillermo Sapiro
%     ICML 2009.
%
%     Potentially, n can be very large with this algorithm.
%
% Inputs: X:  double m x n matrix   (input signals)
%               m is the signal size
%               n is the number of signals to decompose
%         param: struct
%            param.K (number of required factors)
%            param.iter (number of iterations).  If a negative number 
%              is provided it will perform the computation during the
%              corresponding number of seconds. For instance param.iter=-5
%              learns the dictionary during 5 seconds.
%            param.batchsize (optional, size of the minibatch, by default 
%               512)
%            param.modeParam (optimization mode).
%               1) if param.modeParam=0, the optimization uses the 
%                  parameter free strategy of the ICML paper
%               2) if param.modeParam=1, the optimization uses the 
%                  parameters rho as in arXiv:0908.0050
%               3) if param.modeParam=2, the optimization uses exponential 
%                  decay weights with updates of the form  
%                  A_{t} <- rho A_{t-1} + alpha_t alpha_t^T
%            param.rho (optional) tuning parameter (see paper 
%              arXiv:0908.0050)
%            param.t0 (optional) tuning parameter (see paper 
%              arXiv:0908.0050)
%            param.clean (optional, true by default. prunes automatically 
%              the dictionary from unused elements).
%            param.batch (optional, false by default, use batch learning 
%              instead of online learning)
%            param.numThreads (optional, number of threads for exploiting
%                 multi-core / multi-cpus. By default, it takes the value -1,
%                 which automatically selects all the available CPUs/cores).
%         model: struct (optional) learned model for "retraining" the data.
%
% Output:
%         U: double m x p matrix   
%         V: double p x n matrix   (optional)
%         model: struct (optional) learned model to be used for 
%           "retraining" the data.
%
% Author: Julien Mairal, 2009
function [U V] = nmf(X,param)

param.lambda=0;
param.mode=2;
param.posAlpha=1;
param.posD=1;
param.whiten=0;
U=mexTrainDL(X,param);
param.pos=1;
if nargout == 2
   if issparse(X) % todo allow sparse matrices X for mexLasso
      maxbatch=ceil(10000000/size(X,1));
      for jj = 1:maxbatch:size(X,2)
         indbatch=jj:min((jj+maxbatch-1),size(X,2));
         Xb=full(X(:,indbatch));
         V(:,indbatch)=mexLasso(Xb,U,param);
      end
   else
      V=mexLasso(X,U,param);
   end
end

