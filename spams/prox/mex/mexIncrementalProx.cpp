/* Software SPAMS v2.4 - Copyright 2009-2013 Julien Mairal 
 *
 * This file is part of SPAMS.
 *
 * SPAMS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPAMS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPAMS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <mex.h>
#include <mexutils.h>
#include <surrogate.h>

// mexStochSurrogate(y,X,W0,param)
template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
      const int nlhs) {
   if (!mexCheckType<T>(prhs[2]))
      mexErrMsgTxt("type of argument 3 is not consistent");
   if (mxIsSparse(prhs[2]))
      mexErrMsgTxt("argument 3 should not be sparse");

   if (!mxIsStruct(prhs[3])) 
      mexErrMsgTxt("argument 4 should be a struct");

   Vector<T> y;
   getVector(prhs[0],y);
   INTM n = y.n();

   const mwSize* dimsX=mxGetDimensions(prhs[1]);
   INTM p=static_cast<INTM>(dimsX[0]);
   INTM nX=static_cast<INTM>(dimsX[1]);
   if (nX != n) mexErrMsgTxt("second argument should be p x n");

   Matrix<T> w0;
   getMatrix(prhs[2],w0);
   int pw = w0.m();
   if (pw != p) mexErrMsgTxt("third argument should be p x nlambda");

   mxArray *pr_lambdas = mxGetField(prhs[3],0,"lambda");
   if (!pr_lambdas) 
      mexErrMsgTxt("Missing field lambda");
   Vector<T> lambdas;
   getVector(pr_lambdas,lambdas);
   int nlambdas=lambdas.n();
   if (nlambdas != w0.n()) mexErrMsgTxt("third argument should be p x nlambda");

   plhs[0]=createMatrix<T>(p,nlambdas);
   T* pr_w=reinterpret_cast<T*>(mxGetPr(plhs[0]));
   Matrix<T> w;
   getMatrix(plhs[0],w);

   ParamSurrogate<T> param;
   param.num_threads = getScalarStructDef<int>(prhs[3],"numThreads",-1);
   const int seed=getScalarStructDef<int>(prhs[3],"seed",0);
   param.strategy=getScalarStructDef<int>(prhs[3],"strategy",3);
   srandom(seed);
   param.epochs = getScalarStruct<long>(prhs[3],"epochs");
   const bool seq = getScalarStructDef<bool>(prhs[3],"warm_restart",false);
   param.minibatches = getScalarStructDef<int>(prhs[3],"minibatches",1);
   param.normalized = getScalarStructDef<bool>(prhs[3],"normalized",false);
   param.verbose = getScalarStructDef<bool>(prhs[3],"verbose",false);

   ParamFISTA<T> paramprox;
   getStringStruct(prhs[3],"regul",paramprox.name_regul,paramprox.length_names);
   paramprox.regul = regul_from_string(paramprox.name_regul);
   if (paramprox.regul==INCORRECT_REG)
      mexErrMsgTxt("Unknown regularization");
   getStringStruct(prhs[3],"loss",paramprox.name_loss,paramprox.length_names);
   paramprox.loss = loss_from_string(paramprox.name_loss);
   if (paramprox.loss==INCORRECT_LOSS)
      mexErrMsgTxt("Unknown loss");

   if (param.num_threads == -1) {
#ifdef _OPENMP
      init_omp(MIN(MAX_THREADS,omp_get_num_procs()));
#endif
   } else {
#ifdef _OPENMP
      init_omp(param.num_threads);
#endif
   }

   Matrix<T> optim;
   if (nlhs==2) {
      plhs[1]=createMatrix<T>(3,nlambdas);
      getMatrix<T>(plhs[1],optim);
   } else {
      optim.resize(3,nlambdas);
   }
   if (mxIsSparse(prhs[1])) {
      double* X_v;
      mwSize* X_r, *X_pB, *X_pE;
      INTM* X_r2, *X_pB2, *X_pE2;
      T* X_v2;
      X_v=static_cast<double*>(mxGetPr(prhs[1]));
      X_r=mxGetIr(prhs[1]);
      X_pB=mxGetJc(prhs[1]);
      X_pE=X_pB+1;
      createCopySparse<T>(X_v2,X_r2,X_pB2,X_pE2,
            X_v,X_r,X_pB,X_pE,n);
      SpMatrix<T> X(X_v2,X_r2,X_pB2,X_pE2,p,n,X_pB2[n]);
      if (seq) {
         incrementalProximalSeq(y,X,w0,w,paramprox,param,lambdas,optim);
      } else {
         incrementalProximal(y,X,w0,w,paramprox,param,lambdas,optim);
      }
      deleteCopySparse<T>(X_v2,X_r2,X_pB2,X_pE2,X_v,X_r);
   } else {
      T* prX = reinterpret_cast<T*>(mxGetPr(prhs[1]));
      Matrix<T> X(prX,p,n);
      if (seq) {
         incrementalProximalSeq(y,X,w0,w,paramprox,param,lambdas,optim);
      } else {
         incrementalProximal(y,X,w0,w,paramprox,param,lambdas,optim);
      }
   }
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
   if (nrhs != 4)
      mexErrMsgTxt("Bad number of inputs arguments");

   if (nlhs != 1 && nlhs != 2) 
      mexErrMsgTxt("Bad number of output arguments");

   if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
      callFunction<double>(plhs,prhs,nlhs);
   } else {
      callFunction<float>(plhs,prhs,nlhs);
   }
}




