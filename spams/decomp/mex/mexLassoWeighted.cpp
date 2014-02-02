
/* Software SPAMS v2.1 - Copyright 2009-2011 Julien Mairal 
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

/*!
 * \file
 *                toolbox decomp
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File mexLasso.h
 * \brief mex-file, function mexLasso
 * Usage: alpha = mexLassoWeights(X,D,W,param); 
 * */


#include <mexutils.h>
#include <decomp.h>

template <typename T>
   inline void callFunction(mxArray* plhs[], const mxArray*prhs[]) {
      if (!mexCheckType<T>(prhs[0])) 
         mexErrMsgTxt("type of argument 1 is not consistent");
      if (mxIsSparse(prhs[0])) 
         mexErrMsgTxt("argument 1 should be full");
      if (!mexCheckType<T>(prhs[1])) 
         mexErrMsgTxt("type of argument 2 is not consistent");
      if (mxIsSparse(prhs[1])) 
         mexErrMsgTxt("argument 2 should be full");
      if (mxIsSparse(prhs[2])) 
         mexErrMsgTxt("argument 3 should be full");
      if (!mxIsStruct(prhs[3])) 
         mexErrMsgTxt("argument 4 should be struct");


      T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0]));
      const mwSize* dimsX=mxGetDimensions(prhs[0]);
      int n=static_cast<int>(dimsX[0]);
      int M=static_cast<int>(dimsX[1]);

      T* prD = reinterpret_cast<T*>(mxGetPr(prhs[1]));
      const mwSize* dimsD=mxGetDimensions(prhs[1]);
      int nD=static_cast<int>(dimsD[0]);
      int K=static_cast<int>(dimsD[1]);
      if (n != nD) mexErrMsgTxt("argument sizes are not consistent");

      T lambda = getScalarStruct<T>(prhs[3],"lambda");
      int L = getScalarStructDef<int>(prhs[3],"L",K);
      int numThreads = getScalarStructDef<int>(prhs[3],"numThreads",-1);
      bool pos = getScalarStructDef<bool>(prhs[3],"pos",false);
      constraint_type mode = (constraint_type)getScalarStructDef<int>(prhs[3],"mode",PENALTY);
      if (L > n) {
         printf("L is changed to %d\n",n);
         L=n;
      }
      if (L > K) {
         printf("L is changed to %d\n",K);
         L=K;
      }
      Matrix<T> X(prX,n,M);
      Matrix<T> D(prD,n,K);


      T* prWeight = reinterpret_cast<T*>(mxGetPr(prhs[2]));
      const mwSize* dimsW=mxGetDimensions(prhs[2]);
      int KK=static_cast<int>(dimsW[0]);
      int MM=static_cast<int>(dimsW[1]);
      if (K != KK || M != MM) mexErrMsgTxt("argument sizes are not consistent");


      Matrix<T> weight(prWeight,KK,MM);
      
      SpMatrix<T> alpha;
      lassoWeight<T>(X,D,weight,alpha,L,lambda,mode,pos,numThreads);
      convertSpMatrix(plhs[0],alpha.m(),alpha.n(),alpha.n(),
            alpha.nzmax(),alpha.v(),alpha.r(),alpha.pB());
   }

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs != 4)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (!(nlhs == 1 || nlhs == 1))
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs);
      } else {
         callFunction<float>(plhs,prhs);
      }
   }



