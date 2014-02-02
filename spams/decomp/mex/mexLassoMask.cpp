
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
 * Usage: alpha = mexLasso(X,D,mask,param); 
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
   if (!mexCheckType<bool>(prhs[2])) 
      mexErrMsgTxt("type of argument 3 should be boolean");
   if (mxIsSparse(prhs[2])) 
      mexErrMsgTxt("argument 3 should be full");
   if (!mxIsStruct(prhs[3])) 
      mexErrMsgTxt("argument 4 should be struct");

   T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0]));
   const mwSize* dimsX=mxGetDimensions(prhs[0]);
   INTM n=static_cast<INTM>(dimsX[0]);
   INTM M=static_cast<INTM>(dimsX[1]);

   T* prD = reinterpret_cast<T*>(mxGetPr(prhs[1]));
   const mwSize* dimsD=mxGetDimensions(prhs[1]);
   INTM nD=static_cast<INTM>(dimsD[0]);
   INTM K=static_cast<INTM>(dimsD[1]);
   if (n != nD) mexErrMsgTxt("argument sizes are not consistent");

   bool* prmask = reinterpret_cast<bool*>(mxGetPr(prhs[2]));
   const mwSize* dimsM=mxGetDimensions(prhs[2]);
   INTM nM=static_cast<INTM>(dimsM[0]);
   INTM mM=static_cast<INTM>(dimsM[1]);
   if (nM != n || mM != M) mexErrMsgTxt("argument sizes are not consistent");

   T lambda = getScalarStruct<T>(prhs[3],"lambda");
   T lambda2 = getScalarStructDef<T>(prhs[3],"lambda2",0);
   int L = getScalarStructDef<int>(prhs[3],"L",K);
   int numThreads = getScalarStructDef<int>(prhs[3],"numThreads",-1);
   bool pos = getScalarStructDef<bool>(prhs[3],"pos",false);
   bool verbose = getScalarStructDef<bool>(prhs[3],"verbose",true);
   constraint_type mode = (constraint_type)getScalarStructDef<int>(prhs[3],"mode",PENALTY);
   if (L > n && !(mode == PENALTY && isZero(lambda) && !pos && lambda2 > 0)) {
      if (verbose)
         printf("L is changed to %d\n",(int)n);
      L=n;
   }
   if (L > K) {
      if (verbose)
         printf("L is changed to %d\n",(int)K);
      L=K;
   }
   Matrix<T> X(prX,n,M);
   Matrix<T> D(prD,n,K);
   Matrix<bool> mask(prmask,n,M);
   SpMatrix<T> alpha;

   lasso_mask<T>(X,D,alpha,mask,L,lambda,lambda2,mode,pos,numThreads);
   convertSpMatrix(plhs[0],alpha.m(),alpha.n(),alpha.n(),
         alpha.nzmax(),alpha.v(),alpha.r(),alpha.pB());
}

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs != 4)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (nlhs != 1)
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs);
      } else {
         callFunction<float>(plhs,prhs);
      }
   }




