
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
 *                File mexIST.h
 * \brief mex-file, function mexCD
 * Usage: alpha = mexIST(X,D,alpha0,lambda,mode,itermax,tol); 
 * Usage: alpha = mexIST(X,D,alpha0,param);
 * mode, itermax, tol and p are optional
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
   if (mxIsSparse(prhs[1])) mexErrMsgTxt("argument 2 should be full");
   if (!mxIsSparse(prhs[2])) mexErrMsgTxt("argument 3 should be sparse");
   if (!mxIsStruct(prhs[3])) 
      mexErrMsgTxt("argument 4 should be struct");

   T* prX=reinterpret_cast<T*>(mxGetPr(prhs[0]));
   const mwSize* dims=mxGetDimensions(prhs[0]);
   INTM n=static_cast<INTM>(dims[0]);
   INTM M=static_cast<INTM>(dims[1]);

   T * prD = reinterpret_cast<T*>(mxGetPr(prhs[1]));
   const mwSize* dimsD=mxGetDimensions(prhs[1]);
   INTM nD=static_cast<INTM>(dimsD[0]);
   if (nD != n) mexErrMsgTxt("wrong size for argument 2");
   INTM K=static_cast<INTM>(dimsD[1]);

   const mwSize* dimsA=mxGetDimensions(prhs[2]);
   INTM Ka = static_cast<INTM>(dimsA[0]);
   INTM Ma = static_cast<INTM>(dimsA[1]);
   if (Ma != M || Ka != K) mexErrMsgTxt("wrong size for argument 3");
   double * alpha_v=static_cast<double*>(mxGetPr(prhs[2]));
   mwSize* alpha_r=mxGetIr(prhs[2]);
   mwSize* alpha_pB=mxGetJc(prhs[2]);
   mwSize* alpha_pE=alpha_pB+1;

   constraint_type mode = (constraint_type)getScalarStructDef<int>(prhs[3],"mode",PENALTY);
   int numThreads = getScalarStructDef<int>(prhs[3],"numThreads",-1);
   int itermax = getScalarStructDef<int>(prhs[3],"itermax",100);
   T tol = getScalarStructDef<T>(prhs[3],"tol",0.001);
   T lambda = getScalarStruct<T>(prhs[3],"lambda");

   INTM* alpha_r2, *alpha_pB2, *alpha_pE2;
   T* alpha_v2;
   createCopySparse<T>(alpha_v2,alpha_r2,alpha_pB2,alpha_pE2,
         alpha_v,alpha_r,alpha_pB,alpha_pE,M);

   Matrix<T> X(prX,n,M);
   Matrix<T> D(prD,n,K);
   SpMatrix<T> alpha(alpha_v2,alpha_r2,alpha_pB2,
         alpha_pE2,K,M,alpha_pB2[M]);

   ist<T>(X,D,alpha,lambda,mode,itermax,tol,numThreads);

   convertSpMatrix(plhs[0],K,M,alpha.n(),alpha.nzmax(),alpha.v(),alpha.r(),
         alpha.pB());

   deleteCopySparse<T>(alpha_v2,alpha_r2,alpha_pB2,alpha_pE2,
         alpha_v,alpha_r);
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

