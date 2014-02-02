
/* Software SPAMS v2.3 - Copyright 2009-2012 Julien Mairal 
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
 *                File mexOMPMask.h
 * \brief mex-file, function mexOMPMask
 * Usage: [alpha path] = mexOMP(X,D,mask,param); */



#include <mexutils.h>
#include <decomp.h>

template <typename T>
   inline void callFunction(mxArray* plhs[], const mxArray*prhs[], 
         const int nlhs) {
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

      Matrix<T> X(prX,n,M);
      Matrix<bool> mask(prmask,n,M);
      Matrix<T> D(prD,n,K);
      SpMatrix<T> alpha;

      int numThreads = getScalarStructDef<int>(prhs[3],"numThreads",-1);
      mxArray* pr_L=mxGetField(prhs[3],0,"L");
      mxArray* pr_eps=mxGetField(prhs[3],0,"eps");
      mxArray* pr_lambda=mxGetField(prhs[3],0,"lambda");
      if (!pr_L && !pr_eps && !pr_lambda) mexErrMsgTxt("You should either provide L, eps or lambda");
      
      int sizeL = 1;
      int L=MIN(n,K);
      int *pL = &L;
      if (pr_L) {
         const mwSize* dimsL= mxGetDimensions(pr_L);
         sizeL=static_cast<int>(dimsL[0])*static_cast<int>(dimsL[1]);
         if (sizeL > 1) {
            if (!mexCheckType<int>(pr_L)) 
               mexErrMsgTxt("Type of param.L should be int32");
            pL = reinterpret_cast<int*>(mxGetPr(pr_L));
         }
         L=MIN(L,static_cast<int>(mxGetScalar(pr_L)));
      }

      int sizeE=1;
      T eps=0;
      T* pE=&eps;
      if (pr_eps) {
         const mwSize* dimsE=mxGetDimensions(pr_eps);
         sizeE=static_cast<int>(dimsE[0])*static_cast<int>(dimsE[1]);
         eps=static_cast<T>(mxGetScalar(pr_eps));
         if (sizeE > 1)
            pE = reinterpret_cast<T*>(mxGetPr(pr_eps));
      }

      T lambda=0;
      int sizeLambda=1;
      T* pLambda=&lambda;
      if (pr_lambda) {
         const mwSize* dimsLambda=mxGetDimensions(pr_lambda);
         sizeLambda=static_cast<int>(dimsLambda[0])*static_cast<int>(dimsLambda[1]);
         lambda=static_cast<T>(mxGetScalar(pr_lambda));
         if (sizeLambda > 1)
            pLambda = reinterpret_cast<T*>(mxGetPr(pr_lambda));
      }

      Matrix<T>* prPath=NULL;
      if (nlhs == 2) {
         plhs[1]=createMatrix<T>(K,L);
         T* pr_path=reinterpret_cast<T*>(mxGetPr(plhs[1]));
         Matrix<T> path(pr_path,K,L);
         path.setZeros();
         prPath=&path;
      }
      omp_mask<T>(X,D,alpha,mask,pL,pE,pLambda,sizeL > 1,sizeE > 1,sizeLambda > 1,
            numThreads,prPath);
      convertSpMatrix(plhs[0],K,M,alpha.n(),alpha.nzmax(),alpha.v(),alpha.r(),
            alpha.pB());
   }

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
   if (nrhs != 4)
      mexErrMsgTxt("Bad number of inputs arguments");

   if (nlhs != 1 && nlhs != 2) 
      mexErrMsgTxt("Bad number of output arguments");

   if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
      callFunction<double>(plhs,prhs, nlhs);
   } else {
      callFunction<float>(plhs,prhs, nlhs);
   }
}




