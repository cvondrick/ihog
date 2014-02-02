
/* Software SPAMS v2.2 - Copyright 2009-2011 Julien Mairal 
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
 * Usage: alpha = mexLasso(X,D,param); 
 * Usage: alpha = mexLasso(X,G,DtR,param); 
 * */

#include <mexutils.h>
#include <decomp.h>

template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[],const int nrhs,
      const int nlhs) {
   if (nrhs==3) {
      if (!mexCheckType<T>(prhs[0])) 
         mexErrMsgTxt("type of argument 1 is not consistent");
      if (mxIsSparse(prhs[0])) 
         mexErrMsgTxt("argument 1 should be full");
      if (!mexCheckType<T>(prhs[1])) 
         mexErrMsgTxt("type of argument 2 is not consistent");
      if (mxIsSparse(prhs[1])) 
         mexErrMsgTxt("argument 2 should be full");
      if (!mxIsStruct(prhs[2])) 
         mexErrMsgTxt("argument 3 should be struct");

      T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0]));
      const mwSize* dimsX=mxGetDimensions(prhs[0]);
      int n=static_cast<int>(dimsX[0]);
      int M=static_cast<int>(dimsX[1]);

      T* prD = reinterpret_cast<T*>(mxGetPr(prhs[1]));
      const mwSize* dimsD=mxGetDimensions(prhs[1]);
      int nD=static_cast<int>(dimsD[0]);
      int K=static_cast<int>(dimsD[1]);
      if (n != nD) mexErrMsgTxt("argument sizes are not consistent");
      T lambda = getScalarStruct<T>(prhs[2],"lambda");
      T lambda2 = getScalarStructDef<T>(prhs[2],"lambda2",0);
      int L = getScalarStructDef<int>(prhs[2],"L",K);
      int length_path = MAX(2,getScalarStructDef<int>(prhs[2],"length_path",4*L));
      int numThreads = getScalarStructDef<int>(prhs[2],"numThreads",-1);
      bool pos = getScalarStructDef<bool>(prhs[2],"pos",false);
      bool verbose = getScalarStructDef<bool>(prhs[2],"verbose",false);
      bool ols = getScalarStructDef<bool>(prhs[2],"ols",false);
      bool cholesky = ols || getScalarStructDef<bool>(prhs[2],"cholesky",false);
      constraint_type mode = (constraint_type)getScalarStructDef<int>(prhs[2],"mode",PENALTY);
      if (L > n && !(mode == PENALTY && isZero(lambda) && !pos && lambda2 > 0)) {
//         if (verbose)
//            printf("L is changed to %d\n",n);
         L=n;
      }
      if (L > K) {
//         if (verbose)
//            printf("L is changed to %d\n",K);
         L=K;
      }
      Matrix<T> X(prX,n,M);
      Matrix<T> D(prD,n,K);
      SpMatrix<T> alpha;

      if (nlhs == 2) {
         Matrix<T> norm(K,length_path);
         norm.setZeros();
         if (cholesky) {
            lasso<T>(X,D,alpha,L,lambda,lambda2,mode,pos,ols,numThreads,&norm,length_path);
         } else {
            lasso2<T>(X,D,alpha,L,lambda,lambda2,mode,pos,numThreads,&norm,length_path);
         }
         Vector<T> norms_col;
         norm.norm_2_cols(norms_col);
         int length=1;
         for (int i = 1; i<norms_col.n(); ++i)
            if (norms_col[i]) ++length;
         plhs[1]=createMatrix<T>(K,length);
         T* pr_norm=reinterpret_cast<T*>(mxGetPr(plhs[1]));
         Matrix<T> norm2(pr_norm,K,length);
         Vector<T> col;
         for (int i = 0; i<length; ++i) {
            norm2.refCol(i,col);
            norm.copyCol(i,col);
         }
      } else {
         if (cholesky) {
            lasso<T>(X,D,alpha,L,lambda,lambda2,mode,pos,ols,numThreads,NULL,length_path);
         } else {
            lasso2<T>(X,D,alpha,L,lambda,lambda2,mode,pos,numThreads,NULL,length_path);
         }
      }
      convertSpMatrix(plhs[0],alpha.m(),alpha.n(),alpha.n(),
            alpha.nzmax(),alpha.v(),alpha.r(),alpha.pB());
   } else {
      if (!mexCheckType<T>(prhs[0])) 
         mexErrMsgTxt("type of argument 1 is not consistent");
      if (mxIsSparse(prhs[0])) 
         mexErrMsgTxt("argument 1 should be full");
      if (!mexCheckType<T>(prhs[1])) 
         mexErrMsgTxt("type of argument 2 is not consistent");
      if (mxIsSparse(prhs[1])) 
         mexErrMsgTxt("argument 2 should be full");
      if (!mexCheckType<T>(prhs[2])) 
         mexErrMsgTxt("type of argument 3 is not consistent");
      if (mxIsSparse(prhs[2])) 
         mexErrMsgTxt("argument 3 should be full");
      if (!mxIsStruct(prhs[3])) 
         mexErrMsgTxt("argument 4 should be struct");

      T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0]));
      const mwSize* dimsX=mxGetDimensions(prhs[0]);
      int n=static_cast<int>(dimsX[0]);
      int M=static_cast<int>(dimsX[1]);

      T* prG = reinterpret_cast<T*>(mxGetPr(prhs[1]));
      const mwSize* dimsD=mxGetDimensions(prhs[1]);
      int K1=static_cast<int>(dimsD[0]);
      int K2=static_cast<int>(dimsD[1]);
      if (K1 != K2) mexErrMsgTxt("argument sizes are not consistent");
      int K=K1;

      T* prDtR = reinterpret_cast<T*>(mxGetPr(prhs[2]));
      const mwSize* dimsDtR=mxGetDimensions(prhs[2]);
      int K3=static_cast<int>(dimsDtR[0]);
      int M2=static_cast<int>(dimsDtR[1]);
      if (K1 != K3) mexErrMsgTxt("argument sizes are not consistent");
      if (M != M2) mexErrMsgTxt("argument sizes are not consistent");

      T lambda = getScalarStruct<T>(prhs[3],"lambda");
      T lambda2 = getScalarStructDef<T>(prhs[3],"lambda2",0);
      int L = getScalarStructDef<int>(prhs[3],"L",K1);
      int length_path = getScalarStructDef<int>(prhs[3],"length_path",4*L);
      int numThreads = getScalarStructDef<int>(prhs[3],"numThreads",-1);
      bool pos = getScalarStructDef<bool>(prhs[3],"pos",false);
      bool verbose = getScalarStructDef<bool>(prhs[3],"verbose",true);
      bool ols = getScalarStructDef<bool>(prhs[3],"ols",false);
      bool cholesky = ols || getScalarStructDef<bool>(prhs[3],"cholesky",false);
      constraint_type mode = (constraint_type)getScalarStructDef<int>(prhs[3],"mode",PENALTY);
      if (L > n && !(mode == PENALTY && isZero(lambda) && !pos && lambda2 > 0)) {
//         if (verbose)
//            printf("L is changed to %d\n",n);
         L=n;
      }
      if (L > K) {
//         if (verbose)
//            printf("L is changed to %d\n",K);
         L=K;
      }
      Matrix<T> X(prX,n,M);
      Matrix<T> G(prG,K,K);
      Matrix<T> DtR(prDtR,K,M);
      SpMatrix<T> alpha;

      if (nlhs == 2) {
         Matrix<T> norm(K,length_path);
         norm.setZeros();
         if (cholesky) {
            lasso<T>(X,G,DtR,alpha,L,lambda,mode,pos,ols,numThreads,&norm,length_path);
         } else {
            lasso2<T>(X,G,DtR,alpha,L,lambda,mode,pos,numThreads,&norm,length_path);
         }
         Vector<T> norms_col;
         norm.norm_2_cols(norms_col);
         int length=1;
         for (int i = 1; i<norms_col.n(); ++i)
            if (norms_col[i]) ++length;
         plhs[1]=createMatrix<T>(K,length);
         T* pr_norm=reinterpret_cast<T*>(mxGetPr(plhs[1]));
         Matrix<T> norm2(pr_norm,K,length);
         Vector<T> col;
         for (int i = 0; i<length; ++i) {
            norm2.refCol(i,col);
            norm.copyCol(i,col);
         }
      } else {
         if (cholesky) {
            lasso<T>(X,G,DtR,alpha,L,lambda,mode,pos,ols,numThreads,NULL,length_path);
         } else {
            lasso2<T>(X,G,DtR,alpha,L,lambda,mode,pos,numThreads,NULL,length_path);
         }
      }
      convertSpMatrix(plhs[0],alpha.m(),alpha.n(),alpha.n(),
            alpha.nzmax(),alpha.v(),alpha.r(),alpha.pB());
   }
}

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs != 3 && nrhs != 4)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (!(nlhs == 1 || nlhs == 2))
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nrhs,nlhs);
      } else {
         callFunction<float>(plhs,prhs,nrhs,nlhs);
      }
   }



