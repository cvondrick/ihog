
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
 *                toolbox Linalg 
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File mexL1Project.cpp
 * \brief mex-file, function mexL1Project
 * Usage: Y = mexSparseProject(X,param); */

#include <mex.h>
#include <mexutils.h>

template <typename T>
   inline void callFunction(mxArray* plhs[], const mxArray*prhs[]) {
      if (!mexCheckType<T>(prhs[0])) 
         mexErrMsgTxt("type of argument 1 is not consistent");
      if (mxIsSparse(prhs[0])) 
         mexErrMsgTxt("argument 1 should be full");
      if (!mxIsStruct(prhs[1])) 
         mexErrMsgTxt("argument 2 should be struct");

      T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0]));
      const mwSize* dimsX=mxGetDimensions(prhs[0]);
      int m=static_cast<int>(dimsX[0]);
      int n=static_cast<int>(dimsX[1]);

      plhs[0]=createMatrix<T>(m,n);
      T* prY=reinterpret_cast<T*>(mxGetPr(plhs[0]));

      Matrix<T> X(prX,m,n);
      Matrix<T> Y(prY,m,n);

      T thrs = getScalarStructDef<T>(prhs[1],"thrs",T(1.0));
      T lambda1 = getScalarStructDef<T>(prhs[1],"lambda1",0);
      T lambda2 = getScalarStructDef<T>(prhs[1],"lambda2",0);
      T lambda3 = getScalarStructDef<T>(prhs[1],"lambda3",0);
      bool pos = getScalarStructDef<bool>(prhs[1],"pos",false);
      int mode = getScalarStructDef<int>(prhs[1],"mode",1);
      int numThreads = getScalarStructDef<int>(prhs[1],"numThreads",-1);
      if (pos && mode >= 5) 
         mexErrMsgTxt("mode >= 5 is not compatible with positivity constraints");
      X.sparseProject(Y,thrs,mode,lambda1,lambda2,lambda3,pos,numThreads);
   }

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
   if (nrhs != 2)
      mexErrMsgTxt("Bad number of inputs arguments");

   if (nlhs != 1) 
      mexErrMsgTxt("Bad number of output arguments");

   if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
      callFunction<double>(plhs,prhs);
   } else {
      callFunction<float>(plhs,prhs);
   }
}




