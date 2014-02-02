
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
 *                File mexConjGrad.h
 * \brief mex-file, function mexConjGrad.h 
 * Usage: x = mexConjGrad(A,b,x0,tol,itermax); */


#include <mex.h>
#include <mexutils.h>

template <typename T>
   inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
         const int nrhs) {
      if (!mexCheckType<T>(prhs[0])) 
         mexErrMsgTxt("type of argument 1 is not consistent");
      if (mxIsSparse(prhs[0])) 
         mexErrMsgTxt("argument 1 should be full");
      if (!mexCheckType<T>(prhs[1])) 
         mexErrMsgTxt("type of argument 2 is not consistent");
      if (mxIsSparse(prhs[1])) 
         mexErrMsgTxt("argument 2 should be full");
      if (nrhs >= 3) {
         if (!mexCheckType<T>(prhs[2])) 
            mexErrMsgTxt("type of argument 3 is not consistent");
         if (mxIsSparse(prhs[2])) 
            mexErrMsgTxt("argument 3 should be full");
      }

      T* prA = reinterpret_cast<T*>(mxGetPr(prhs[0]));
      const mwSize* dimsX=mxGetDimensions(prhs[0]);
      int m=static_cast<int>(dimsX[0]);
      int n=static_cast<int>(dimsX[1]);
      T* prb = reinterpret_cast<T*>(mxGetPr(prhs[1]));
      const mwSize* dimsY=mxGetDimensions(prhs[1]);
      int nb=static_cast<int>(dimsY[0]*dimsY[1]);
      if (nb != m)
         mexErrMsgTxt("argument sizes are not consistent");

      plhs[0]=createMatrix<T>(n,1);
      T* prx=reinterpret_cast<T*>(mxGetPr(plhs[0]));
      Vector<T> x(prx,n);
      x.setZeros();

      if (nrhs >= 3) {
         T* prx0 = reinterpret_cast<T*>(mxGetPr(prhs[2]));
         const mwSize* dimsx0=mxGetDimensions(prhs[2]);
         int nx=static_cast<int>(dimsx0[0]*dimsx0[1]);
         if (nx != n)
            mexErrMsgTxt("argument sizes are not consistent");
         Vector<T> x0(prx0,nx);
         x.copy(x0);
      }

      T tol=1e-10;
      if (nrhs >= 4) {
         T* prtol = reinterpret_cast<T*>(mxGetPr(prhs[3]));
         tol=*prtol;
      }
      int itermax=n;
      if (nrhs >= 5) {
         T* pritermax = reinterpret_cast<T*>(mxGetPr(prhs[4]));
         itermax=static_cast<int>(*pritermax);
      }
      
      Matrix<T> A(prA,m,n);
      Vector<T> b(prb,nb);
      A.conjugateGradient(b,x,tol,itermax);
   }

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs < 2 || nrhs > 5)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (nlhs != 1) 
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nrhs);
      } else {
         callFunction<float>(plhs,prhs,nrhs);
      }
   }




