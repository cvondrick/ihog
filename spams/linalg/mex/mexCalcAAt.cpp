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
 *                File mexCalcAAt.h
 * \brief mex-file, function mexCalcAAt.h 
 * Usage: AAt = mexCalcAAt(A); */


#include <mex.h>
#include <mexutils.h>

template <typename T>
   inline void callFunction(mxArray* plhs[], const mxArray*prhs[]) {
      if (!mexCheckType<T>(prhs[0])) 
         mexErrMsgTxt("type of argument 1 is not consistent");
      if (!mxIsSparse(prhs[0])) 
         mexErrMsgTxt("argument 1 should be sparse");

      double* alpha_v = reinterpret_cast<double*>(mxGetPr(prhs[0]));
      const mwSize* dims=mxGetDimensions(prhs[0]);
      INTM K=static_cast<INTM>(dims[0]);
      INTM M=static_cast<INTM>(dims[1]);
      mwSize* alpha_r=mxGetIr(prhs[0]);
      mwSize* alpha_pB=mxGetJc(prhs[0]);
      mwSize* alpha_pE=alpha_pB+1;

      INTM* alpha_r2, *alpha_pB2, *alpha_pE2;
      T* alpha_v2;
      createCopySparse<T>(alpha_v2,alpha_r2,alpha_pB2,alpha_pE2,
            alpha_v,alpha_r,alpha_pB,alpha_pE,M);

      plhs[0]=createMatrix<T>(K,K);
      T* prAAt=reinterpret_cast<T*>(mxGetPr(plhs[0]));

      Matrix<T> aat(prAAt,K,K);
      SpMatrix<T> alpha(alpha_v2,alpha_r2,alpha_pB2,
            alpha_pE2,K,M,alpha_pB2[M]);

      alpha.AAt(aat);

      deleteCopySparse<T>(alpha_v2,alpha_r2,alpha_pB2,alpha_pE2,
            alpha_v,alpha_r);
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
   if (nrhs != 1)
      mexErrMsgTxt("Bad number of inputs arguments");

   if (nlhs != 1) 
      mexErrMsgTxt("Bad number of output arguments");

   if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
      callFunction<double>(plhs,prhs);
   } else {
      callFunction<float>(plhs,prhs);
   }
}




