/*!
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
 *                toolbox DAGs
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *  */

#include <mex.h>
#include <mexutils.h>
#include <dag.h>


template <typename T>
   inline void callFunction(mxArray* plhs[], const mxArray*prhs[]) {
      if (!mexCheckType<T>(prhs[0])) 
         mexErrMsgTxt("type of argument 1 is not consistent");
      if (!mxIsSparse(prhs[0])) 
         mexErrMsgTxt("argument 1 should be sparse");

      const mwSize* dimsD=mxGetDimensions(prhs[0]);
      long mD=static_cast<long>(dimsD[0]);
      long p=static_cast<long>(dimsD[1]);
      const long n = p;
      double* D_v;
      mwSize* D_r, *D_pB, *D_pE;
      long* D_r2, *D_pB2, *D_pE2;
      T* D_v2;
      D_v=static_cast<double*>(mxGetPr(prhs[0]));
      D_r=mxGetIr(prhs[0]);
      D_pB=mxGetJc(prhs[0]);
      D_pE=D_pB+1;
      createCopySparse<T>(D_v2,D_r2,D_pB2,D_pE2,
            D_v,D_r,D_pB,D_pE,p);
      SpMatrix<T> G(D_v2,D_r2,D_pB2,D_pE2,mD,p,D_pB2[p]);

      SpMatrix<T> G2;
      remove_cycles(G,G2);

      deleteCopySparse<T>(D_v2,D_r2,D_pB2,D_pE2,
            D_v,D_r);
      convertSpMatrix(plhs[0],G2.m(),G2.n(),G2.n(),
            G2.nzmax(),G2.v(),G2.r(),G2.pB());
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




