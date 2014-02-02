
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
 *                File mexSOMP.h
 * \brief mex-file, function mexSOMP
 * Usage: alpha = mexSOMP(X,D,list_groups,param);
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
   if (mxIsSparse(prhs[2])) mexErrMsgTxt("argument 3 should be full");
   if (!mexCheckType<int>(prhs[2])) 
      mexErrMsgTxt("type of argument 3 is not consistent");

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

   const mwSize* dimsList = mxGetDimensions(prhs[2]);
   int Ng = static_cast<int>(dimsList[0]*dimsList[1]);
   int* list_groups=reinterpret_cast<int*>(mxGetPr(prhs[2]));

   int L= getScalarStruct<int>(prhs[3],"L");
   T eps= getScalarStructDef<T>(prhs[3],"eps",0);
   int numThreads = getScalarStructDef<int>(prhs[3],"numThreads",-1);

   Matrix<T> D(prD,n,K);
   Matrix<T>* X = new Matrix<T>[Ng];
   if (list_groups[0] != 0)
      mexErrMsgTxt("First group index should be zero");
   for (int i = 0; i<Ng-1; ++i) {
      if (list_groups[i] >= M) 
         mexErrMsgTxt("Size of groups is not consistent");
      if (list_groups[i] >= list_groups[i+1]) 
         mexErrMsgTxt("Group indices should be a strictly non-decreasing sequence");
      X[i].setData(prX+list_groups[i]*n,n,list_groups[i+1]-list_groups[i]);
   }
   X[Ng-1].setData(prX+list_groups[Ng-1]*n,n,M-list_groups[Ng-1]);
   SpMatrix<T>* spAlpha = new SpMatrix<T>[Ng];

   somp(X,D,spAlpha,Ng,L,eps,numThreads);

   INTM nzmax=0;
   for (INTM i = 0; i<Ng; ++i) {
      nzmax += spAlpha[i].nzmax();
   }
   plhs[0]=mxCreateSparse(K,M,nzmax,mxREAL);
   double* Pr = mxGetPr(plhs[0]);
   mwSize* Ir = mxGetIr(plhs[0]);
   mwSize* Jc = mxGetJc(plhs[0]);
   INTM count=0;
   INTM countcol=0;
   INTM offset=0;
   for (INTM i = 0; i<Ng; ++i) {
      const T* v = spAlpha[i].v();
      const INTM* r = spAlpha[i].r();
      const INTM* pB = spAlpha[i].pB();
      INTM nn = spAlpha[i].n();
      nzmax = spAlpha[i].nzmax();
      if (nn != 0) {
         for (INTM j = 0; j<pB[nn]; ++j) {
            Pr[count]=static_cast<double>(v[j]);
            Ir[count++]=static_cast<mwSize>(r[j]);
         }
         for (INTM j = 0; j<=nn; ++j) 
            Jc[countcol++]=static_cast<mwSize>(offset+pB[j]);
         --countcol;
         offset = Jc[countcol];
      }
   }

   delete[](X);
   delete[](spAlpha);
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

