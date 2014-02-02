
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
 *                toolbox dictLearn
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File mexTrainDL.h
 * \brief mex-file, function mexTrainDL
 * Usage: [D] = mexTrainDLOffline(X,param);
 * */



#include <mexutils.h>
#include <dicts.h>

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
      int n=static_cast<int>(dimsX[0]);
      int M=static_cast<int>(dimsX[1]);
      Matrix<T> X(prX,n,M);

      int NUM_THREADS = getScalarStructDef<int>(prhs[1],"numThreads",-1);
#ifdef _OPENMP
      NUM_THREADS = NUM_THREADS == -1 ? omp_get_num_procs() : NUM_THREADS;
#else
      NUM_THREADS=1;
#endif 
      int batch_size = getScalarStructDef<int>(prhs[1],"batchsize",
            256*(NUM_THREADS+1));
      mxArray* pr_D = mxGetField(prhs[1],0,"D");
      Trainer<T>* trainer;

      if (!pr_D) {
         int K = getScalarStruct<int>(prhs[1],"K");
         trainer = new Trainer<T>(K,batch_size,NUM_THREADS);
      } else {
         T* prD = reinterpret_cast<T*>(mxGetPr(pr_D));
         const mwSize* dimsD=mxGetDimensions(pr_D);
         int nD=static_cast<int>(dimsD[0]);
         int K=static_cast<int>(dimsD[1]);
         if (n != nD) mexErrMsgTxt("sizes of D are not consistent");
         Matrix<T> D1(prD,n,K);
         trainer = new Trainer<T>(D1,batch_size,NUM_THREADS);
      }

      ParamDictLearn<T> param;
      param.lambda = getScalarStruct<T>(prhs[1],"lambda");
      param.iter=getScalarStruct<int>(prhs[1],"iter");
      param.mode = (constraint_type)getScalarStructDef<int>(prhs[1],"mode",PENALTY);
      if (param.mode != PENALTY && param.mode != L2ERROR) 
         mexErrMsgTxt("param.mode is not compatible with the offline setting");
      param.posD = getScalarStructDef<bool>(prhs[1],"posD",false);
      param.modeD=(constraint_type_D)(getScalarStructDef<int>(prhs[1],"modeD",0));
      param.whiten = getScalarStructDef<bool>(prhs[1],"whiten",false);
      param.modeParam = static_cast<mode_compute>(getScalarStructDef<int>(prhs[1],"modeParam",0));
      param.clean = getScalarStructDef<bool>(prhs[1],"clean",true);
      param.gamma1 = getScalarStructDef<T>(prhs[1],"gamma1",0);
      param.gamma2 = getScalarStructDef<T>(prhs[1],"gamma2",0);
      param.rho = getScalarStructDef<T>(prhs[1],"rho",T(1.0));
      param.iter_updateD = getScalarStructDef<int>(prhs[1],"iter_udpateD",1);
      trainer->trainOffline(X,param);
      Matrix<T> D;
      trainer->getD(D);
      int K  = D.n();
      plhs[0] = createMatrix<T>(n,K);
      T* prD2 = reinterpret_cast<T*>(mxGetPr(plhs[0]));
      Matrix<T> D2(prD2,n,K);
      D2.copy(D);
      delete(trainer);
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




