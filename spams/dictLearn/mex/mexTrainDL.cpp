
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
 * Usage: [D model] = mexTrainDL(X,param);
 * Usage: [D model] = mexTrainDL(X,param,model);
 * output model is optional
 * */



#include <mexutils.h>
#include <dicts.h>

template <typename T>
   inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
         const int nlhs,const int nrhs) {
      if (!mexCheckType<T>(prhs[0])) 
         mexErrMsgTxt("type of argument 1 is not consistent");

      if (!mxIsStruct(prhs[1])) 
         mexErrMsgTxt("argument 2 should be struct");

      if (nrhs == 3)
         if (!mxIsStruct(prhs[2])) 
            mexErrMsgTxt("argument 3 should be struct");

      Data<T> *X;
      const mwSize* dimsX=mxGetDimensions(prhs[0]);
      INTM n=static_cast<INTM>(dimsX[0]);
      INTM M=static_cast<int>(dimsX[1]);
      if (mxIsSparse(prhs[0])) {
         double * X_v=static_cast<double*>(mxGetPr(prhs[0]));
         mwSize* X_r=mxGetIr(prhs[0]);
         mwSize* X_pB=mxGetJc(prhs[0]);
         mwSize* X_pE=X_pB+1;
         INTM* X_r2, *X_pB2, *X_pE2;
         T* X_v2;
         createCopySparse<T>(X_v2,X_r2,X_pB2,X_pE2,
               X_v,X_r,X_pB,X_pE,M);
         X = new SpMatrix<T>(X_v2,X_r2,X_pB2,X_pE2,n,M,X_pB2[M]);
      } else {
         T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0]));
         X= new Matrix<T>(prX,n,M);
      }

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
         if (nrhs == 3) {
            mxArray* pr_A = mxGetField(prhs[2],0,"A");
            if (!pr_A) mexErrMsgTxt("field A is not provided");
            T* prA = reinterpret_cast<T*>(mxGetPr(pr_A));
            const mwSize* dimsA=mxGetDimensions(pr_A);
            int xA=static_cast<int>(dimsA[0]);
            int yA=static_cast<int>(dimsA[1]);
            if (xA != K || yA != K) mexErrMsgTxt("Size of A is not consistent");
            Matrix<T> A(prA,K,K);

            mxArray* pr_B = mxGetField(prhs[2],0,"B");
            if (!pr_B) mexErrMsgTxt("field B is not provided");
            T* prB = reinterpret_cast<T*>(mxGetPr(pr_B));
            const mwSize* dimsB=mxGetDimensions(pr_B);
            int xB=static_cast<int>(dimsB[0]);
            int yB=static_cast<int>(dimsB[1]);
            if (xB != n || yB != K) mexErrMsgTxt("Size of B is not consistent");
            Matrix<T> B(prB,n,K);
            int iter = getScalarStruct<int>(prhs[2],"iter");
            trainer = new Trainer<T>(A,B,D1,iter,batch_size,NUM_THREADS);
         } else {
            trainer = new Trainer<T>(D1,batch_size,NUM_THREADS);
         }
      }

      ParamDictLearn<T> param;
      param.lambda = getScalarStruct<T>(prhs[1],"lambda");
      param.lambda2 = getScalarStructDef<T>(prhs[1],"lambda2",10e-10);
      param.iter=getScalarStruct<int>(prhs[1],"iter");
      param.t0 = getScalarStructDef<T>(prhs[1],"t0",1e-5);
      param.mode =(constraint_type)getScalarStructDef<int>(prhs[1],"mode",PENALTY);
      param.posAlpha = getScalarStructDef<bool>(prhs[1],"posAlpha",false);
      param.posD = getScalarStructDef<bool>(prhs[1],"posD",false);
      param.expand= getScalarStructDef<bool>(prhs[1],"expand",false);
      param.modeD=(constraint_type_D)getScalarStructDef<int>(prhs[1],"modeD",L2);
      param.whiten = getScalarStructDef<bool>(prhs[1],"whiten",false);
      param.clean = getScalarStructDef<bool>(prhs[1],"clean",true);
      param.verbose = getScalarStructDef<bool>(prhs[1],"verbose",true);
      param.gamma1 = getScalarStructDef<T>(prhs[1],"gamma1",0);
      param.gamma2 = getScalarStructDef<T>(prhs[1],"gamma2",0);
      param.rho = getScalarStructDef<T>(prhs[1],"rho",T(1.0));
      param.stochastic = 
         getScalarStructDef<bool>(prhs[1],"stochastic_deprecated",
               false);
      param.modeParam = static_cast<mode_compute>(getScalarStructDef<int>(prhs[1],"modeParam",0));
      param.batch = getScalarStructDef<bool>(prhs[1],"batch",false);
      param.iter_updateD = getScalarStructDef<T>(prhs[1],"iter_updateD",param.batch ? 5 : 1);
      param.log = getScalarStructDef<bool>(prhs[1],"log_deprecated",
            false);
      if (param.log) {
         mxArray *stringData = mxGetField(prhs[1],0,
               "logName_deprecated");
         if (!stringData) 
            mexErrMsgTxt("Missing field logName_deprecated");
         int stringLength = mxGetN(stringData)+1;
         param.logName= new char[stringLength];
         mxGetString(stringData,param.logName,stringLength);
      }

      trainer->train(*X,param);
      if (param.log)
         mxFree(param.logName);

      Matrix<T> D;
      trainer->getD(D);
      int K  = D.n();
      plhs[0] = createMatrix<T>(n,K);
      T* prD2 = reinterpret_cast<T*>(mxGetPr(plhs[0]));
      Matrix<T> D2(prD2,n,K);
      D2.copy(D);

      if (nlhs == 2) {
         mwSize dims[1] = {1};
         int nfields=3; 
         const char *names[] = {"A", "B", "iter"};
         plhs[1]=mxCreateStructArray(1, dims,nfields, names);
         mxArray* prA = createMatrix<T>(K,K);
         T* pr_A= reinterpret_cast<T*>(mxGetPr(prA));
         Matrix<T> A(pr_A,K,K);
         trainer->getA(A);
         mxSetField(plhs[1],0,"A",prA);
         mxArray* prB = createMatrix<T>(n,K);
         T* pr_B= reinterpret_cast<T*>(mxGetPr(prB));
         Matrix<T> B(pr_B,n,K);
         trainer->getB(B);
         mxSetField(plhs[1],0,"B",prB);
         mxArray* priter = createScalar<T>();
         *mxGetPr(priter) = static_cast<T>(trainer->getIter());
         mxSetField(plhs[1],0,"iter",priter);
      }
      delete(trainer);
      delete(X);
   }

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs < 2 || nrhs > 3)
         mexErrMsgTxt("Bad number of inputs arguments");

      if ((nlhs < 1) || (nlhs > 2))
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nlhs,nrhs);
      } else {
         callFunction<float>(plhs,prhs,nlhs,nrhs);
      }
   }




