/*!
 * \file
 *                toolbox Linalg 
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File mexConjGrad.h
 * \brief mex-file, function mexConjGrad.h 
 * Usage: x = mexRidgeRegression(b,A,x0,param); 
 * Usage: x = mexRidgeRegression(b,delta,A,x0,param); */


#include <mex.h>
#include <mexutils.h>

template <typename T>
   inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
         const int nrhs) {
      if (!mexCheckType<T>(prhs[0])) 
         mexErrMsgTxt("type of argument 1 is not consistent");
      if (mxIsSparse(prhs[0])) 
         mexErrMsgTxt("argument 1 should not be sparse");
      if (!mexCheckType<T>(prhs[1])) 
         mexErrMsgTxt("type of argument 2 is not consistent");
      if (!mexCheckType<T>(prhs[2])) 
         mexErrMsgTxt("type of argument 3 is not consistent");
      const mxArray* arrayb=prhs[0];
      const mxArray *arrayA=nrhs==5 ? prhs[2] : prhs[1];
      const mxArray* arrayx0=nrhs==5 ? prhs[3] : prhs[2];
      const mxArray* arrayparam=nrhs==5 ? prhs[4] : prhs[3];
      const mxArray* arraydelta=nrhs==5 ? prhs[1] : NULL;
      if (nrhs==5) {
         if (!mexCheckType<T>(prhs[3])) 
            mexErrMsgTxt("type of argument 4 is not consistent");
         if (mxIsSparse(prhs[3])) 
            mexErrMsgTxt("argument 4 should not be sparse");
         if (mxIsSparse(prhs[1])) 
            mexErrMsgTxt("argument 1 should not be sparse");
      } else {
         if (mxIsSparse(prhs[2])) 
            mexErrMsgTxt("argument 3 should not be sparse");
      }

      T* prb = reinterpret_cast<T*>(mxGetPr(arrayb));
      const mwSize* dimsb=mxGetDimensions(arrayb);
      INTM nb=static_cast<INTM>(dimsb[0]);
      INTM mb=static_cast<INTM>(dimsb[1]);
      Matrix<T> b(prb,nb,mb);

      const mwSize* dimsA=mxGetDimensions(arrayA);
      INTM m=static_cast<INTM>(dimsA[0]);
      INTM n=static_cast<INTM>(dimsA[1]);
      AbstractMatrixB<T>* A;
      AbstractMatrixB<T>* A2 = NULL;
      AbstractMatrixB<T>* A3 = NULL;


      double* A_v;
      mwSize* A_r, *A_pB, *A_pE;
      INTM* A_r2, *A_pB2, *A_pE2;
      T* A_v2;

      const int shifts = getScalarStructDef<int>(arrayparam,"shifts",1);
      const T lambda = getScalarStructDef<T>(arrayparam,"lambda",0);
      const T tol = getScalarStructDef<T>(arrayparam,"tol",0.0000001);
      const int itermax = getScalarStructDef<int>(arrayparam,"itermax",MAX(m,n));
      const int numThreads= getScalarStructDef<int>(arrayparam,"numThreads",-1);

      if (mxIsSparse(arrayA)) {
         A_v=static_cast<double*>(mxGetPr(arrayA));
         A_r=mxGetIr(arrayA);
         A_pB=mxGetJc(arrayA);
         A_pE=A_pB+1;
         createCopySparse<T>(A_v2,A_r2,A_pB2,A_pE2,
               A_v,A_r,A_pB,A_pE,n);
         A = new SpMatrix<T>(A_v2,A_r2,A_pB2,A_pE2,m,n,A_pB2[n]);
      } else {
         T* prA = reinterpret_cast<T*>(mxGetPr(arrayA));
         A = new Matrix<T>(prA,m,n);
      }

      const bool double_rows = getScalarStructDef<bool>(arrayparam,"double_rows",false);
      if (double_rows) {
         A2=A;
         A=new DoubleRowMatrix<T>(*A);
      }

      if (shifts > 1) {
         const bool center_shifts = getScalarStructDef<bool>(arrayparam,"center_shifts",false);
         A3=A;
         A=new ShiftMatrix<T>(*A,shifts,center_shifts);
      }

      T* pr_x = reinterpret_cast<T*>(mxGetPr(arrayx0));
      const mwSize* dimsx=mxGetDimensions(arrayx0);
      INTM nx=static_cast<INTM>(dimsx[0]);
      INTM mx=static_cast<INTM>(dimsx[1]);
      Matrix<T> x(pr_x,nx,mx);

      plhs[0]=createMatrix<T>(nx,mx);
      T* pr_xout=reinterpret_cast<T*>(mxGetPr(plhs[0]));
      Matrix<T> xout(pr_xout,nx,mx);
      xout.copy(x);

      if (nrhs==4) {
         A->ridgeCG(b,xout,lambda,tol,itermax,numThreads);
      } else {
         T* pr_d = reinterpret_cast<T*>(mxGetPr(arraydelta));
         const mwSize* dimsd=mxGetDimensions(arraydelta);
         INTM nd=static_cast<INTM>(dimsd[0]);
         INTM md=static_cast<INTM>(dimsd[1]);
         Matrix<T> delta(pr_d,nd,md);
         A->ridgeCG(b,delta,xout,lambda,tol,itermax,numThreads);
      }

      delete(A);
      delete(A2);
      delete(A3);
   }

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
   if (nrhs != 4 && nrhs != 5)
      mexErrMsgTxt("Bad number of inputs arguments");

   if (nlhs != 1) 
      mexErrMsgTxt("Bad number of output arguments");

   if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
      callFunction<double>(plhs,prhs,nrhs);
   } else {
      callFunction<float>(plhs,prhs,nrhs);
   }
}




