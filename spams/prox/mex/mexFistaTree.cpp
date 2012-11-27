
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

#include <mex.h>
#include <mexutils.h>
#include <fista.h>

// alpha = mexFistaLasso(X,D,alpha0,tree,param)

using namespace FISTA;

template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
      const long nlhs) {
   if (!mexCheckType<T>(prhs[0])) 
      mexErrMsgTxt("type of argument 1 is not consistent");
   if (mxIsSparse(prhs[0])) 
      mexErrMsgTxt("argument 1 should not be sparse");

   if (!mexCheckType<T>(prhs[1])) 
      mexErrMsgTxt("type of argument 2 is not consistent");

   if (!mexCheckType<T>(prhs[2])) 
      mexErrMsgTxt("type of argument 3 is not consistent");
   if (mxIsSparse(prhs[2])) 
      mexErrMsgTxt("argument 3 should not be sparse");

   if (!mxIsStruct(prhs[3])) 
         mexErrMsgTxt("argument 4 should be struct");
   if (!mxIsStruct(prhs[4])) 
      mexErrMsgTxt("argument 5 should be struct");

   T* prX = reinterpret_cast<T*>(mxGetPr(prhs[0]));
   const mwSize* dimsX=mxGetDimensions(prhs[0]);
   long m=static_cast<long>(dimsX[0]);
   long n=static_cast<long>(dimsX[1]);
   Matrix<T> X(prX,m,n);

   const mwSize* dimsD=mxGetDimensions(prhs[1]);
   long mD=static_cast<long>(dimsD[0]);
   long p=static_cast<long>(dimsD[1]);
   AbstractMatrixB<T>* D;

   double* D_v;
   mwSize* D_r, *D_pB, *D_pE;
   long* D_r2, *D_pB2, *D_pE2;
   T* D_v2;
   if (mxIsSparse(prhs[1])) {
      D_v=static_cast<double*>(mxGetPr(prhs[1]));
      D_r=mxGetIr(prhs[1]);
      D_pB=mxGetJc(prhs[1]);
      D_pE=D_pB+1;
      createCopySparse<T>(D_v2,D_r2,D_pB2,D_pE2,
            D_v,D_r,D_pB,D_pE,p);
      D = new SpMatrix<T>(D_v2,D_r2,D_pB2,D_pE2,mD,p,D_pB2[p]);
   } else {
      T* prD = reinterpret_cast<T*>(mxGetPr(prhs[1]));
      D = new Matrix<T>(prD,m,p);
   }

   T* pr_alpha0 = reinterpret_cast<T*>(mxGetPr(prhs[2]));
   const mwSize* dimsAlpha=mxGetDimensions(prhs[2]);
   long pAlpha=static_cast<long>(dimsAlpha[0]);
   long nAlpha=static_cast<long>(dimsAlpha[1]);
   Matrix<T> alpha0(pr_alpha0,pAlpha,nAlpha);

   mxArray* ppr_own_variables = mxGetField(prhs[3],0,"own_variables");
   if (!mexCheckType<long>(ppr_own_variables)) 
         mexErrMsgTxt("own_variables field should be int32");
   if (!ppr_own_variables) mexErrMsgTxt("field own_variables is not provided");
   long* pr_own_variables = reinterpret_cast<long*>(mxGetPr(ppr_own_variables));
   const mwSize* dims_groups =mxGetDimensions(ppr_own_variables);
   long num_groups=static_cast<long>(dims_groups[0])*static_cast<long>(dims_groups[1]);

   mxArray* ppr_N_own_variables = mxGetField(prhs[3],0,"N_own_variables");
   if (!ppr_N_own_variables) mexErrMsgTxt("field N_own_variables is not provided");
   if (!mexCheckType<long>(ppr_N_own_variables)) 
         mexErrMsgTxt("N_own_variables field should be int32");
   const mwSize* dims_var =mxGetDimensions(ppr_N_own_variables);
   long num_groups2=static_cast<long>(dims_var[0])*static_cast<long>(dims_var[1]);
   if (num_groups != num_groups2)
      mexErrMsgTxt("Error in tree definition");
   long* pr_N_own_variables = reinterpret_cast<long*>(mxGetPr(ppr_N_own_variables));
   long num_var=0;
   for (long i = 0; i<num_groups; ++i)
      num_var+=pr_N_own_variables[i];
   if (pAlpha < num_var) 
      mexErrMsgTxt("Input alpha is too small");

   mxArray* ppr_lambda_g = mxGetField(prhs[3],0,"eta_g");
   if (!ppr_lambda_g) mexErrMsgTxt("field eta_g is not provided");
   const mwSize* dims_weights =mxGetDimensions(ppr_lambda_g);
   long num_groups3=static_cast<long>(dims_weights[0])*static_cast<long>(dims_weights[1]);
   if (num_groups != num_groups3)
      mexErrMsgTxt("Error in tree definition");
   T* pr_lambda_g = reinterpret_cast<T*>(mxGetPr(ppr_lambda_g));

   mxArray* ppr_groups = mxGetField(prhs[3],0,"groups");
   const mwSize* dims_gg =mxGetDimensions(ppr_groups);
   if ((num_groups != static_cast<long>(dims_gg[0])) || 
         (num_groups != static_cast<long>(dims_gg[1])))
      mexErrMsgTxt("Error in tree definition");
   if (!ppr_groups) mexErrMsgTxt("field groups is not provided");
   mwSize* pr_groups_ir = reinterpret_cast<mwSize*>(mxGetIr(ppr_groups));
   mwSize* pr_groups_jc = reinterpret_cast<mwSize*>(mxGetJc(ppr_groups));

   plhs[0]=createMatrix<T>(pAlpha,nAlpha);
   T* pr_alpha=reinterpret_cast<T*>(mxGetPr(plhs[0]));
   Matrix<T> alpha(pr_alpha,pAlpha,nAlpha);

   FISTA::ParamFISTA<T> param;
   param.num_threads = getScalarStructDef<long>(prhs[4],"numThreads",-1);
   param.max_it = getScalarStructDef<long>(prhs[4],"max_it",1000);
   param.tol = getScalarStructDef<T>(prhs[4],"tol",0.000001);
   param.it0 = getScalarStructDef<long>(prhs[4],"it0",100);
   param.pos = getScalarStructDef<bool>(prhs[4],"pos",false);
   param.compute_gram = getScalarStructDef<bool>(prhs[4],"compute_gram",false);
   param.max_iter_backtracking = getScalarStructDef<long>(prhs[4],"max_iter_backtracking",1000);
   param.L0 = getScalarStructDef<T>(prhs[4],"L0",1.0);
   param.fixed_step = getScalarStructDef<T>(prhs[4],"fixed_step",false);
   param.gamma = MAX(1.01,getScalarStructDef<T>(prhs[4],"gamma",1.5));
   param.c = getScalarStructDef<T>(prhs[4],"c",1.0);
   param.lambda= getScalarStructDef<T>(prhs[4],"lambda",1.0);
   param.lambda2= getScalarStructDef<T>(prhs[4],"lambda2",0.0);
   param.lambda3= getScalarStructDef<T>(prhs[4],"lambda3",0.0);
   param.size_group= getScalarStructDef<long>(prhs[4],"size_group",1);
   param.delta = getScalarStructDef<T>(prhs[4],"delta",1.0);
   param.admm = getScalarStructDef<bool>(prhs[4],"admm",false);
   param.lin_admm = getScalarStructDef<bool>(prhs[4],"lin_admm",false);
   param.sqrt_step = getScalarStructDef<bool>(prhs[4],"sqrt_step",true);
   getStringStruct(prhs[4],"regul",param.name_regul,param.length_names);
   param.is_inner_weights = getScalarStructDef<bool>(prhs[4],"is_inner_weights",false);
   param.transpose = getScalarStructDef<bool>(prhs[4],"transpose",false);
   if (param.is_inner_weights) {
      mxArray* ppr_inner_weights = mxGetField(prhs[4],0,"inner_weights");
      if (!ppr_inner_weights) mexErrMsgTxt("field inner_weights is not provided");
      if (!mexCheckType<T>(ppr_inner_weights)) 
         mexErrMsgTxt("type of inner_weights is not correct");
      param.inner_weights = reinterpret_cast<T*>(mxGetPr(ppr_inner_weights));
   }

   param.regul = regul_from_string(param.name_regul);
   if (param.regul==INCORRECT_REG)
      mexErrMsgTxt("Unknown regularization");
   getStringStruct(prhs[4],"loss",param.name_loss,param.length_names);
   param.loss = loss_from_string(param.name_loss);
   if (param.loss==INCORRECT_LOSS)
      mexErrMsgTxt("Unknown loss");

   param.intercept = getScalarStructDef<bool>(prhs[4],"intercept",false);
   param.resetflow = getScalarStructDef<bool>(prhs[4],"resetflow",false);
   param.verbose = getScalarStructDef<bool>(prhs[4],"verbose",false);
   param.clever = getScalarStructDef<bool>(prhs[4],"clever",false);
   param.ista= getScalarStructDef<bool>(prhs[4],"ista",false);
   param.subgrad= getScalarStructDef<bool>(prhs[4],"subgrad",false);
   param.log= getScalarStructDef<bool>(prhs[4],"log",false);
   param.a= getScalarStructDef<T>(prhs[4],"a",T(1.0));
   param.b= getScalarStructDef<T>(prhs[4],"b",0);

   if (param.log) {
      mxArray *stringData = mxGetField(prhs[4],0,"logName");
      if (!stringData) 
         mexErrMsgTxt("Missing field logName");
      long stringLength = mxGetN(stringData)+1;
      param.logName= new char[stringLength];
      mxGetString(stringData,param.logName,stringLength);
   }

   if ((param.loss != CUR && param.loss != MULTILOG) && (pAlpha != p || nAlpha != n || mD != m)) { 
      mexErrMsgTxt("Argument sizes are not consistent");
   } else if (param.loss == MULTILOG) {
      Vector<T> Xv;
      X.toVect(Xv);
      long maxval = static_cast<long>(Xv.maxval());
      long minval = static_cast<long>(Xv.minval());
      if (minval != 0)
         mexErrMsgTxt("smallest class should be 0");
      if (maxval*X.n() > nAlpha || mD != m) {
         cerr << "Number of classes: " << maxval << endl;
         //cerr << "Alpha: " << pAlpha << " x " << nAlpha << endl;
         //cerr << "X: " << X.m() << " x " << X.n() << endl;
         mexErrMsgTxt("Argument sizes are not consistent");
      }
   } else if (param.loss == CUR && (pAlpha != D->n() || nAlpha != D->m())) {
      mexErrMsgTxt("Argument sizes are not consistent");
   }

   if (param.regul==GRAPH || param.regul==GRAPHMULT) 
      mexErrMsgTxt("Error: mexFistaGraph should be used instead");

   if (param.regul==TREEMULT && abs<T>(param.lambda2 - 0) < 1e-20) {
      mexErrMsgTxt("Error: with multi-task-tree, lambda2 should be > 0");
   }

   if (param.num_threads == -1) {
      param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   } 

   TreeStruct<T> tree;
   tree.Nv=0;
   for (long i = 0; i<num_groups; ++i) tree.Nv+=pr_N_own_variables[i];
   tree.Ng=num_groups;
   tree.weights=pr_lambda_g;
   tree.own_variables=pr_own_variables;
   tree.N_own_variables=pr_N_own_variables;
   tree.groups_ir=pr_groups_ir;
   tree.groups_jc=pr_groups_jc;

   Matrix<T> duality_gap;
   FISTA::solver<T>(X,*D,alpha0,alpha,param,duality_gap,NULL,&tree);
   if (nlhs==2) {
      plhs[1]=createMatrix<T>(duality_gap.m(),duality_gap.n());
      T* pr_dualitygap=reinterpret_cast<T*>(mxGetPr(plhs[1]));
      for (long i = 0; i<duality_gap.n()*duality_gap.m(); ++i) pr_dualitygap[i]=duality_gap[i];
   }
   if (param.logName) delete[](param.logName);

   if (mxIsSparse(prhs[1])) {
      deleteCopySparse<T>(D_v2,D_r2,D_pB2,D_pE2,
            D_v,D_r);
   }
   delete(D);
}

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs != 5)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (nlhs != 1 && nlhs !=2) 
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nlhs);
      } else {
         callFunction<float>(plhs,prhs,nlhs);
      }
   }




