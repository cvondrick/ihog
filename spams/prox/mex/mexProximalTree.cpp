
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
      const int nlhs) {
   if (!mexCheckType<T>(prhs[0])) 
      mexErrMsgTxt("type of argument 1 is not consistent");
   if (mxIsSparse(prhs[0])) 
      mexErrMsgTxt("argument 1 should not be sparse");

   if (!mxIsStruct(prhs[1])) 
         mexErrMsgTxt("argument 2 should be struct");
   if (!mxIsStruct(prhs[2])) 
      mexErrMsgTxt("argument 3 should be struct");

   T* pr_alpha0 = reinterpret_cast<T*>(mxGetPr(prhs[0]));
   const mwSize* dimsAlpha=mxGetDimensions(prhs[0]);
   int pAlpha=static_cast<int>(dimsAlpha[0]);
   int nAlpha=static_cast<int>(dimsAlpha[1]);
   Matrix<T> alpha0(pr_alpha0,pAlpha,nAlpha);

   mxArray* ppr_own_variables = mxGetField(prhs[1],0,"own_variables");
   if (!mexCheckType<int>(ppr_own_variables)) 
         mexErrMsgTxt("own_variables field should be int32");
   if (!ppr_own_variables) mexErrMsgTxt("field own_variables is not provided");
   int* pr_own_variables = reinterpret_cast<int*>(mxGetPr(ppr_own_variables));
   const mwSize* dims_groups =mxGetDimensions(ppr_own_variables);
   int num_groups=static_cast<int>(dims_groups[0])*static_cast<int>(dims_groups[1]);

   mxArray* ppr_N_own_variables = mxGetField(prhs[1],0,"N_own_variables");
   if (!ppr_N_own_variables) mexErrMsgTxt("field N_own_variables is not provided");
   if (!mexCheckType<int>(ppr_N_own_variables)) 
         mexErrMsgTxt("N_own_variables field should be int32");
   const mwSize* dims_var =mxGetDimensions(ppr_N_own_variables);
   int num_groups2=static_cast<int>(dims_var[0])*static_cast<int>(dims_var[1]);
   if (num_groups != num_groups2)
      mexErrMsgTxt("Error in tree definition");
   int* pr_N_own_variables = reinterpret_cast<int*>(mxGetPr(ppr_N_own_variables));
   int num_var=0;
   for (int i = 0; i<num_groups; ++i)
      num_var+=pr_N_own_variables[i];
   if (pAlpha < num_var) 
      mexErrMsgTxt("Input alpha is too small");

   mxArray* ppr_lambda_g = mxGetField(prhs[1],0,"eta_g");
   if (!ppr_lambda_g) mexErrMsgTxt("field eta_g is not provided");
   const mwSize* dims_weights =mxGetDimensions(ppr_lambda_g);
   int num_groups3=static_cast<int>(dims_weights[0])*static_cast<int>(dims_weights[1]);
   if (num_groups != num_groups3)
      mexErrMsgTxt("Error in tree definition");
   T* pr_lambda_g = reinterpret_cast<T*>(mxGetPr(ppr_lambda_g));

   mxArray* ppr_groups = mxGetField(prhs[1],0,"groups");
   const mwSize* dims_gg =mxGetDimensions(ppr_groups);
   if ((num_groups != static_cast<int>(dims_gg[0])) || 
         (num_groups != static_cast<int>(dims_gg[1])))
      mexErrMsgTxt("Error in tree definition");
   if (!ppr_groups) mexErrMsgTxt("field groups is not provided");
   mwSize* pr_groups_ir = reinterpret_cast<mwSize*>(mxGetIr(ppr_groups));
   mwSize* pr_groups_jc = reinterpret_cast<mwSize*>(mxGetJc(ppr_groups));

   plhs[0]=createMatrix<T>(pAlpha,nAlpha);
   T* pr_alpha=reinterpret_cast<T*>(mxGetPr(plhs[0]));
   Matrix<T> alpha(pr_alpha,pAlpha,nAlpha);

   FISTA::ParamFISTA<T> param;
   param.num_threads = getScalarStructDef<int>(prhs[2],"numThreads",-1);
   param.pos = getScalarStructDef<bool>(prhs[2],"pos",false);
   param.lambda= getScalarStructDef<T>(prhs[2],"lambda",1.0);
   param.lambda2= getScalarStructDef<T>(prhs[2],"lambda2",0.0);
   param.lambda3= getScalarStructDef<T>(prhs[2],"lambda3",0.0);
   param.size_group= getScalarStructDef<int>(prhs[2],"size_group",1);
   param.intercept = getScalarStructDef<bool>(prhs[2],"intercept",false);
   param.resetflow = getScalarStructDef<bool>(prhs[2],"resetflow",false);
   param.verbose = getScalarStructDef<bool>(prhs[2],"verbose",false);
   param.transpose = getScalarStructDef<bool>(prhs[2],"transpose",false);
   getStringStruct(prhs[2],"regul",param.name_regul,param.length_names);
   param.regul = regul_from_string(param.name_regul);
   param.eval = nlhs==2;
   if (param.regul==INCORRECT_REG)
      mexErrMsgTxt("Unknown regularization");

   if (param.num_threads == -1) {
      param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   } 

   TreeStruct<T> tree;
   tree.Nv=0;
   for (int i = 0; i<num_groups; ++i) tree.Nv+=pr_N_own_variables[i];
   tree.Ng=num_groups;
   tree.weights=pr_lambda_g;
   tree.own_variables=pr_own_variables;
   tree.N_own_variables=pr_N_own_variables;
   tree.groups_ir=pr_groups_ir;
   tree.groups_jc=pr_groups_jc;

   if (param.intercept) {
      if (tree.Nv != pAlpha-1)
         mexErrMsgTxt("Error in tree definition");
   } else {
      if (tree.Nv != pAlpha)
         mexErrMsgTxt("Error in tree definition");
   }
   if (param.regul==TREEMULT && abs<T>(param.lambda2 - 0) < 1e-20) {
      mexErrMsgTxt("Error: with multi-task-tree, lambda2 should be > 0");
   }

   if (param.regul==GRAPH || param.regul==GRAPHMULT) 
      mexErrMsgTxt("Error: mexProximalGraph should be used instead");

   Vector<T> val;
   FISTA::PROX<T>(alpha0,alpha,param,val,NULL,&tree);
   if (nlhs==2) {
      plhs[1]=createMatrix<T>(1,val.n());
      T* pr_val=reinterpret_cast<T*>(mxGetPr(plhs[1]));
      for (int i = 0; i<val.n(); ++i) pr_val[i]=val[i];
   }
}

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs != 3)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (nlhs != 1 && nlhs != 2) 
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nlhs);
      } else {
         callFunction<float>(plhs,prhs,nlhs);
      }
   }




