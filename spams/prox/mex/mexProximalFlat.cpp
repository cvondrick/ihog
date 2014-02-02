

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

// alpha = mexProximalFlat(alpha0,param)

using namespace FISTA;

template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
      const int nlhs) {
   if (!mexCheckType<T>(prhs[0])) 
      mexErrMsgTxt("type of argument 1 is not consistent");
   if (mxIsSparse(prhs[0])) 
      mexErrMsgTxt("argument 1 should not be sparse");

   if (!mxIsStruct(prhs[1])) 
      mexErrMsgTxt("argument 2 should be a struct");

   T* pr_alpha0 = reinterpret_cast<T*>(mxGetPr(prhs[0]));
   const mwSize* dimsAlpha=mxGetDimensions(prhs[0]);
   int pAlpha=static_cast<int>(dimsAlpha[0]);
   int nAlpha=static_cast<int>(dimsAlpha[1]);
   Matrix<T> alpha0(pr_alpha0,pAlpha,nAlpha);

   plhs[0]=createMatrix<T>(pAlpha,nAlpha);
   T* pr_alpha=reinterpret_cast<T*>(mxGetPr(plhs[0]));
   Matrix<T> alpha(pr_alpha,pAlpha,nAlpha);

   FISTA::ParamFISTA<T> param;
   param.num_threads = getScalarStructDef<int>(prhs[1],"numThreads",-1);
   param.pos = getScalarStructDef<bool>(prhs[1],"pos",false);
   param.lambda= getScalarStructDef<T>(prhs[1],"lambda",1.0);
   param.lambda2= getScalarStructDef<T>(prhs[1],"lambda2",0.0);
   param.lambda3= getScalarStructDef<T>(prhs[1],"lambda3",0.0);
   mxArray* ppr_groups = mxGetField(prhs[1],0,"groups");
   if (ppr_groups) {
      if (!mexCheckType<int>(ppr_groups))
         mexErrMsgTxt("param.groups should be int32 (starting group is 1)");
      int* pr_groups = reinterpret_cast<int*>(mxGetPr(ppr_groups));
      const mwSize* dims_groups =mxGetDimensions(ppr_groups);
      int num_groups=static_cast<int>(dims_groups[0])*static_cast<int>(dims_groups[1]);
      if (num_groups != pAlpha) mexErrMsgTxt("Wrong size of param.groups");
      param.ngroups=num_groups;
      param.groups=pr_groups;
   } else {
      param.size_group= getScalarStructDef<int>(prhs[1],"size_group",1);
   }

   getStringStruct(prhs[1],"regul",param.name_regul,param.length_names);
   param.regul = regul_from_string(param.name_regul);
   if (param.regul==INCORRECT_REG)
      mexErrMsgTxt("Unknown regularization");
   param.intercept = getScalarStructDef<bool>(prhs[1],"intercept",false);
   param.verbose = getScalarStructDef<bool>(prhs[1],"verbose",false);
   param.clever = getScalarStructDef<bool>(prhs[1],"clever",true);
   param.resetflow = getScalarStructDef<bool>(prhs[1],"resetflow",false);
   param.transpose = getScalarStructDef<bool>(prhs[1],"transpose",false);
   param.eval = nlhs==2;

   if (param.num_threads == -1) {
      param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   }

   if (param.regul==GRAPH || param.regul==GRAPHMULT) 
      mexErrMsgTxt("Error: mexProximalGraph should be used instead");
   if (param.regul==TREE_L0 || param.regul==TREEMULT || param.regul==TREE_L2 || param.regul==TREE_LINF) 
      mexErrMsgTxt("Error: mexProximalTree should be used instead");

   Vector<T> val_reg;
   FISTA::PROX<T>(alpha0,alpha,param,val_reg);
   if (nlhs==2) {
      plhs[1]=createMatrix<T>(1,val_reg.n());
      T* pr_val=reinterpret_cast<T*>(mxGetPr(plhs[1]));
      for (int i = 0; i<val_reg.n(); ++i) pr_val[i]=val_reg[i];
   }
}

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs != 2)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (nlhs != 1 && nlhs != 2) 
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nlhs);
      } else {
         callFunction<float>(plhs,prhs,nlhs);
      }
   }




