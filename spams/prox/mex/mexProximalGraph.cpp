
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

// alpha = mexProximalGraph(X,graph,param)

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

   mxArray* ppr_GG = mxGetField(prhs[1],0,"groups");
   if (!mxIsSparse(ppr_GG)) 
      mexErrMsgTxt("field groups should be sparse");
   mwSize* GG_r=mxGetIr(ppr_GG);
   mwSize* GG_pB=mxGetJc(ppr_GG);
   const mwSize* dims_GG=mxGetDimensions(ppr_GG);
   int GGm=static_cast<int>(dims_GG[0]);
   int GGn=static_cast<int>(dims_GG[1]);
   if (GGm != GGn)
      mexErrMsgTxt("size of field groups is not consistent");

   mxArray* ppr_GV = mxGetField(prhs[1],0,"groups_var");
   if (!mxIsSparse(ppr_GV)) 
      mexErrMsgTxt("field groups_var should be sparse");
   mwSize* GV_r=mxGetIr(ppr_GV);
   mwSize* GV_pB=mxGetJc(ppr_GV);
   const mwSize* dims_GV=mxGetDimensions(ppr_GV);
   int nV=static_cast<int>(dims_GV[0]);
   int nG=static_cast<int>(dims_GV[1]);
   if (nV <= 0 || nG != GGn)
      mexErrMsgTxt("size of field groups-var is not consistent");

   mxArray* ppr_weights = mxGetField(prhs[1],0,"eta_g");
   if (mxIsSparse(ppr_weights)) 
      mexErrMsgTxt("field eta_g should not be sparse");
   T* pr_weights = reinterpret_cast<T*>(mxGetPr(ppr_weights));
   const mwSize* dims_weights=mxGetDimensions(ppr_weights);
   int mm1=static_cast<int>(dims_weights[0]);
   int nnG=static_cast<int>(dims_weights[1]);
   if (mm1 != 1 || nnG != nG)
      mexErrMsgTxt("size of field eta_g is not consistent");

   plhs[0]=createMatrix<T>(pAlpha,nAlpha);
   T* pr_alpha=reinterpret_cast<T*>(mxGetPr(plhs[0]));
   Matrix<T> alpha(pr_alpha,pAlpha,nAlpha);

   FISTA::ParamFISTA<T> param;
   param.num_threads = getScalarStructDef<int>(prhs[2],"numThreads",-1);
   param.pos = getScalarStructDef<bool>(prhs[2],"pos",false);
   param.lambda= getScalarStructDef<T>(prhs[2],"lambda",T(1.0));
   param.lambda2= getScalarStructDef<T>(prhs[2],"lambda2",0);
   param.lambda3= getScalarStructDef<T>(prhs[2],"lambda3",0);
   param.size_group= getScalarStructDef<int>(prhs[2],"size_group",1);
   getStringStruct(prhs[2],"regul",param.name_regul,param.length_names);
   param.regul = regul_from_string(param.name_regul);
   if (param.regul==INCORRECT_REG)
      mexErrMsgTxt("Unknown regularization");
   param.intercept = getScalarStructDef<bool>(prhs[2],"intercept",false);
   param.resetflow = getScalarStructDef<bool>(prhs[2],"resetflow",false);
   param.transpose = getScalarStructDef<bool>(prhs[2],"transpose",false);
   param.verbose = getScalarStructDef<bool>(prhs[2],"verbose",false);
   param.clever = getScalarStructDef<bool>(prhs[2],"clever",true);
   param.eval = nlhs==2;

   if (param.num_threads == -1) {
      param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   } 
   if (param.regul==GRAPHMULT && isZero(param.lambda2)) {
      mexErrMsgTxt("Error: with multi-task-graph, lambda2 should be > 0");
   }

   if (param.regul==TREE_L0 || param.regul==TREEMULT || param.regul==TREE_L2 || param.regul==TREE_LINF) 
      mexErrMsgTxt("Error: mexProximalTree should be used instead");
   GraphStruct<T> graph;
   graph.Nv=nV;
   graph.Ng=nG;
   graph.weights=pr_weights;
   graph.gg_ir=GG_r;
   graph.gg_jc=GG_pB;
   graph.gv_ir=GV_r;
   graph.gv_jc=GV_pB;

   Vector<T> val;
   FISTA::PROX<T>(alpha0,alpha,param,val,&graph);
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




