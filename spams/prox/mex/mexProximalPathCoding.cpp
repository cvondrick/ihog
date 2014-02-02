/* Software SPAMS v2.3 - Copyright 2009-2011 Julien Mairal 
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

   mxArray* ppr_GG = mxGetField(prhs[1],0,"weights");
   if (!mxIsSparse(ppr_GG)) 
      mexErrMsgTxt("field groups should be sparse");
   T* graph_weights = reinterpret_cast<T*>(mxGetPr(ppr_GG));
   mwSize* GG_r=mxGetIr(ppr_GG);
   mwSize* GG_pB=mxGetJc(ppr_GG);
   const mwSize* dims_GG=mxGetDimensions(ppr_GG);
   int GGm=static_cast<int>(dims_GG[0]);
   int GGn=static_cast<int>(dims_GG[1]);
   if (GGm != GGn || GGm != pAlpha)
      mexErrMsgTxt("size of field groups is not consistent");

   mxArray* ppr_weights = mxGetField(prhs[1],0,"start_weights");
   if (mxIsSparse(ppr_weights)) 
      mexErrMsgTxt("field start_weights should not be sparse");
   T* start_weights = reinterpret_cast<T*>(mxGetPr(ppr_weights));
   const mwSize* dims_weights=mxGetDimensions(ppr_weights);
   int nweights=static_cast<int>(dims_weights[0])*static_cast<int>(dims_weights[1]);
   if (nweights != pAlpha)
      mexErrMsgTxt("size of field start_weights is not consistent");

   mxArray* ppr_weights2 = mxGetField(prhs[1],0,"stop_weights");
   if (mxIsSparse(ppr_weights2)) 
      mexErrMsgTxt("field stop_weights should not be sparse");
   T* stop_weights = reinterpret_cast<T*>(mxGetPr(ppr_weights2));
   const mwSize* dims_weights2=mxGetDimensions(ppr_weights2);
   int nweights2=static_cast<int>(dims_weights2[0])*static_cast<int>(dims_weights2[1]);
   if (nweights2 != pAlpha)
      mexErrMsgTxt("size of field stop_weights is not consistent");


   plhs[0]=createMatrix<T>(pAlpha,nAlpha);
   T* pr_alpha=reinterpret_cast<T*>(mxGetPr(plhs[0]));
   Matrix<T> alpha(pr_alpha,pAlpha,nAlpha);

   FISTA::ParamFISTA<T> param;
   param.num_threads = getScalarStructDef<int>(prhs[2],"numThreads",-1);
   param.pos = getScalarStructDef<bool>(prhs[2],"pos",false);
   param.lambda= getScalarStructDef<T>(prhs[2],"lambda",T(1.0));
   param.lambda2= getScalarStructDef<T>(prhs[2],"lambda2",0);
   getStringStruct(prhs[2],"regul",param.name_regul,param.length_names);
   param.regul = regul_from_string(param.name_regul);
   if (param.regul==INCORRECT_REG)
      mexErrMsgTxt("Unknown regularization");
   param.intercept = getScalarStructDef<bool>(prhs[2],"intercept",false);
   param.verbose = getScalarStructDef<bool>(prhs[2],"verbose",false);
   param.transpose = getScalarStructDef<bool>(prhs[2],"transpose",false);
   param.eval = nlhs==2;

   if (param.regul != GRAPH_PATH_L0 && param.regul != GRAPH_PATH_CONV)
      mexErrMsgTxt("Use a different mexProximal* function");

   if (param.num_threads == -1) {
      param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   }
   
   GraphPathStruct<T> graph;
   graph.n=pAlpha;
   graph.m=GG_pB[graph.n]-GG_pB[0];
   graph.weights=graph_weights;
   graph.start_weights=start_weights;
   graph.stop_weights=stop_weights;
   graph.ir=GG_r;
   graph.jc=GG_pB;
   graph.precision = getScalarStructDef<long long>(prhs[2],"precision",100000000000000000);

   Vector<T> val;
   FISTA::PROX<T>(alpha0,alpha,param,val,NULL,NULL,&graph);
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




