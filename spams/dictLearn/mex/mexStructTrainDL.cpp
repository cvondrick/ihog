
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
 * Usage: [D model] = mexStructTrainDL(X,param);
 * Usage: [D model] = mexStructTrainDL(X,param,model);
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
      INTM n=static_cast<int>(dimsX[0]);
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
      mxArray* mxgraph = mxGetField(prhs[1],0,"graph");
      mxArray* mxtree = mxGetField(prhs[1],0,"tree");
      int K;
      if (!pr_D) {
	K = getScalarStruct<int>(prhs[1],"K");
         trainer = new Trainer<T>(K,batch_size,NUM_THREADS);
      } else {
         T* prD = reinterpret_cast<T*>(mxGetPr(pr_D));
         const mwSize* dimsD=mxGetDimensions(pr_D);
         int nD=static_cast<int>(dimsD[0]);
         K=static_cast<int>(dimsD[1]);
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
      FISTA::ParamFISTA<T> fista_param;
      param.lambda = getScalarStruct<T>(prhs[1],"lambda");
      param.lambda2 = getScalarStructDef<T>(prhs[1],"lambda2",10e-10);
      param.lambda3 = getScalarStructDef<T>(prhs[1],"lambda3",0.);
      param.iter=getScalarStruct<int>(prhs[1],"iter");
      param.t0 = getScalarStructDef<T>(prhs[1],"t0",1e-5);
      param.tol = getScalarStructDef<T>(prhs[1],"tol",0.000001);
      param.ista = getScalarStructDef<T>(prhs[1],"ista",false);
      param.fixed_step = getScalarStructDef<T>(prhs[1],"fixed_step",true);
  
      param.mode = FISTAMODE;
      getStringStruct(prhs[1],"regul",fista_param.name_regul,fista_param.length_names);
      param.regul = FISTA::regul_from_string(fista_param.name_regul);
      if (param.regul==FISTA::INCORRECT_REG)
	mexErrMsgTxt("Unknown regularization");
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
      /* graph */
      if (param.regul==FISTA::GRAPHMULT && abs<T>(param.lambda2 - 0) < 1e-20) {
	mexErrMsgTxt("Error: with multi-task-graph, lambda2 should be > 0");
      }
      GraphStruct<T> graph;
      GraphStruct<T> *pgraph = (GraphStruct<T> *) 0;
      if (param.regul==FISTA::GRAPH || param.regul==FISTA::GRAPH_RIDGE || 
	  param.regul==FISTA::GRAPH_L2) {
	if (!mxgraph) 
	  mexErrMsgTxt("param.graph is missing");
	pgraph = &graph;
	mxArray* ppr_GG = mxGetField(mxgraph,0,"groups");
	if (!mxIsSparse(ppr_GG)) 
	  mexErrMsgTxt("field groups should be sparse");
	mwSize* GG_r=mxGetIr(ppr_GG);
	mwSize* GG_pB=mxGetJc(ppr_GG);
	const mwSize* dims_GG=mxGetDimensions(ppr_GG);
	int GGm=static_cast<int>(dims_GG[0]);
	int GGn=static_cast<int>(dims_GG[1]);
	if (GGm != GGn)
	  mexErrMsgTxt("size of field groups is not consistent");

	mxArray* ppr_GV = mxGetField(mxgraph,0,"groups_var");
	if (!mxIsSparse(ppr_GV)) 
	  mexErrMsgTxt("field groups_var should be sparse");
	mwSize* GV_r=mxGetIr(ppr_GV);
	mwSize* GV_pB=mxGetJc(ppr_GV);
	const mwSize* dims_GV=mxGetDimensions(ppr_GV);
	int nV=static_cast<int>(dims_GV[0]);
	int nG=static_cast<int>(dims_GV[1]);
	if(nV != K)
	  mexErrMsgTxt("number of variables in graph must be equal to K");
	if (nV <= 0 || nG != GGn)
	  mexErrMsgTxt("size of field groups-var is not consistent");

	mxArray* ppr_weights = mxGetField(mxgraph,0,"eta_g");
	if (mxIsSparse(ppr_weights)) 
	  mexErrMsgTxt("field eta_g should not be sparse");
	T* pr_weights = reinterpret_cast<T*>(mxGetPr(ppr_weights));
	const mwSize* dims_weights=mxGetDimensions(ppr_weights);
	int mm1=static_cast<int>(dims_weights[0]);
	int nnG=static_cast<int>(dims_weights[1]);
	if (mm1 != 1 || nnG != nG)
	  mexErrMsgTxt("size of field eta_g is not consistent");
	
	graph.Nv=nV;
	graph.Ng=nG;
	graph.weights=pr_weights;
	graph.gg_ir=GG_r;
	graph.gg_jc=GG_pB;
	graph.gv_ir=GV_r;
	graph.gv_jc=GV_pB;
	   
      }
      /* tree */
      TreeStruct<T> tree;
      TreeStruct<T> *ptree = (TreeStruct<T> *) 0;
      tree.Nv=0;
      if (param.regul==FISTA::TREE_L0 || param.regul==FISTA::TREE_L2 || param.regul==FISTA::TREE_LINF) {
         mxArray* ppr_own_variables = mxGetField(mxtree,0,"own_variables");
         if (!mexCheckType<int>(ppr_own_variables)) 
            mexErrMsgTxt("own_variables field should be int32");
         if (!ppr_own_variables) mexErrMsgTxt("field own_variables is not provided");
         int* pr_own_variables = reinterpret_cast<int*>(mxGetPr(ppr_own_variables));
         const mwSize* dims_groups =mxGetDimensions(ppr_own_variables);
         int num_groups=static_cast<int>(dims_groups[0])*static_cast<int>(dims_groups[1]);
         mxArray* ppr_N_own_variables = mxGetField(mxtree,0,"N_own_variables");
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
         mxArray* ppr_lambda_g = mxGetField(mxtree,0,"eta_g");
         if (!ppr_lambda_g) mexErrMsgTxt("field eta_g is not provided");
         const mwSize* dims_weights =mxGetDimensions(ppr_lambda_g);
         int num_groups3=static_cast<int>(dims_weights[0])*static_cast<int>(dims_weights[1]);
         if (num_groups != num_groups3)
            mexErrMsgTxt("Error in tree definition");
         T* pr_lambda_g = reinterpret_cast<T*>(mxGetPr(ppr_lambda_g));
         mxArray* ppr_groups = mxGetField(mxtree,0,"groups");
         const mwSize* dims_gg =mxGetDimensions(ppr_groups);
         if ((num_groups != static_cast<int>(dims_gg[0])) || 
               (num_groups != static_cast<int>(dims_gg[1])))
            mexErrMsgTxt("Error in tree definition");
         if (!ppr_groups) mexErrMsgTxt("field groups is not provided");
         mwSize* pr_groups_ir = reinterpret_cast<mwSize*>(mxGetIr(ppr_groups));
         mwSize* pr_groups_jc = reinterpret_cast<mwSize*>(mxGetJc(ppr_groups));

         for (int i = 0; i<num_groups; ++i) tree.Nv+=pr_N_own_variables[i];
         tree.Ng=num_groups;
         tree.weights=pr_lambda_g;
         tree.own_variables=pr_own_variables;
         tree.N_own_variables=pr_N_own_variables;
         tree.groups_ir=pr_groups_ir;
         tree.groups_jc=pr_groups_jc;
         ptree=&tree;
      }


      /* */
      trainer->train_fista(*X,param,pgraph,ptree);
      if (param.log)
         mxFree(param.logName);

      Matrix<T> D;
      trainer->getD(D);
      K  = D.n();
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
      if (nrhs < 2 && nrhs > 3)
         mexErrMsgTxt("Bad number of inputs arguments");

      if ((nlhs < 1) && (nlhs > 2))
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nlhs,nrhs);
      } else {
         callFunction<float>(plhs,prhs,nlhs,nrhs);
      }
   }




