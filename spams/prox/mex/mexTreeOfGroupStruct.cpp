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
#include <groups-graph.h>
#include <mexgrouputils.h>
#include<stdio.h>

inline void callFunction(mxArray* plhs[], const mxArray*prhs[],const int nlhs) {
  if (!mxIsCell(prhs[0])) 
    mexErrMsgTxt("argument 2 should be a cell");
  std::vector<NodeElem *> *gstruct = mexMatlabToCgroups(prhs[0]);
  mwSize dims[1] = {1};
  const char *names[] = {"eta_g", "groups", "own_variables","N_own_variables"};
  plhs[1]=mxCreateStructArray(1,dims,4,names);
  SpMatrix<bool> *groups;
  Vector<int> *own_variables;
  Vector<int> *N_own_variables;
  Vector<double> *eta_g;
  int *permutations;
  int nb_perm;
  int nb_vars = _treeOfGroupStruct<double>(gstruct,&permutations,&nb_perm,&eta_g,&groups,&own_variables,&N_own_variables);
  del_gstruct(gstruct);
  mxArray* mxeta_g = makeVector<double>(eta_g);
  mxArray* mxown_variables = makeVector<int>(own_variables);
  mxArray* mxN_own_variables = makeVector<int>(N_own_variables);
  mxArray* mxgroups[1];
  convertSpMatrix<bool>(mxgroups[0],groups->m(),groups->n(),groups->n(),
			groups->nzmax(),groups->v(),groups->r(),groups->pB());
  delete eta_g;
  delete groups;
  delete own_variables;
  delete N_own_variables;
  mxSetField(plhs[1],0,"eta_g",mxeta_g);
  mxSetField(plhs[1],0,"groups",mxgroups[0]);
  mxSetField(plhs[1],0,"own_variables",mxown_variables);
  mxSetField(plhs[1],0,"N_own_variables",mxN_own_variables);
  dims[0] = nb_perm;
  mxArray *mxperm = mxCreateNumericArray((mwSize)1,dims,mxINT32_CLASS,mxREAL);
  if(nb_perm > 0)
    memcpy(mxGetPr(mxperm),permutations,nb_perm * sizeof(int));
  plhs[0] = mxperm;
  dims[0] = 1;
  plhs[2]=mxCreateNumericArray((mwSize)1,dims,mxINT32_CLASS,mxREAL);
  int* pr_out=reinterpret_cast<int *>(mxGetPr(plhs[2]));
  *pr_out = nb_vars;
}



void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
  if (nrhs != 1)
    mexErrMsgTxt("Bad number of input arguments");
  
  if (nlhs != 3) 
    mexErrMsgTxt("Bad number of output arguments");
  
  callFunction(plhs,prhs,nlhs);
}




