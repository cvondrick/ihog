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
  const char *names[] = {"eta_g", "groups", "groups_var"};
  plhs[0]=mxCreateStructArray(1,dims,3,names);
  SpMatrix<bool> *groups;
  SpMatrix<bool> *groups_var;
  Vector<double> *eta_g = _graphOfGroupStruct<double>(gstruct,&groups,&groups_var);
  del_gstruct(gstruct);
  mxArray* mxeta_g = makeVector<double>(eta_g);
  delete eta_g;
  mxSetField(plhs[0],0,"eta_g",mxeta_g);
  mxArray* mxgroups[1];
  convertSpMatrix<bool>(mxgroups[0],groups->m(),groups->n(),groups->n(),
			groups->nzmax(),groups->v(),groups->r(),groups->pB());
  mxArray* mxgroups_var;
  convertSpMatrix<bool>(mxgroups_var,groups_var->m(),groups_var->n(),groups_var->n(),
			groups_var->nzmax(),groups_var->v(),groups_var->r(),groups_var->pB());
  mxSetField(plhs[0],0,"groups",mxgroups[0]);
  mxSetField(plhs[0],0,"groups_var",mxgroups_var);
  delete groups;
  delete groups_var;
}



void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
  if (nrhs != 1)
    mexErrMsgTxt("Bad number of input arguments");
  
  if (nlhs != 1) 
    mexErrMsgTxt("Bad number of output arguments");
  
  callFunction(plhs,prhs,nlhs);
}




