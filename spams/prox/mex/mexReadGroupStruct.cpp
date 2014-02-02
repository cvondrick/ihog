 
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
  if (mxGetClassID((prhs[0])) !=  mxCHAR_CLASS)
    mexErrMsgTxt("mexSimpleGroupTree : type of argument 1 must be string");
  char fname[2048];
  std::vector<NodeElem *> *gstruct;
  if(mxGetString(prhs[0],fname,2048)) 
    mexErrMsgTxt("Pb with filename\n");
  MEX_TRY(gstruct = _readGroupStruct<double>(fname);)
  mwSize cdims[1];
  cdims[0] = gstruct->size();
  mxArray* mxgstruct = mxCreateCellArray((mwSize)1,cdims);
  mexCgroupsToMatlab(gstruct,mxgstruct);
  plhs[0] = mxgstruct;
  // mxCreateCellArray
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
  if (nrhs != 1)
    mexErrMsgTxt("Bad number of input arguments");
  
  if (nlhs != 1) 
    mexErrMsgTxt("Bad number of output arguments");
  
  callFunction(plhs,prhs,nlhs);
}




