/*!
 * \file
 *                toolbox Prox
 *
 *                by Jean-Paul Chièze
 *                
 *
 *                File mexgrouputils.h
 * \brief Contains miscellaneous functions for group structs in mex files */

#ifndef MEXGROUPUTILS_H
#define MEXGROUPUTILS_H

#include <mex.h>
#include <typeinfo>
#include <stdlib.h>
#include <iostream>
#ifndef MATLAB_MEX_FILE
#define MATLAB_MEX_FILE
#endif
#include <groups-graph.h>


typedef StructNodeElem<double> NodeElem;

void mexCgroupsToMatlab(std::vector<NodeElem *> *gstruct,mxArray* mxgstruct) {
  int i = 0;
  mwSize cdims[1];
  for(std::vector<NodeElem *>::iterator it = gstruct->begin();it != gstruct->end();it++, i++) {
    cdims[0] = 4;
    mxArray* mxnode = mxCreateCellArray((mwSize)1,cdims);
    NodeElem *node = *it;
    cdims[0] = node->vars->size();
    mxArray* vars = mxCreateNumericArray((mwSize)1,(const mwSize*)cdims,mxINT32_CLASS,mxREAL);
    memcpy(mxGetPr(vars),node->vars->data(),cdims[0] * sizeof(int));
    cdims[0] = node->children->size();
    mxArray* children = mxCreateNumericArray((mwSize)1,cdims,mxINT32_CLASS,mxREAL);
    memcpy(mxGetPr(children),node->children->data(),cdims[0] * sizeof(int));
    cdims[0] = 1;
    mxArray * mxnum = mxCreateNumericArray((mwSize)1,cdims,mxINT32_CLASS,mxREAL);
    int *pnum = reinterpret_cast<int *>(mxGetPr(mxnum));
    *pnum = node->node_num;
    mxSetCell(mxnode,0,mxnum);
    mxSetCell(mxnode,1,mxCreateDoubleScalar(node->weight));
    mxSetCell(mxnode,2,vars);
    mxSetCell(mxnode,3,children);
    mxSetCell(mxgstruct,(mwSize)i,mxnode);
  }
  del_gstruct(gstruct);

}
#define MEX_TRY(x) try{ x } catch(char const *_e) {mexErrMsgTxt(_e);}

std::vector<NodeElem *> *mexMatlabToCgroups(const mxArray *mxgstruct) {
  std::vector<NodeElem *> *gstruct = new std::vector<NodeElem *>;
  const mwSize* dims=mxGetDimensions(mxgstruct);
  for(int i = 0;i < *dims; i++) {
    mxArray *m_node = mxGetCell(mxgstruct,(mwIndex) i);
    int num_node = mxGetScalar(mxGetCell(m_node,0));
    float weight = mxGetScalar(mxGetCell(m_node,1));
    mxArray *mxvars = mxGetCell(m_node,2);
    int *pvars = reinterpret_cast<int *>(mxGetPr(mxvars));
    mxArray *mxchildren = mxGetCell(m_node,3);
    int *pchildren = reinterpret_cast<int *>(mxGetPr(mxchildren));
    const mwSize* tmpdims=mxGetDimensions(mxvars);
    std::vector<int> *vars = new std::vector<int>;
    for(int j= 0;j < *tmpdims;j++) vars->push_back(*(pvars+j));
    tmpdims = mxGetDimensions(mxchildren);
    std::vector<int> *children = new std::vector<int>;
    for(int j= 0;j < *tmpdims;j++) children->push_back(*(pchildren+j));
    NodeElem *node = new NodeElem(num_node,weight,vars,children);
    gstruct->push_back(node);
  }
  return gstruct;
}
template <> inline mxArray* createMatrix<int>(int m, int n) {
   return mxCreateNumericMatrix(static_cast<mwSize>(m),
         static_cast<mwSize>(n),mxINT32_CLASS,mxREAL);
};

template<typename T> mxArray *makeVector(Vector<T> *v) {
  mxArray *mxvect = createMatrix<T>(1,v->n());
  T *pv = reinterpret_cast<T*>(mxGetPr(mxvect));
  memcpy(pv,v->rawX(),v->n() * sizeof(T));
  return mxvect;
}

#endif // MEXGROUPUTILS_H
