/*!
 * \file
 *                toolbox Linalg 
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File mexutils.h
 * \brief Contains miscellaneous functions for mex files */

#ifndef MEXUTILS_H
#define MEXUTILS_H

#include <mex.h>
#include <typeinfo>
#include <stdlib.h>
#include <iostream>
#ifndef MATLAB_MEX_FILE
#define MATLAB_MEX_FILE
#endif
#include <utils.h>
#include <misc.h>


//#ifndef EM64T
//#define mwSize int
//#endif

/// Check the type of an array
template <typename T>
bool mexCheckType(const mxArray* array);

/// Check the type of an array (double)
template <> inline bool mexCheckType<double>(const mxArray* array) {
   return mxGetClassID(array) == mxDOUBLE_CLASS && !mxIsComplex(array);
};

/// Check the type of an array (float)
template <> inline bool mexCheckType<float>(const mxArray* array) {
   return mxGetClassID(array) == mxSINGLE_CLASS && !mxIsComplex(array);
};

/// Check the type of an array (int)
template <> inline bool mexCheckType<int>(const mxArray* array) {
   return mxGetClassID(array) == mxINT32_CLASS && !mxIsComplex(array);
};

/// Check the type of an array (int)
template <> inline bool mexCheckType<bool>(const mxArray* array) {
   return mxGetClassID(array) == mxLOGICAL_CLASS && !mxIsComplex(array);
};


/// Check the size of a 2D-array
bool checkSize(const mxArray* array, const int m, const int n) {
   const mwSize* dims=mxGetDimensions(array);
   int _m=static_cast<int>(dims[0]);
   int _n=static_cast<int>(dims[1]);
   return _n==n && _m==m;
};

/// Create a sparse copy of an array. Useful to deal with non-standard 
/// sparse matlab matrices
template <typename T>
void createCopySparse(T*& alpha_v2, int*& alpha_r2, int*& alpha_pB2, int*& alpha_pE2,
      double* alpha_v, mwSize* alpha_r, mwSize* alpha_pB, mwSize* alpha_pE, int M) {
   if (typeid(alpha_v) == typeid(alpha_v2)) {
      alpha_v2=reinterpret_cast<T*>(alpha_v);
   } else {
      alpha_v2 = new T[alpha_pB[M]];
      for (mwSize i = 0; i<alpha_pB[M]; ++i) alpha_v2[i] = static_cast<T>(alpha_v[i]);
   }
   if (typeid(alpha_r2) == typeid(alpha_r)) {
      alpha_r2=reinterpret_cast<int*>(alpha_r);
      alpha_pB2=reinterpret_cast<int*>(alpha_pB);
      alpha_pE2=reinterpret_cast<int*>(alpha_pE);
   } else {
      alpha_r2= new int[alpha_pB[M]];
      for (mwSize i = 0; i<alpha_pB[M]; ++i) alpha_r2[i]=static_cast<int>(alpha_r[i]);
      alpha_pB2= new int[M+1];
      for (int i = 0; i<=M; ++i) alpha_pB2[i]=static_cast<int>(alpha_pB[i]);
      alpha_pE2=alpha_pB2+1;
   }
};

/// Delete a sparse matrix which has been created using createCopySparse
template <typename T>
inline void deleteCopySparse(T*& alpha_v2, int*& alpha_r2, int*& alpha_pB2, int*& alpha_pE2,
      double* alpha_v, mwSize* alpha_r) {
   if (typeid(alpha_v) != typeid(alpha_v2)) {
      delete[](alpha_v2);
   }
   if (typeid(alpha_r2) != typeid(alpha_r)) {
      delete[](alpha_r2);
      delete[](alpha_pB2);
   }
   alpha_v2=NULL;
   alpha_r2=NULL;
   alpha_pB2=NULL;
   alpha_pE2=NULL;
};

/// Create a m x n matrix
template <typename T>
inline mxArray* createMatrix(int m, int n);

/// Create a m x n double matrix
template <> inline mxArray* createMatrix<double>(int m, int n) {
   return mxCreateNumericMatrix(static_cast<mwSize>(m),
         static_cast<mwSize>(n),mxDOUBLE_CLASS,mxREAL);
};

/// Create a m x n float matrix
template <> inline mxArray* createMatrix<float>(int m, int n) {
   return mxCreateNumericMatrix(static_cast<mwSize>(m),
         static_cast<mwSize>(n),mxSINGLE_CLASS,mxREAL);
};

/// Create a h x w x V image
template <typename T>
inline mxArray* createImage(int h, int w, int V);

/// Create a h x w x V double image
template <>
inline mxArray* createImage<double>(int h, int w, int V) {
   if (V ==1) {
      return createMatrix<double>(h,w);
   } else {
      mwSize dims[3];
      dims[0]=h;
      dims[1]=w;
      dims[2]=V;
      return mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
   }
}

/// Create a h x w x V float image
template <>
inline mxArray* createImage<float>(int h, int w, int V) {
   if (V ==1) {
      return createMatrix<float>(h,w);
   } else {
      mwSize dims[3];
      dims[0]=h;
      dims[1]=w;
      dims[2]=V;
      return mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
   }
}

/// Create a scalar
template <typename T> inline mxArray* createScalar() {
   return createMatrix<T>(1,1);
};

/// convert sparse matrix to Matlab sparse matrix
template <typename T> inline void convertSpMatrix(mxArray*& matlab_mat, int K,
      int M, int n, int nzmax, const T* v, const int* r, const int* pB) {
   matlab_mat=mxCreateSparse(K,M,nzmax,mxREAL);
   double* Pr=mxGetPr(matlab_mat);
   for (int i = 0; i<nzmax; ++i) Pr[i]=static_cast<double>(v[i]);
   mwSize* Ir=mxGetIr(matlab_mat);
   for (int i = 0; i<nzmax; ++i) Ir[i]=static_cast<mwSize>(r[i]);
   mwSize* Jc=mxGetJc(matlab_mat);
   if (n == 0) return;
   for (int i = 0; i<=n; ++i) Jc[i]=static_cast<mwSize>(pB[i]);
};

/// get a scalar from a struct
template <typename T> inline T getScalarStruct(const mxArray* pr_struct,
      const char* name) {
   mxArray *pr_field = mxGetField(pr_struct,0,name);
   if (!pr_field) {
      mexPrintf("Missing field: ");
      mexErrMsgTxt(name);
   }
   return static_cast<T>(mxGetScalar(pr_field));
};

/// get a scalar from a struct
inline void getStringStruct(const mxArray* pr_struct,
      const char* name, char* field, const mwSize length) {
   mxArray *pr_field = mxGetField(pr_struct,0,name);
   if (!pr_field) {
      mexPrintf("Missing field: ");
      mexErrMsgTxt(name);
   }
   mxGetString(pr_field,field,length);
};

/// get a scalar from a struct
inline bool checkField(const mxArray* pr_struct,
      const char* name) {
   mxArray *pr_field = mxGetField(pr_struct,0,name);
   if (!pr_field) {
      mexPrintf("Missing field: ");
      mexPrintf(name);
      return false;
   }
   return true;
};

/// get a scalar from a struct  and provide a default value
template <typename T> inline T getScalarStructDef(const mxArray* pr_struct,
      const char* name, const T def) {
   mxArray *pr_field = mxGetField(pr_struct,0,name);
   return pr_field ? (T)(mxGetScalar(pr_field)) :
      def;
}

void super_flush(std::ostream& stream) {
    std::flush(stream);   
    mexEvalString("pause(0.0000000001);"); // to dump string.
}

#define flush super_flush

#endif
