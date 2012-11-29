#ifndef MKL_TEMPLATE
#define MKL_TEMPLATE

//#include "cblas.h" dependency on cblas has been removed

/// a few static variables for lapack
static char low='l';
static char nonUnit='n';
static char upper='u';
static int info=0;
static char incr='I';
static char decr='D';

/// external functions
#ifdef HAVE_MKL
extern "C" {
#endif
   size_t cblas_idamin(const int n, const double* X, const int incX);
   size_t cblas_isamin(const int n, const float* X, const int incX);
#ifdef HAVE_MKL
};
#endif

extern "C" {
   void dsytrf_(char* uplo, int* n, double* a, int* lda, int* ipiv,
         double* work, int* lwork, int* info);
   void ssytrf_(char* uplo, int* n, float* a, int* lda, int* ipiv,
         float* work, int* lwork, int* info);
   void dsytri_(char* uplo, int* n, double* a, int* lda, int* ipiv,
         double* work, int* info);
   void ssytri_(char* uplo, int* n, float* a, int* lda, int* ipiv,
         float* work, int* info);
   void dtrtri_(char* uplo, char* diag, int* n, double * a, int* lda, 
         int* info);
   void strtri_(char* uplo, char* diag, int* n, float * a, int* lda, 
         int* info);
   void dlasrt_(char* id, const int* n, double *d, int* info);
   void slasrt_(char* id, const int* n, float*d, int* info);
   void dlasrt2_(char* id, const int* n, double *d, int* key, int* info);
   void slasrt2_(char* id, const int* n, float*d, int* key, int* info);
};

#ifdef HAVE_MKL
extern "C" {
   void vdSqr(const int n, const double* vecIn, double* vecOut);
   void vsSqr(const int n, const float* vecIn, float* vecOut);
   void vdSqrt(const int n, const double* vecIn, double* vecOut);
   void vsSqrt(const int n, const float* vecIn, float* vecOut);
   void vdInvSqrt(const int n, const double* vecIn, double* vecOut);
   void vsInvSqrt(const int n, const float* vecIn, float* vecOut);
   void vdSub(const int n, const double* vecIn, const double* vecIn2, double* vecOut);
   void vsSub(const int n, const float* vecIn, const float* vecIn2, float* vecOut);
   void vdDiv(const int n, const double* vecIn, const double* vecIn2, double* vecOut);
   void vsDiv(const int n, const float* vecIn, const float* vecIn2, float* vecOut);
   void vdExp(const int n, const double* vecIn, double* vecOut);
   void vsExp(const int n, const float* vecIn, float* vecOut);
   void vdInv(const int n, const double* vecIn, double* vecOut);
   void vsInv(const int n, const float* vecIn, float* vecOut);
   void vdAdd(const int n, const double* vecIn, const double* vecIn2, double* vecOut);
   void vsAdd(const int n, const float* vecIn, const float* vecIn2, float* vecOut);
   void vdMul(const int n, const double* vecIn, const double* vecIn2, double* vecOut);
   void vsMul(const int n, const float* vecIn, const float* vecIn2, float* vecOut);
   void vdAbs(const int n, const double* vecIn, double* vecOut);
   void vsAbs(const int n, const float* vecIn, float* vecOut);
}
#endif


// Interfaces to a few BLAS function, Level 1
/// interface to cblas_*nrm2
template <typename T> T cblas_nrm2(const int n, const T* X, const int incX);
/// interface to cblas_*copy
template <typename T> void cblas_copy(const int n, const T* X, const int incX, 
      T* Y, const int incY);
/// interface to cblas_*axpy
template <typename T> void cblas_axpy(const int n, const T a, const T* X, 
      const int incX, T* Y, const int incY);
/// interface to cblas_*scal
template <typename T> void cblas_scal(const int n, const T a, T* X, 
      const int incX);
/// interface to cblas_*asum
template <typename T> T cblas_asum(const int n, const T* X, const int incX);
/// interface to cblas_*adot
template <typename T> T cblas_dot(const int n, const T* X, const int incX, 
      const T* Y,const int incY);
/// interface to cblas_i*amin
template <typename T> int cblas_iamin(const int n, const T* X, const int incX);
/// interface to cblas_i*amax
template <typename T> int cblas_iamax(const int n, const T* X, const int incX);

// Interfaces to a few BLAS function, Level 2

/// interface to cblas_*gemv
template <typename T> void cblas_gemv(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const int M, 
      const int N, const T alpha, const T *A, const int lda, const T *X, 
      const int incX, const T beta,T *Y,  const int incY);
/// interface to cblas_*trmv
template <typename T> void inline cblas_trmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const int N,
      const T *A, const int lda, T *X, const int incX);
/// interface to cblas_*syr
template <typename T> void inline cblas_syr(const CBLAS_ORDER order, 
      const  CBLAS_UPLO Uplo, const int N, const T alpha, 
      const T *X, const int incX, T *A, const int lda);

/// interface to cblas_*symv
template <typename T> inline void cblas_symv(const CBLAS_ORDER order,
      const CBLAS_UPLO Uplo, const int N, 
      const T alpha, const T *A, const int lda, const T *X, 
      const int incX, const T beta,T *Y,  const int incY);


// Interfaces to a few BLAS function, Level 3
/// interface to cblas_*gemm
template <typename T> void cblas_gemm(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const int M, const int N, const int K, const T alpha, 
      const T *A, const int lda, const T *B, const int ldb,
      const T beta, T *C, const int ldc);
/// interface to cblas_*syrk
template <typename T> void cblas_syrk(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const int N, const int K,
      const T alpha, const T *A, const int lda,
      const T beta, T*C, const int ldc);
/// interface to cblas_*ger
template <typename T> void cblas_ger(const CBLAS_ORDER order, 
      const int M, const int N, const T alpha, const T *X, const int incX,
      const T* Y, const int incY, T*A, const int lda);
/// interface to cblas_*trmm
template <typename T> void cblas_trmm(const CBLAS_ORDER Order, 
      const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
      const int M, const int N, const T alpha, 
      const T*A, const int lda,T *B, const int ldb);

// Interfaces to a few functions from the Intel Vector Mathematical Library
/// interface to v*Sqr
template <typename T> void vSqrt(const int n, const T* vecIn, T* vecOut);
/// interface to v*Sqr
template <typename T> void vInvSqrt(const int n, const T* vecIn, T* vecOut);
/// interface to v*Sqr
template <typename T> void vSqr(const int n, const T* vecIn, T* vecOut);
/// interface to v*Sub
template <typename T> void vSub(const int n, const T* vecIn, const T* vecIn2, T* vecOut);
/// interface to v*Div
template <typename T> void vDiv(const int n, const T* vecIn, const T* vecIn2, T* vecOut);
/// interface to v*Exp
template <typename T> void vExp(const int n, const T* vecIn, T* vecOut);
/// interface to v*Inv
template <typename T> void vInv(const int n, const T* vecIn, T* vecOut);
/// interface to v*Add
template <typename T> void vAdd(const int n, const T* vecIn, const T* vecIn2, T* vecOut);
/// interface to v*Mul
template <typename T> void vMul(const int n, const T* vecIn, const T* vecIn2, T* vecOut);
/// interface to v*Abs
template <typename T> void vAbs(const int n, const T* vecIn, T* vecOut);

// Interfaces to a few LAPACK functions
/// interface to *trtri
template <typename T> void trtri(char& uplo, char& diag, 
      int& n, T * a, int& lda);
/// interface to *sytrf
template <typename T> void sytrf(char& uplo, int& n, T* a, int& lda, int* ipiv,
      T* work, int& lwork);
/// interface to *sytri
template <typename T> void sytri(char& uplo, int& n, T* a, int& lda, int* ipiv,
      T* work);
/// interaface to *lasrt
template <typename T> void lasrt(char& id, const int& n, T *d);
template <typename T> void lasrt2(char& id, const int& n, T *d, int* key);



/* ******************
 * Implementations
 * *****************/


// Implementations of the interfaces, BLAS Level 1
/// Implementation of the interface for cblas_dnrm2
template <> inline double cblas_nrm2<double>(const int n, const double* X, 
      const int incX) {
   return cblas_dnrm2(n,X,incX);
};
/// Implementation of the interface for cblas_snrm2
template <> inline float cblas_nrm2<float>(const int n, const float* X, 
      const int incX) {
   return cblas_snrm2(n,X,incX);
};
/// Implementation of the interface for cblas_dcopy
template <> inline void cblas_copy<double>(const int n, const double* X, 
      const int incX, double* Y, const int incY) {
   cblas_dcopy(n,X,incX,Y,incY);
};
/// Implementation of the interface for cblas_scopy
template <> inline void cblas_copy<float>(const int n, const float* X, const int incX, 
      float* Y, const int incY) {
   cblas_scopy(n,X,incX,Y,incY);
};
/// Implementation of the interface for cblas_scopy
template <> inline void cblas_copy<int>(const int n, const int* X, const int incX, 
      int* Y, const int incY) {
   for (int i = 0; i<n; ++i)
      Y[incY*i]=X[incX*i];
};
/// Implementation of the interface for cblas_scopy
template <> inline void cblas_copy<bool>(const int n, const bool* X, const int incX, 
      bool* Y, const int incY) {
   for (int i = 0; i<n; ++i)
      Y[incY*i]=X[incX*i];
};

/// Implementation of the interface for cblas_daxpy
template <> inline void cblas_axpy<double>(const int n, const double a, const double* X, 
      const int incX, double* Y, const int incY) {
   cblas_daxpy(n,a,X,incX,Y,incY);
};
/// Implementation of the interface for cblas_saxpy
template <> inline void cblas_axpy<float>(const int n, const float a, const float* X,
      const int incX, float* Y, const int incY) {
   cblas_saxpy(n,a,X,incX,Y,incY);
};

/// Implementation of the interface for cblas_saxpy
template <> inline void cblas_axpy<int>(const int n, const int a, const int* X,
      const int incX, int* Y, const int incY) {
   for (int i = 0; i<n; ++i)
      Y[i] += a*X[i];
};

/// Implementation of the interface for cblas_saxpy
template <> inline void cblas_axpy<bool>(const int n, const bool a, const bool* X,
      const int incX, bool* Y, const int incY) {
   for (int i = 0; i<n; ++i)
      Y[i] = a*X[i];
};


/// Implementation of the interface for cblas_dscal
template <> inline void cblas_scal<double>(const int n, const double a, double* X,
      const int incX) {
   cblas_dscal(n,a,X,incX);
};
/// Implementation of the interface for cblas_sscal
template <> inline void cblas_scal<float>(const int n, const float a, float* X, 
      const int incX) {
   cblas_sscal(n,a,X,incX);
};
/// Implementation of the interface for cblas_sscal
template <> inline void cblas_scal<int>(const int n, const int a, int* X, 
      const int incX) {
   for (int i = 0; i<n; ++i) X[i*incX]*=a;
};
/// Implementation of the interface for cblas_sscal
template <> inline void cblas_scal<bool>(const int n, const bool a, bool* X, 
      const int incX) {
   /// not implemented
};

/// Implementation of the interface for cblas_dasum
template <> inline double cblas_asum<double>(const int n, const double* X, const int incX) {
   return cblas_dasum(n,X,incX);
};
/// Implementation of the interface for cblas_sasum
template <> inline float cblas_asum<float>(const int n, const float* X, const int incX) {
   return cblas_sasum(n,X,incX);
};
/// Implementation of the interface for cblas_ddot
template <> inline double cblas_dot<double>(const int n, const double* X,
      const int incX, const double* Y,const int incY) {
   return cblas_ddot(n,X,incX,Y,incY);
};
/// Implementation of the interface for cblas_sdot
template <> inline float cblas_dot<float>(const int n, const float* X,
      const int incX, const float* Y,const int incY) {
   return cblas_sdot(n,X,incX,Y,incY);
};
template <> inline int cblas_dot<int>(const int n, const int* X,
      const int incX, const int* Y,const int incY) {
   int total=0;
   int i,j;
   j=0;
   for (i = 0; i<n; ++i) {
      total+=X[i*incX]*Y[j];
      j+=incY;
   }
   return total;
};
/// Implementation of the interface for cblas_sdot
template <> inline bool cblas_dot<bool>(const int n, const bool* X,
      const int incX, const bool* Y,const int incY) {
   /// not implemented
   return true;
};

// Implementations of the interfaces, BLAS Level 2
///  Implementation of the interface for cblas_dgemv
template <> inline void cblas_gemv<double>(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const int M, const int N,
      const double alpha, const double *A, const int lda,
      const double *X, const int incX, const double beta,
      double *Y, const int incY) {
   cblas_dgemv(order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
};
///  Implementation of the interface for cblas_sgemv
template <> inline void cblas_gemv<float>(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const int M, const int N,
      const float alpha, const float *A, const int lda,
      const float *X, const int incX, const float beta,
      float *Y, const int incY) {
   cblas_sgemv(order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
};
///  Implementation of the interface for cblas_sgemv
template <> inline void cblas_gemv<int>(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const int M, const int N,
      const int alpha, const int *A, const int lda,
      const int *X, const int incX, const int beta,
      int *Y, const int incY) {
   ///  not implemented
};
///  Implementation of the interface for cblas_sgemv
template <> inline void cblas_gemv<bool>(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const int M, const int N,
      const bool alpha, const bool *A, const int lda,
      const bool *X, const int incX, const bool beta,
      bool *Y, const int incY) {
   /// not implemented
};

///  Implementation of the interface for cblas_dger
template <> inline void cblas_ger<double>(const CBLAS_ORDER order, 
      const int M, const int N, const double alpha, const double *X, const int incX,
      const double* Y, const int incY, double *A, const int lda) {
   cblas_dger(order,M,N,alpha,X,incX,Y,incY,A,lda);
};
///  Implementation of the interface for cblas_sger
template <> inline void cblas_ger<float>(const CBLAS_ORDER order, 
      const int M, const int N, const float alpha, const float *X, const int incX,
      const float* Y, const int incY, float *A, const int lda) {
   cblas_sger(order,M,N,alpha,X,incX,Y,incY,A,lda);
};
///  Implementation of the interface for cblas_dtrmv
template <> inline void cblas_trmv<double>(const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const int N,
      const double *A, const int lda, double *X, const int incX) {
   cblas_dtrmv(order,Uplo,TransA,Diag,N,A,lda,X,incX);
};
///  Implementation of the interface for cblas_strmv
template <> inline void cblas_trmv<float>(const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const int N,
      const float *A, const int lda, float *X, const int incX) {
   cblas_strmv(order,Uplo,TransA,Diag,N,A,lda,X,incX);
};
/// Implementation of cblas_dsyr
template <> inline void cblas_syr(const CBLAS_ORDER order, 
      const  CBLAS_UPLO Uplo,
      const int N, const double alpha, const double*X,
      const int incX, double *A, const int lda) {
   cblas_dsyr(order,Uplo,N,alpha,X,incX,A,lda);
};
/// Implementation of cblas_ssyr
template <> inline void cblas_syr(const CBLAS_ORDER order, 
      const  CBLAS_UPLO Uplo,
      const int N, const float alpha, const float*X,
      const int incX, float *A, const int lda) {
   cblas_ssyr(order,Uplo,N,alpha,X,incX,A,lda);
};
/// Implementation of cblas_ssymv
template <> inline void cblas_symv(const CBLAS_ORDER order,
      const CBLAS_UPLO Uplo, const int N, 
      const float alpha, const float *A, const int lda, const float *X, 
      const int incX, const float beta,float *Y,  const int incY) {
   cblas_ssymv(order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);
}
/// Implementation of cblas_dsymv
template <> inline void cblas_symv(const CBLAS_ORDER order,
      const CBLAS_UPLO Uplo, const int N, 
      const double alpha, const double *A, const int lda, const double *X, 
      const int incX, const double beta,double *Y,  const int incY) {
   cblas_dsymv(order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);
}


// Implementations of the interfaces, BLAS Level 3
///  Implementation of the interface for cblas_dgemm
template <> inline void cblas_gemm<double>(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const int M, const int N, const int K, const double alpha, 
      const double *A, const int lda, const double *B, const int ldb,
      const double beta, double *C, const int ldc) {
   cblas_dgemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
};
///  Implementation of the interface for cblas_sgemm
template <> inline void cblas_gemm<float>(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const int M, const int N, const int K, const float alpha, 
      const float *A, const int lda, const float *B, const int ldb,
      const float beta, float *C, const int ldc) {
   cblas_sgemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
};
template <> inline void cblas_gemm<int>(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const int M, const int N, const int K, const int alpha, 
      const int *A, const int lda, const int *B, const int ldb,
      const int beta, int *C, const int ldc) {
   /// not implemented
};
///  Implementation of the interface for cblas_sgemm
template <> inline void cblas_gemm<bool>(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const int M, const int N, const int K, const bool alpha, 
      const bool *A, const int lda, const bool *B, const int ldb,
      const bool beta, bool *C, const int ldc) {
   /// not implemented
};

///  Implementation of the interface for cblas_dsyrk
template <> inline void cblas_syrk<double>(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const int N, const int K,
      const double alpha, const double *A, const int lda,
      const double beta, double *C, const int ldc) {
   cblas_dsyrk(Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);
};
///  Implementation of the interface for cblas_ssyrk
template <> inline void cblas_syrk<float>(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const int N, const int K,
      const float alpha, const float *A, const int lda,
      const float beta, float *C, const int ldc) {
   cblas_ssyrk(Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);
};
///  Implementation of the interface for cblas_ssyrk
template <> inline void cblas_syrk<int>(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const int N, const int K,
      const int alpha, const int *A, const int lda,
      const int beta, int *C, const int ldc) {
   /// not implemented
};
///  Implementation of the interface for cblas_ssyrk
template <> inline void cblas_syrk<bool>(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const int N, const int K,
      const bool alpha, const bool *A, const int lda,
      const bool beta, bool *C, const int ldc) {
   /// not implemented
};

///  Implementation of the interface for cblas_dtrmm
template <> inline void cblas_trmm<double>(const CBLAS_ORDER Order, 
      const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
      const int M, const int N, const double alpha, 
      const double *A, const int lda,double *B, const int ldb) {
   cblas_dtrmm(Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);
};
///  Implementation of the interface for cblas_strmm
template <> inline void cblas_trmm<float>(const CBLAS_ORDER Order, 
      const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
      const int M, const int N, const float alpha, 
      const float *A, const int lda,float *B, const int ldb) {
   cblas_strmm(Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);
};
///  Implementation of the interface for cblas_idamax
template <> inline int cblas_iamax<double>(const int n, const double* X,
      const int incX) {
   return cblas_idamax(n,X,incX);
};
///  Implementation of the interface for cblas_isamax
template <> inline int cblas_iamax<float>(const int n, const float* X, 
      const int incX) {
   return cblas_isamax(n,X,incX);
};

// Implementations of the interfaces, LAPACK
/// Implemenation of the interface for dtrtri
template <> inline void trtri<double>(char& uplo, char& diag, 
      int& n, double * a, int& lda) {
   dtrtri_(&uplo,&diag,&n,a,&lda,&info);
};
/// Implemenation of the interface for strtri
template <> inline void trtri<float>(char& uplo, char& diag, 
      int& n, float* a, int& lda) {
   strtri_(&uplo,&diag,&n,a,&lda,&info);
};
/// Implemenation of the interface for dsytrf
template <> void inline sytrf<double>(char& uplo, int& n, double* a, 
      int& lda, int* ipiv, double* work, int& lwork) {
   dsytrf_(&uplo,&n,a,&lda,ipiv,work,&lwork,&info);
};
/// Implemenation of the interface for ssytrf
template <> inline void sytrf<float>(char& uplo, int& n, float* a, 
      int& lda, int* ipiv, float* work, int& lwork) {
   ssytrf_(&uplo,&n,a,&lda,ipiv,work,&lwork,&info);
};
/// Implemenation of the interface for dsytri
template <> inline void sytri<double>(char& uplo, int& n, double* a, 
      int& lda, int* ipiv, double* work) {
   dsytri_(&uplo,&n,a,&lda,ipiv,work,&info);
};
/// Implemenation of the interface for ssytri
template <> inline void sytri<float>(char& uplo, int& n, float* a, 
      int& lda, int* ipiv, float* work) {
   ssytri_(&uplo,&n,a,&lda,ipiv,work,&info);
};
/// interaface to *lasrt
template <> inline void lasrt(char& id, const int& n, double *d) {
   dlasrt_(&id,const_cast<int*>(&n),d,&info);
};
/// interaface to *lasrt
template <> inline void lasrt(char& id, const int& n, float *d) {
   slasrt_(&id,const_cast<int*>(&n),d,&info);
};
template <> inline void lasrt2(char& id, const int& n, double *d,int* key) {
   dlasrt2_(&id,const_cast<int*>(&n),d,key,&info);
};
/// interaface to *lasrt
template <> inline void lasrt2(char& id, const int& n, float *d, int* key) {
   slasrt2_(&id,const_cast<int*>(&n),d,key,&info);
};


/// If the MKL is not present, a slow implementation is used instead.
#ifdef HAVE_MKL 
/// Implemenation of the interface for vdSqr
template <> inline void vSqr<double>(const int n, const double* vecIn, 
      double* vecOut) {
   vdSqr(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsSqr
template <> inline void vSqr<float>(const int n, const float* vecIn, 
      float* vecOut) {
   vsSqr(n,vecIn,vecOut);
};
template <> inline void vSqrt<double>(const int n, const double* vecIn, 
      double* vecOut) {
   vdSqrt(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsSqr
template <> inline void vSqrt<float>(const int n, const float* vecIn, 
      float* vecOut) {
   vsSqrt(n,vecIn,vecOut);
};
template <> inline void vInvSqrt<double>(const int n, const double* vecIn, 
      double* vecOut) {
   vdInvSqrt(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsSqr
template <> inline void vInvSqrt<float>(const int n, const float* vecIn, 
      float* vecOut) {
   vsInvSqrt(n,vecIn,vecOut);
};

/// Implemenation of the interface for vdSub
template <> inline void vSub<double>(const int n, const double* vecIn, 
      const double* vecIn2, double* vecOut) {
   vdSub(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsSub
template <> inline void vSub<float>(const int n, const float* vecIn, 
      const float* vecIn2, float* vecOut) {
   vsSub(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vdDiv
template <> inline void vDiv<double>(const int n, const double* vecIn, 
      const double* vecIn2, double* vecOut) {
   vdDiv(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsDiv
template <> inline void vDiv<float>(const int n, const float* vecIn, 
      const float* vecIn2, float* vecOut) {
   vsDiv(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vdExp
template <> inline void vExp<double>(const int n, const double* vecIn, 
      double* vecOut) {
   vdExp(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsExp
template <> inline void vExp<float>(const int n, const float* vecIn, 
      float* vecOut) {
   vsExp(n,vecIn,vecOut);
};
/// Implemenation of the interface for vdInv
template <> inline void vInv<double>(const int n, const double* vecIn, 
      double* vecOut) {
   vdInv(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsInv
template <> inline void vInv<float>(const int n, const float* vecIn, 
      float* vecOut) {
   vsInv(n,vecIn,vecOut);
};
/// Implemenation of the interface for vdAdd
template <> inline void vAdd<double>(const int n, const double* vecIn, 
      const double* vecIn2, double* vecOut) {
   vdAdd(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsAdd
template <> inline void vAdd<float>(const int n, const float* vecIn, 
      const float* vecIn2, float* vecOut) {
   vsAdd(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vdMul
template <> inline void vMul<double>(const int n, const double* vecIn, 
      const double* vecIn2, double* vecOut) {
   vdMul(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsMul
template <> inline void vMul<float>(const int n, const float* vecIn, 
      const float* vecIn2, float* vecOut) {
   vsMul(n,vecIn,vecIn2,vecOut);
};

/// interface to vdAbs
template <> inline void vAbs(const int n, const double* vecIn, 
      double* vecOut) {
   vdAbs(n,vecIn,vecOut);
};
/// interface to vdAbs
template <> inline void vAbs(const int n, const float* vecIn, 
      float* vecOut) {
   vsAbs(n,vecIn,vecOut);
};


/// Implemenation of the interface of the non-offical BLAS, Level 1 function 
/// cblas_idamin
template <> inline int cblas_iamin<double>(const int n, const double* X,
      const int incX) {
   return (int) cblas_idamin(n,X,incX);
};
/// Implemenation of the interface of the non-offical BLAS, Level 1 function 
/// cblas_isamin
template <> inline int cblas_iamin<float>(const int n, const float* X, 
      const int incX) {
   return (int) cblas_isamin(n,X,incX);
};
/// slow alternative implementation of some MKL function
#else
/// Slow implementation of vdSqr and vsSqr
template <typename T> inline void vSqr(const int n, const T* vecIn, T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=vecIn[i]*vecIn[i];
};
template <typename T> inline void vSqrt(const int n, const T* vecIn, T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=sqr<T>(vecIn[i]);
};
template <typename T> inline void vInvSqrt(const int n, const T* vecIn, T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=T(1.0)/sqr<T>(vecIn[i]);
};

/// Slow implementation of vdSub and vsSub
template <typename T> inline void vSub(const int n, const T* vecIn1, 
      const T* vecIn2, T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=vecIn1[i]-vecIn2[i];
};
/// Slow implementation of vdInv and vsInv
template <typename T> inline void vInv(const int n, const T* vecIn, T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=1.0/vecIn[i];
};
/// Slow implementation of vdExp and vsExp
template <typename T> inline void vExp(const int n, const T* vecIn, T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=exp(vecIn[i]);
};
/// Slow implementation of vdAdd and vsAdd
template <typename T> inline void vAdd(const int n, const T* vecIn1, 
      const T* vecIn2, T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=vecIn1[i]+vecIn2[i];
};
/// Slow implementation of vdMul and vsMul
template <typename T> inline void vMul(const int n, const T* vecIn1, 
      const T* vecIn2, T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=vecIn1[i]*vecIn2[i];
};
/// Slow implementation of vdDiv and vsDiv
template <typename T> inline void vDiv(const int n, const T* vecIn1, 
      const T* vecIn2, T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=vecIn1[i]/vecIn2[i];
};
/// Slow implementation of vAbs
template <typename T> inline void vAbs(const int n, const T* vecIn, 
      T* vecOut) {
   for (int i = 0; i<n; ++i) vecOut[i]=abs<T>(vecIn[i]);
};

/// Slow implementation of cblas_idamin and cblas_isamin
template <typename T> inline int cblas_iamin(int n, T* X, int incX) {
   int imin=0;
   double min=fabs(X[0]);
   for (int j = 1; j<n; j+=incX) {
      double cur = fabs(X[j]);
      if (cur < min) {
         imin=j;
         min = cur;
      }
   }
   return imin;
}
#endif

#endif 
