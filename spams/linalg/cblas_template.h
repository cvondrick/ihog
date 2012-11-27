#ifndef MKL_TEMPLATE
#define MKL_TEMPLATE

//#include "cblas.h" dependency on cblas has been removed

/// a few static variables for lapack
static char low='l';
static char nonUnit='n';
static char upper='u';
static long info=0;
static char incr='I';
static char decr='D';

/// external functions
#ifdef HAVE_MKL
extern "C" {
#endif
   size_t cblas_idamin(const long n, const double* X, const long incX);
   size_t cblas_isamin(const long n, const float* X, const long incX);
#ifdef HAVE_MKL
};
#endif

extern "C" {
   void dsytrf_(char* uplo, long* n, double* a, long* lda, long* ipiv,
         double* work, long* lwork, long* info);
   void ssytrf_(char* uplo, long* n, float* a, long* lda, long* ipiv,
         float* work, long* lwork, long* info);
   void dsytri_(char* uplo, long* n, double* a, long* lda, long* ipiv,
         double* work, long* info);
   void ssytri_(char* uplo, long* n, float* a, long* lda, long* ipiv,
         float* work, long* info);
   void dtrtri_(char* uplo, char* diag, long* n, double * a, long* lda, 
         long* info);
   void strtri_(char* uplo, char* diag, long* n, float * a, long* lda, 
         long* info);
   void dlasrt_(char* id, const long* n, double *d, long* info);
   void slasrt_(char* id, const long* n, float*d, long* info);
   void dlasrt2_(char* id, const long* n, double *d, long* key, long* info);
   void slasrt2_(char* id, const long* n, float*d, long* key, long* info);
};

#ifdef HAVE_MKL
extern "C" {
   void vdSqr(const long n, const double* vecIn, double* vecOut);
   void vsSqr(const long n, const float* vecIn, float* vecOut);
   void vdSqrt(const long n, const double* vecIn, double* vecOut);
   void vsSqrt(const long n, const float* vecIn, float* vecOut);
   void vdInvSqrt(const long n, const double* vecIn, double* vecOut);
   void vsInvSqrt(const long n, const float* vecIn, float* vecOut);
   void vdSub(const long n, const double* vecIn, const double* vecIn2, double* vecOut);
   void vsSub(const long n, const float* vecIn, const float* vecIn2, float* vecOut);
   void vdDiv(const long n, const double* vecIn, const double* vecIn2, double* vecOut);
   void vsDiv(const long n, const float* vecIn, const float* vecIn2, float* vecOut);
   void vdExp(const long n, const double* vecIn, double* vecOut);
   void vsExp(const long n, const float* vecIn, float* vecOut);
   void vdInv(const long n, const double* vecIn, double* vecOut);
   void vsInv(const long n, const float* vecIn, float* vecOut);
   void vdAdd(const long n, const double* vecIn, const double* vecIn2, double* vecOut);
   void vsAdd(const long n, const float* vecIn, const float* vecIn2, float* vecOut);
   void vdMul(const long n, const double* vecIn, const double* vecIn2, double* vecOut);
   void vsMul(const long n, const float* vecIn, const float* vecIn2, float* vecOut);
   void vdAbs(const long n, const double* vecIn, double* vecOut);
   void vsAbs(const long n, const float* vecIn, float* vecOut);
}
#endif


// Interfaces to a few BLAS function, Level 1
/// interface to cblas_*nrm2
template <typename T> T cblas_nrm2(const long n, const T* X, const long incX);
/// interface to cblas_*copy
template <typename T> void cblas_copy(const long n, const T* X, const long incX, 
      T* Y, const long incY);
/// interface to cblas_*axpy
template <typename T> void cblas_axpy(const long n, const T a, const T* X, 
      const long incX, T* Y, const long incY);
/// interface to cblas_*scal
template <typename T> void cblas_scal(const long n, const T a, T* X, 
      const long incX);
/// interface to cblas_*asum
template <typename T> T cblas_asum(const long n, const T* X, const long incX);
/// interface to cblas_*adot
template <typename T> T cblas_dot(const long n, const T* X, const long incX, 
      const T* Y,const long incY);
/// interface to cblas_i*amin
template <typename T> long cblas_iamin(const long n, const T* X, const long incX);
/// interface to cblas_i*amax
template <typename T> long cblas_iamax(const long n, const T* X, const long incX);

// Interfaces to a few BLAS function, Level 2

/// interface to cblas_*gemv
template <typename T> void cblas_gemv(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const long M, 
      const long N, const T alpha, const T *A, const long lda, const T *X, 
      const long incX, const T beta,T *Y,  const long incY);
/// interface to cblas_*trmv
template <typename T> void inline cblas_trmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const long N,
      const T *A, const long lda, T *X, const long incX);
/// interface to cblas_*syr
template <typename T> void inline cblas_syr(const CBLAS_ORDER order, 
      const  CBLAS_UPLO Uplo, const long N, const T alpha, 
      const T *X, const long incX, T *A, const long lda);

/// interface to cblas_*symv
template <typename T> inline void cblas_symv(const CBLAS_ORDER order,
      const CBLAS_UPLO Uplo, const long N, 
      const T alpha, const T *A, const long lda, const T *X, 
      const long incX, const T beta,T *Y,  const long incY);


// Interfaces to a few BLAS function, Level 3
/// interface to cblas_*gemm
template <typename T> void cblas_gemm(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const long M, const long N, const long K, const T alpha, 
      const T *A, const long lda, const T *B, const long ldb,
      const T beta, T *C, const long ldc);
/// interface to cblas_*syrk
template <typename T> void cblas_syrk(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const long N, const long K,
      const T alpha, const T *A, const long lda,
      const T beta, T*C, const long ldc);
/// interface to cblas_*ger
template <typename T> void cblas_ger(const CBLAS_ORDER order, 
      const long M, const long N, const T alpha, const T *X, const long incX,
      const T* Y, const long incY, T*A, const long lda);
/// interface to cblas_*trmm
template <typename T> void cblas_trmm(const CBLAS_ORDER Order, 
      const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
      const long M, const long N, const T alpha, 
      const T*A, const long lda,T *B, const long ldb);

// Interfaces to a few functions from the Intel Vector Mathematical Library
/// interface to v*Sqr
template <typename T> void vSqrt(const long n, const T* vecIn, T* vecOut);
/// interface to v*Sqr
template <typename T> void vInvSqrt(const long n, const T* vecIn, T* vecOut);
/// interface to v*Sqr
template <typename T> void vSqr(const long n, const T* vecIn, T* vecOut);
/// interface to v*Sub
template <typename T> void vSub(const long n, const T* vecIn, const T* vecIn2, T* vecOut);
/// interface to v*Div
template <typename T> void vDiv(const long n, const T* vecIn, const T* vecIn2, T* vecOut);
/// interface to v*Exp
template <typename T> void vExp(const long n, const T* vecIn, T* vecOut);
/// interface to v*Inv
template <typename T> void vInv(const long n, const T* vecIn, T* vecOut);
/// interface to v*Add
template <typename T> void vAdd(const long n, const T* vecIn, const T* vecIn2, T* vecOut);
/// interface to v*Mul
template <typename T> void vMul(const long n, const T* vecIn, const T* vecIn2, T* vecOut);
/// interface to v*Abs
template <typename T> void vAbs(const long n, const T* vecIn, T* vecOut);

// Interfaces to a few LAPACK functions
/// interface to *trtri
template <typename T> void trtri(char& uplo, char& diag, 
      long& n, T * a, long& lda);
/// interface to *sytrf
template <typename T> void sytrf(char& uplo, long& n, T* a, long& lda, long* ipiv,
      T* work, long& lwork);
/// interface to *sytri
template <typename T> void sytri(char& uplo, long& n, T* a, long& lda, long* ipiv,
      T* work);
/// interaface to *lasrt
template <typename T> void lasrt(char& id, const long& n, T *d);
template <typename T> void lasrt2(char& id, const long& n, T *d, long* key);



/* ******************
 * Implementations
 * *****************/


// Implementations of the interfaces, BLAS Level 1
/// Implementation of the interface for cblas_dnrm2
template <> inline double cblas_nrm2<double>(const long n, const double* X, 
      const long incX) {
   return cblas_dnrm2(n,X,incX);
};
/// Implementation of the interface for cblas_snrm2
template <> inline float cblas_nrm2<float>(const long n, const float* X, 
      const long incX) {
   return cblas_snrm2(n,X,incX);
};
/// Implementation of the interface for cblas_dcopy
template <> inline void cblas_copy<double>(const long n, const double* X, 
      const long incX, double* Y, const long incY) {
   cblas_dcopy(n,X,incX,Y,incY);
};
/// Implementation of the interface for cblas_scopy
template <> inline void cblas_copy<float>(const long n, const float* X, const long incX, 
      float* Y, const long incY) {
   cblas_scopy(n,X,incX,Y,incY);
};
/// Implementation of the interface for cblas_scopy
template <> inline void cblas_copy<long>(const long n, const long* X, const long incX, 
      long* Y, const long incY) {
   for (long i = 0; i<n; ++i)
      Y[incY*i]=X[incX*i];
};
/// Implementation of the interface for cblas_scopy
template <> inline void cblas_copy<bool>(const long n, const bool* X, const long incX, 
      bool* Y, const long incY) {
   for (long i = 0; i<n; ++i)
      Y[incY*i]=X[incX*i];
};

/// Implementation of the interface for cblas_daxpy
template <> inline void cblas_axpy<double>(const long n, const double a, const double* X, 
      const long incX, double* Y, const long incY) {
   cblas_daxpy(n,a,X,incX,Y,incY);
};
/// Implementation of the interface for cblas_saxpy
template <> inline void cblas_axpy<float>(const long n, const float a, const float* X,
      const long incX, float* Y, const long incY) {
   cblas_saxpy(n,a,X,incX,Y,incY);
};

/// Implementation of the interface for cblas_saxpy
template <> inline void cblas_axpy<long>(const long n, const long a, const long* X,
      const long incX, long* Y, const long incY) {
   for (long i = 0; i<n; ++i)
      Y[i] += a*X[i];
};

/// Implementation of the interface for cblas_saxpy
template <> inline void cblas_axpy<bool>(const long n, const bool a, const bool* X,
      const long incX, bool* Y, const long incY) {
   for (long i = 0; i<n; ++i)
      Y[i] = a*X[i];
};


/// Implementation of the interface for cblas_dscal
template <> inline void cblas_scal<double>(const long n, const double a, double* X,
      const long incX) {
   cblas_dscal(n,a,X,incX);
};
/// Implementation of the interface for cblas_sscal
template <> inline void cblas_scal<float>(const long n, const float a, float* X, 
      const long incX) {
   cblas_sscal(n,a,X,incX);
};
/// Implementation of the interface for cblas_sscal
template <> inline void cblas_scal<long>(const long n, const long a, long* X, 
      const long incX) {
   for (long i = 0; i<n; ++i) X[i*incX]*=a;
};
/// Implementation of the interface for cblas_sscal
template <> inline void cblas_scal<bool>(const long n, const bool a, bool* X, 
      const long incX) {
   /// not implemented
};

/// Implementation of the interface for cblas_dasum
template <> inline double cblas_asum<double>(const long n, const double* X, const long incX) {
   return cblas_dasum(n,X,incX);
};
/// Implementation of the interface for cblas_sasum
template <> inline float cblas_asum<float>(const long n, const float* X, const long incX) {
   return cblas_sasum(n,X,incX);
};
/// Implementation of the interface for cblas_ddot
template <> inline double cblas_dot<double>(const long n, const double* X,
      const long incX, const double* Y,const long incY) {
   return cblas_ddot(n,X,incX,Y,incY);
};
/// Implementation of the interface for cblas_sdot
template <> inline float cblas_dot<float>(const long n, const float* X,
      const long incX, const float* Y,const long incY) {
   return cblas_sdot(n,X,incX,Y,incY);
};
template <> inline long cblas_dot<long>(const long n, const long* X,
      const long incX, const long* Y,const long incY) {
   long total=0;
   long i,j;
   j=0;
   for (i = 0; i<n; ++i) {
      total+=X[i*incX]*Y[j];
      j+=incY;
   }
   return total;
};
/// Implementation of the interface for cblas_sdot
template <> inline bool cblas_dot<bool>(const long n, const bool* X,
      const long incX, const bool* Y,const long incY) {
   /// not implemented
   return true;
};

// Implementations of the interfaces, BLAS Level 2
///  Implementation of the interface for cblas_dgemv
template <> inline void cblas_gemv<double>(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const long M, const long N,
      const double alpha, const double *A, const long lda,
      const double *X, const long incX, const double beta,
      double *Y, const long incY) {
   cblas_dgemv(order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
};
///  Implementation of the interface for cblas_sgemv
template <> inline void cblas_gemv<float>(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const long M, const long N,
      const float alpha, const float *A, const long lda,
      const float *X, const long incX, const float beta,
      float *Y, const long incY) {
   cblas_sgemv(order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
};
///  Implementation of the interface for cblas_sgemv
template <> inline void cblas_gemv<long>(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const long M, const long N,
      const long alpha, const long *A, const long lda,
      const long *X, const long incX, const long beta,
      long *Y, const long incY) {
   ///  not implemented
};
///  Implementation of the interface for cblas_sgemv
template <> inline void cblas_gemv<bool>(const CBLAS_ORDER order,
      const CBLAS_TRANSPOSE TransA, const long M, const long N,
      const bool alpha, const bool *A, const long lda,
      const bool *X, const long incX, const bool beta,
      bool *Y, const long incY) {
   /// not implemented
};

///  Implementation of the interface for cblas_dger
template <> inline void cblas_ger<double>(const CBLAS_ORDER order, 
      const long M, const long N, const double alpha, const double *X, const long incX,
      const double* Y, const long incY, double *A, const long lda) {
   cblas_dger(order,M,N,alpha,X,incX,Y,incY,A,lda);
};
///  Implementation of the interface for cblas_sger
template <> inline void cblas_ger<float>(const CBLAS_ORDER order, 
      const long M, const long N, const float alpha, const float *X, const long incX,
      const float* Y, const long incY, float *A, const long lda) {
   cblas_sger(order,M,N,alpha,X,incX,Y,incY,A,lda);
};
///  Implementation of the interface for cblas_dtrmv
template <> inline void cblas_trmv<double>(const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const long N,
      const double *A, const long lda, double *X, const long incX) {
   cblas_dtrmv(order,Uplo,TransA,Diag,N,A,lda,X,incX);
};
///  Implementation of the interface for cblas_strmv
template <> inline void cblas_trmv<float>(const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const long N,
      const float *A, const long lda, float *X, const long incX) {
   cblas_strmv(order,Uplo,TransA,Diag,N,A,lda,X,incX);
};
/// Implementation of cblas_dsyr
template <> inline void cblas_syr(const CBLAS_ORDER order, 
      const  CBLAS_UPLO Uplo,
      const long N, const double alpha, const double*X,
      const long incX, double *A, const long lda) {
   cblas_dsyr(order,Uplo,N,alpha,X,incX,A,lda);
};
/// Implementation of cblas_ssyr
template <> inline void cblas_syr(const CBLAS_ORDER order, 
      const  CBLAS_UPLO Uplo,
      const long N, const float alpha, const float*X,
      const long incX, float *A, const long lda) {
   cblas_ssyr(order,Uplo,N,alpha,X,incX,A,lda);
};
/// Implementation of cblas_ssymv
template <> inline void cblas_symv(const CBLAS_ORDER order,
      const CBLAS_UPLO Uplo, const long N, 
      const float alpha, const float *A, const long lda, const float *X, 
      const long incX, const float beta,float *Y,  const long incY) {
   cblas_ssymv(order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);
}
/// Implementation of cblas_dsymv
template <> inline void cblas_symv(const CBLAS_ORDER order,
      const CBLAS_UPLO Uplo, const long N, 
      const double alpha, const double *A, const long lda, const double *X, 
      const long incX, const double beta,double *Y,  const long incY) {
   cblas_dsymv(order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);
}


// Implementations of the interfaces, BLAS Level 3
///  Implementation of the interface for cblas_dgemm
template <> inline void cblas_gemm<double>(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const long M, const long N, const long K, const double alpha, 
      const double *A, const long lda, const double *B, const long ldb,
      const double beta, double *C, const long ldc) {
   cblas_dgemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
};
///  Implementation of the interface for cblas_sgemm
template <> inline void cblas_gemm<float>(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const long M, const long N, const long K, const float alpha, 
      const float *A, const long lda, const float *B, const long ldb,
      const float beta, float *C, const long ldc) {
   cblas_sgemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
};
template <> inline void cblas_gemm<long>(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const long M, const long N, const long K, const long alpha, 
      const long *A, const long lda, const long *B, const long ldb,
      const long beta, long *C, const long ldc) {
   /// not implemented
};
///  Implementation of the interface for cblas_sgemm
template <> inline void cblas_gemm<bool>(const CBLAS_ORDER Order, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
      const long M, const long N, const long K, const bool alpha, 
      const bool *A, const long lda, const bool *B, const long ldb,
      const bool beta, bool *C, const long ldc) {
   /// not implemented
};

///  Implementation of the interface for cblas_dsyrk
template <> inline void cblas_syrk<double>(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const long N, const long K,
      const double alpha, const double *A, const long lda,
      const double beta, double *C, const long ldc) {
   cblas_dsyrk(Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);
};
///  Implementation of the interface for cblas_ssyrk
template <> inline void cblas_syrk<float>(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const long N, const long K,
      const float alpha, const float *A, const long lda,
      const float beta, float *C, const long ldc) {
   cblas_ssyrk(Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);
};
///  Implementation of the interface for cblas_ssyrk
template <> inline void cblas_syrk<long>(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const long N, const long K,
      const long alpha, const long *A, const long lda,
      const long beta, long *C, const long ldc) {
   /// not implemented
};
///  Implementation of the interface for cblas_ssyrk
template <> inline void cblas_syrk<bool>(const CBLAS_ORDER Order, 
      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const long N, const long K,
      const bool alpha, const bool *A, const long lda,
      const bool beta, bool *C, const long ldc) {
   /// not implemented
};

///  Implementation of the interface for cblas_dtrmm
template <> inline void cblas_trmm<double>(const CBLAS_ORDER Order, 
      const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
      const long M, const long N, const double alpha, 
      const double *A, const long lda,double *B, const long ldb) {
   cblas_dtrmm(Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);
};
///  Implementation of the interface for cblas_strmm
template <> inline void cblas_trmm<float>(const CBLAS_ORDER Order, 
      const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, 
      const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
      const long M, const long N, const float alpha, 
      const float *A, const long lda,float *B, const long ldb) {
   cblas_strmm(Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);
};
///  Implementation of the interface for cblas_idamax
template <> inline long cblas_iamax<double>(const long n, const double* X,
      const long incX) {
   return cblas_idamax(n,X,incX);
};
///  Implementation of the interface for cblas_isamax
template <> inline long cblas_iamax<float>(const long n, const float* X, 
      const long incX) {
   return cblas_isamax(n,X,incX);
};

// Implementations of the interfaces, LAPACK
/// Implemenation of the interface for dtrtri
template <> inline void trtri<double>(char& uplo, char& diag, 
      long& n, double * a, long& lda) {
   dtrtri_(&uplo,&diag,&n,a,&lda,&info);
};
/// Implemenation of the interface for strtri
template <> inline void trtri<float>(char& uplo, char& diag, 
      long& n, float* a, long& lda) {
   strtri_(&uplo,&diag,&n,a,&lda,&info);
};
/// Implemenation of the interface for dsytrf
template <> void inline sytrf<double>(char& uplo, long& n, double* a, 
      long& lda, long* ipiv, double* work, long& lwork) {
   dsytrf_(&uplo,&n,a,&lda,ipiv,work,&lwork,&info);
};
/// Implemenation of the interface for ssytrf
template <> inline void sytrf<float>(char& uplo, long& n, float* a, 
      long& lda, long* ipiv, float* work, long& lwork) {
   ssytrf_(&uplo,&n,a,&lda,ipiv,work,&lwork,&info);
};
/// Implemenation of the interface for dsytri
template <> inline void sytri<double>(char& uplo, long& n, double* a, 
      long& lda, long* ipiv, double* work) {
   dsytri_(&uplo,&n,a,&lda,ipiv,work,&info);
};
/// Implemenation of the interface for ssytri
template <> inline void sytri<float>(char& uplo, long& n, float* a, 
      long& lda, long* ipiv, float* work) {
   ssytri_(&uplo,&n,a,&lda,ipiv,work,&info);
};
/// interaface to *lasrt
template <> inline void lasrt(char& id, const long& n, double *d) {
   dlasrt_(&id,const_cast<long*>(&n),d,&info);
};
/// interaface to *lasrt
template <> inline void lasrt(char& id, const long& n, float *d) {
   slasrt_(&id,const_cast<long*>(&n),d,&info);
};
template <> inline void lasrt2(char& id, const long& n, double *d,long* key) {
   dlasrt2_(&id,const_cast<long*>(&n),d,key,&info);
};
/// interaface to *lasrt
template <> inline void lasrt2(char& id, const long& n, float *d, long* key) {
   slasrt2_(&id,const_cast<long*>(&n),d,key,&info);
};


/// If the MKL is not present, a slow implementation is used instead.
#ifdef HAVE_MKL 
/// Implemenation of the interface for vdSqr
template <> inline void vSqr<double>(const long n, const double* vecIn, 
      double* vecOut) {
   vdSqr(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsSqr
template <> inline void vSqr<float>(const long n, const float* vecIn, 
      float* vecOut) {
   vsSqr(n,vecIn,vecOut);
};
template <> inline void vSqrt<double>(const long n, const double* vecIn, 
      double* vecOut) {
   vdSqrt(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsSqr
template <> inline void vSqrt<float>(const long n, const float* vecIn, 
      float* vecOut) {
   vsSqrt(n,vecIn,vecOut);
};
template <> inline void vInvSqrt<double>(const long n, const double* vecIn, 
      double* vecOut) {
   vdInvSqrt(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsSqr
template <> inline void vInvSqrt<float>(const long n, const float* vecIn, 
      float* vecOut) {
   vsInvSqrt(n,vecIn,vecOut);
};

/// Implemenation of the interface for vdSub
template <> inline void vSub<double>(const long n, const double* vecIn, 
      const double* vecIn2, double* vecOut) {
   vdSub(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsSub
template <> inline void vSub<float>(const long n, const float* vecIn, 
      const float* vecIn2, float* vecOut) {
   vsSub(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vdDiv
template <> inline void vDiv<double>(const long n, const double* vecIn, 
      const double* vecIn2, double* vecOut) {
   vdDiv(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsDiv
template <> inline void vDiv<float>(const long n, const float* vecIn, 
      const float* vecIn2, float* vecOut) {
   vsDiv(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vdExp
template <> inline void vExp<double>(const long n, const double* vecIn, 
      double* vecOut) {
   vdExp(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsExp
template <> inline void vExp<float>(const long n, const float* vecIn, 
      float* vecOut) {
   vsExp(n,vecIn,vecOut);
};
/// Implemenation of the interface for vdInv
template <> inline void vInv<double>(const long n, const double* vecIn, 
      double* vecOut) {
   vdInv(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsInv
template <> inline void vInv<float>(const long n, const float* vecIn, 
      float* vecOut) {
   vsInv(n,vecIn,vecOut);
};
/// Implemenation of the interface for vdAdd
template <> inline void vAdd<double>(const long n, const double* vecIn, 
      const double* vecIn2, double* vecOut) {
   vdAdd(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsAdd
template <> inline void vAdd<float>(const long n, const float* vecIn, 
      const float* vecIn2, float* vecOut) {
   vsAdd(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vdMul
template <> inline void vMul<double>(const long n, const double* vecIn, 
      const double* vecIn2, double* vecOut) {
   vdMul(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsMul
template <> inline void vMul<float>(const long n, const float* vecIn, 
      const float* vecIn2, float* vecOut) {
   vsMul(n,vecIn,vecIn2,vecOut);
};

/// interface to vdAbs
template <> inline void vAbs(const long n, const double* vecIn, 
      double* vecOut) {
   vdAbs(n,vecIn,vecOut);
};
/// interface to vdAbs
template <> inline void vAbs(const long n, const float* vecIn, 
      float* vecOut) {
   vsAbs(n,vecIn,vecOut);
};


/// Implemenation of the interface of the non-offical BLAS, Level 1 function 
/// cblas_idamin
template <> inline long cblas_iamin<double>(const long n, const double* X,
      const long incX) {
   return (int) cblas_idamin(n,X,incX);
};
/// Implemenation of the interface of the non-offical BLAS, Level 1 function 
/// cblas_isamin
template <> inline long cblas_iamin<float>(const long n, const float* X, 
      const long incX) {
   return (int) cblas_isamin(n,X,incX);
};
/// slow alternative implementation of some MKL function
#else
/// Slow implementation of vdSqr and vsSqr
template <typename T> inline void vSqr(const long n, const T* vecIn, T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=vecIn[i]*vecIn[i];
};
template <typename T> inline void vSqrt(const long n, const T* vecIn, T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=sqr<T>(vecIn[i]);
};
template <typename T> inline void vInvSqrt(const long n, const T* vecIn, T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=T(1.0)/sqr<T>(vecIn[i]);
};

/// Slow implementation of vdSub and vsSub
template <typename T> inline void vSub(const long n, const T* vecIn1, 
      const T* vecIn2, T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=vecIn1[i]-vecIn2[i];
};
/// Slow implementation of vdInv and vsInv
template <typename T> inline void vInv(const long n, const T* vecIn, T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=1.0/vecIn[i];
};
/// Slow implementation of vdExp and vsExp
template <typename T> inline void vExp(const long n, const T* vecIn, T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=exp(vecIn[i]);
};
/// Slow implementation of vdAdd and vsAdd
template <typename T> inline void vAdd(const long n, const T* vecIn1, 
      const T* vecIn2, T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=vecIn1[i]+vecIn2[i];
};
/// Slow implementation of vdMul and vsMul
template <typename T> inline void vMul(const long n, const T* vecIn1, 
      const T* vecIn2, T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=vecIn1[i]*vecIn2[i];
};
/// Slow implementation of vdDiv and vsDiv
template <typename T> inline void vDiv(const long n, const T* vecIn1, 
      const T* vecIn2, T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=vecIn1[i]/vecIn2[i];
};
/// Slow implementation of vAbs
template <typename T> inline void vAbs(const long n, const T* vecIn, 
      T* vecOut) {
   for (long i = 0; i<n; ++i) vecOut[i]=abs<T>(vecIn[i]);
};

/// Slow implementation of cblas_idamin and cblas_isamin
template <typename T> inline long cblas_iamin(long n, T* X, long incX) {
   long imin=0;
   double min=fabs(X[0]);
   for (long j = 1; j<n; j+=incX) {
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
