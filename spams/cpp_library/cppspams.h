#ifndef CPPSPAMS_H
#define CPPSPAMS_H

#include "spams.h"

/* from decomp */
/**
   cppOMP is an efficient implementation of the
*     Orthogonal Matching Pursuit algorithm. It is optimized
*     for solving a large number of small or medium-sized 
*     decomposition problem (and not for a single large one).
*     It first computes the Gram matrix D'D and then perform
*     a Cholesky-based OMP of the input signals in parallel.
*     X=[x^1,...,x^n] is a matrix of signals, and it returns
*     a matrix A=[alpha^1,...,alpha^n] of coefficients.
*     it addresses for all columns x of X, 
*         min_{alpha} ||alpha||_0  s.t  ||x-Dalpha||_2^2 <= eps
*         or
*         min_{alpha} ||x-Dalpha||_2^2  s.t. ||alpha||_0 <= L
*         or
*         min_{alpha} 0.5||x-Dalpha||_2^2 + lambda||alpha||_0 
*         
*
   @param X: double m x n matrix   (input signals)
            m is the signal size
            n is the number of signals to decompose
   @param D: double m x p matrix   (dictionary)
             p is the number of elements in the dictionary
             All the columns of D should have unit-norm !
   @param L (optional, maximum number of elements in each decomposition, 
               min(m,p) by default)
   @param eps (optional, threshold on the squared l2-norm of the residual,
              0 by default
   @param lambda (optional, penalty parameter, 0 by default
   @param numThreads (optional, number of threads for exploiting
                    multi-core / multi-cpus. By default, it takes the value -1,
                    which automatically selects all the available CPUs/cores).
  @param vecL : false if L is a pointer to a scalar.
  @param vecEps : false if eps is a pointer to a scalar.
  @param vecLambda : false if lambda is a pointer to a scalar.
  @param[out] A double sparse p x n matrix (output coefficients)
  @param[out] path : if not a NULL pointer, double dense p x L matrix (regularization path of the first signal)
 **/
template <typename T>
void cppOMP(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha, 
	 const int *L, const T* eps, const T* lambda, const bool vecL = false,
	 const bool vecEps = false, const bool vecLambda=false, const int numThreads=-1,
	 Matrix<T>* path = NULL)
{
  omp(X,D, spalpha, L, eps, lambda, vecL,vecEps, vecLambda, numThreads,path);
}
/* **************** */
/**
* cppLasso is an efficient implementation of the
*     homotopy-LARS algorithm for solving the Lasso. 
*     
*     if the function is called this way cppLasso(X,D,param),
*     it aims at addressing the following problems
*     for all columns x of X, it computes one column alpha of A
*     that solves
*       1) when param.mode=0
*         min_{alpha} ||x-Dalpha||_2^2 s.t. ||alpha||_1 <= lambda
*       2) when param.mode=1
*         min_{alpha} ||alpha||_1 s.t. ||x-Dalpha||_2^2 <= lambda
*       3) when param.mode=2
*         min_{alpha} 0.5||x-Dalpha||_2^2 + lambda||alpha||_1 +0.5 lambda2||alpha||_2^2
*
*     if the function is called this way cppLasso(X,Q,q,param),
*     it solves the above optimisation problem, when Q=D'D and q=D'x.
*
*     Possibly, when param.pos=true, it solves the previous problems
*     with positivity constraints on the vectors alpha
*
* @param X:  double m x n matrix   (input signals)
*               m is the signal size
*               n is the number of signals to decompose
* @param D:  double m x p matrix   (dictionary)
*               p is the number of elements in the dictionary
* @param lambda  (parameter)
* @param lambda2  (optional parameter for solving the Elastic-Net)
*                for mode=0 and mode=1, it adds a ridge on the Gram Matrix
* @param L (optional), maximum number of steps of the homotopy algorithm (can
*          be used as a stopping criterion)
* @param pos (optional, adds non-negativity constraints on the
*   coefficients, false by default)
* @param mode (see above, by default: 2)
* @param numThreads (optional, number of threads for exploiting
*   multi-core / multi-cpus. By default, it takes the value -1,
*   which automatically selects all the available CPUs/cores).
* @param cholesky (optional, default false),  choose between Cholesky 
*   implementation or one based on the matrix inversion Lemma
* @param ols (optional, default false), perform an orthogonal projection
*   before returning the solution.
* @param max_length_path (optional) maximum length of the path, by default 4*p
*
* @param[out] path: optional,  returns the regularisation path for the first signal
* @return (A) double sparse p x n matrix (output coefficients)
*
* Note: this function admits a few experimental usages, which have not
*     been extensively tested:
*         - single precision setting (even though the output alpha is double 
*           precision)
*
**/
template <typename T>
SpMatrix<T> *cppLasso(Matrix<T> &X, Matrix<T> &D,Matrix<T> **path,bool return_reg_path,
		      int L, const T constraint, const T lambda2 = 0., constraint_type mode= PENALTY,
      const bool pos= false, const bool ols= false, const int numThreads= -1,
		    int max_length_path= -1,const bool verbose= false, bool cholevsky= false) 
{
  return _lassoD(&X,&D,path,return_reg_path,L,constraint,lambda2,mode,pos,ols, numThreads,
		 max_length_path, verbose, cholevsky);
}
 
template <typename T>
SpMatrix<T> *cppLasso(Matrix<T> &X, Matrix<T> &Q, Matrix<T> &q,Matrix<T> **path,bool return_reg_path,
		      int L, const T constraint, const T lambda2 = 0., constraint_type mode= PENALTY,
		      const bool pos= false, const bool ols= false, const int numThreads= -1,
		      int max_length_path= -1,const bool verbose= false, bool cholevsky= false) 
{
  return _lassoD(&X,&Q,&q,path,return_reg_path,L,constraint,lambda2,mode,pos,ols, numThreads,
		 max_length_path, verbose, cholevsky);
}
 
#endif /* SPAMS_H */
