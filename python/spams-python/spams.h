#ifndef SPAMS_H
#define SPAMS_H

#ifdef WIN32
#define random rand
#define srandom srand
#ifdef CYGWIN
// the gcc compiler of cygwin has some defines that conflicts with spams,
// But we preferably use mingw
#include <fstream>
#include <list>
#include <vector>
#include <math.h>
#include <iostream>
typedef unsigned int uint32_t;
typedef int int32_t;

#undef _X
#undef _L
#undef _N
#undef _B
#endif
#endif

#include "dicts.h"
#include "fista.h"
#include "decomp.h"
#include "linalg.h"
#include "cblas_alt_template.h"
#include<iostream>
/* from linalg */

template<typename T> void _sort(Vector<T> *v,bool mode) throw(const char *){
  v->sort(mode);
}


template<typename T> void _AAt(SpMatrix<T> *A,Matrix<T> *B) throw(const char *) {
  
  if(A->m() != B->m() || B->m() != B->n())
    throw("AAt: incompatible dimensions of result matrix");
  A->AAt((Matrix<T>&)(*B));
}

template<typename T> void _XAt(SpMatrix<T> *A,Matrix<T> *X,Matrix<T> *XAt) throw(const char *) {
  if(X->n() != A->n() || X->m() != XAt->m() || A->m() != XAt->n())
    throw("XAt: incompatible dimensions of result matrix");
  A->XAt((Matrix<T>&)(*X),(Matrix<T>&)(*XAt));
}

template<typename T> inline void _mult(Matrix<T> *X,Matrix<T> *Y,Matrix<T> *XY,const bool transX, const bool transY,
      const T a, const T b) throw(const char *) {
  int xrows, xcols, yrows, ycols;
  if(transX) {
    xrows = X->n();
    xcols = X->m();
  } else {
    xrows = X->m();
    xcols = X->n();
  }
  if(transY) {
    yrows = Y->n();
    ycols = Y->m();
  } else {
    yrows = Y->m();
    ycols = Y->n();
  }
  if(xcols != yrows || xrows != XY->m() || ycols != XY->n()) {
    throw("mult: incompatible matrices");
  }
  X->mult((Matrix<T>&)(*Y),(Matrix<T>&)(*XY),transX,transY,a,b);
}
  
template<typename T> void _applyBayerPattern(Vector<T> *v,int offset) throw(const char *){
  v->applyBayerPattern(offset);
}

template<typename T> void _conjugateGradient(Matrix<T> *A,Vector<T> *b,Vector<T> *x,const T tol,const int itermax) throw(const char *){
  if(A->n() != x->n() || A->m() != b->n())
    throw("conjugateGradient: incompatible matrix and vectore sizes");
  A->conjugateGradient((Vector<T> &)(*b),(Vector<T> &)(*x),tol,itermax);
}

template<typename T> void _invSym(Matrix<T> *A) {
  A->invSym();
}

template<typename T> void _normalize(Matrix<T> *A) {
  A->normalize();
}

/* end linalg */

/* from decomp */

template <typename T> inline void _sparseProject(Matrix<T> *U,Matrix<T> *V,
      const T thrs,   const int mode, const T lambda1,
      const T lambda2, const T lambda3, const bool pos,
      const int numThreads) throw(const char *) {
  if(U->m() != V->m() || U->n() != V->n())
    throw("sparseProject: incompatible matrices");
  U->sparseProject((Matrix<T>&)(*V),thrs,mode,lambda1,lambda2,lambda3,pos,numThreads);
}

template <typename T>
SpMatrix<T> *_lassoD(Matrix<T> *X, Matrix<T> *D,Matrix<T> **path,bool return_reg_path,
		    int L, const T constraint, const T lambda2, constraint_type mode,
      const bool pos, const bool ols, const int numThreads,
		    int max_length_path,const bool verbose, bool cholevsky) 
throw(const char *) 
{
  SpMatrix<T> *alpha = new SpMatrix<T>();
  int n = X->m();
  int M = X->n();
  int nD = D->m();
  int K = D->n();
  if (n != nD)
    throw("lasso : incompatible matrix dimensions");
  if(L < 0) L = K;
  if(max_length_path < 0) max_length_path = 4 * L;
  if (L> n && !(mode == PENALTY && isZero(constraint) && !pos && lambda2 > 0)) {
    if (verbose)
      printf("L is changed to %d\n",n);
    L=n;
  }
  if (L > K) {
    if (verbose)
      printf("L is changed to %d\n",K);
    L=K;
  }
  if(return_reg_path)
    *path = new Matrix<T>(K,max_length_path);
  else
    *path = NULL;
  if(ols) cholevsky = ols;
  if (cholevsky) {
    lasso((Matrix<T> &)(*X),(Matrix<T> &)(*D),(SpMatrix<T> &)(*alpha),L,constraint,lambda2,mode,pos,ols,numThreads,*path,max_length_path);
  } else {
    lasso2((Matrix<T> &)(*X),(Matrix<T> &)(*D),(SpMatrix<T> &)(*alpha),L,constraint,lambda2,mode,pos,numThreads,*path,max_length_path);
  }
  return alpha;
}

template <typename T>
SpMatrix<T> *_lassoQq(Matrix<T> *X, Matrix<T> *Q, Matrix<T> *q,Matrix<T> **path,bool return_reg_path,
		      int L, const T constraint, const T lambda2, constraint_type mode,
		      const bool pos, const bool ols, const int numThreads,
		      int max_length_path,const bool verbose, bool cholevsky) 
throw(const char *) 
// lambda2 is ignored
{
  SpMatrix<T> *alpha = new SpMatrix<T>();
  int n = X->m();
  int M = X->n();
  int K1 = Q->m();
  int K2 = Q->n();
  if(K1 != K2)
    throw("lasso : Q must be square");
  int K = K1;
  int K3 = q->m();
  int M2 = q->n();
  if (K1 != K3 || M != M2)
    throw("lasso : incompatible matrix dimensions");

  if(L < 0) L = K1;
  if(max_length_path < 0) max_length_path = 4 * L;
  if (L> n && !(mode == PENALTY && isZero(constraint) && !pos && lambda2 > 0)) {
    if (verbose)
      printf("L is changed to %d\n",n);
    L=n;
  }
  if (L > K) {
    if (verbose)
      printf("L is changed to %d\n",K);
    L=K;
  }
  if(return_reg_path)
    *path = new Matrix<T>(K,max_length_path);
  else
    *path = NULL;
  if(ols) cholevsky = ols;
  if (cholevsky)
    lasso((Data<T> &)(*X),(AbstractMatrix<T> &)(*Q),(AbstractMatrix<T> &)(*q),(SpMatrix<T> &)(*alpha),L,constraint,mode,pos,ols,numThreads,*path,max_length_path);
  else
    lasso2((Data<T> &)(*X),(AbstractMatrix<T> &)(*Q),(AbstractMatrix<T> &)(*q),(SpMatrix<T> &)(*alpha),L,constraint,mode,pos,numThreads,*path,max_length_path);
  return alpha;
}

template <typename T>
SpMatrix<T> *_lassoMask(Matrix<T> *X, Matrix<T> *D,Matrix<bool> *B,
		    int L, const T constraint, const T lambda2, constraint_type mode,
			const bool pos, const int numThreads,bool verbose) 
throw(const char *) 
{
  SpMatrix<T> *alpha = new SpMatrix<T>();
  int n = X->m();
  int M = X->n();
  int nD = D->m();
  int K = D->n();
  if (n != nD)
    throw("lassoMask : incompatible matrix dimensions");
  if(L < 0) L = K;
  if (L> n && !(mode == PENALTY && isZero(constraint) && !pos && lambda2 > 0)) {
    if (verbose)
      printf("L is changed to %d\n",n);
    L=n;
  }
  if (L > K) {
    if (verbose)
      printf("L is changed to %d\n",K);
    L=K;
  }
  lasso_mask((Matrix<T> &)(*X),(Matrix<T> &)(*D),(SpMatrix<T> &)(*alpha),(Matrix<bool> &)(*B),L,constraint,lambda2,mode,pos,numThreads);
  return alpha;
}

template <typename T>
SpMatrix<T> *_lassoWeighted(Matrix<T> *X, Matrix<T> *D,Matrix<T> *W,
		    int L, const T constraint, constraint_type mode,
			const bool pos, const int numThreads,bool verbose) 
throw(const char *) 
{
  SpMatrix<T> *alpha = new SpMatrix<T>();
  int n = X->m();
  int M = X->n();
  int nD = D->m();
  int K = D->n();
  if (n != nD)
    throw("lassoWeighted : incompatible matrix dimensions");
  if(L < 0) L = K;
  if (L> n ) {
    if (verbose)
      printf("L is changed to %d\n",n);
    L=n;
  }
  if (L > K) {
    if (verbose)
      printf("L is changed to %d\n",K);
    L=K;
  }
  int KK = W->m();
  int MM = W->n();
  if (K != KK || M != MM)
    throw("lassoWeighted : inconsistent dimensions of matrix W");

  lassoWeight((Matrix<T> &)(*X),(Matrix<T> &)(*D),(Matrix<T> &)(*W),(SpMatrix<T> &)(*alpha),L,constraint,mode,pos,numThreads);
  return alpha;
}

template <typename T>
SpMatrix<T> *_omp(Matrix<T> *X,Matrix<T> *D,Matrix<T> **path,bool return_reg_path,bool given_L,Vector<int>*L,bool given_eps,Vector<T>*eps,bool given_Lambda,Vector<T>*Lambda,const int numThreads) throw(const char *){
  SpMatrix<T> *alpha = new SpMatrix<T>();
    int n = X->m();
    int M = X->n();
    int nD = D->m();
    int K = D->n();
    if (n != nD)
      throw("omp : incompatible matrix dimensions");
    int sizeL = L->n();
    int sizeE = eps->n();
    int sizeLambda = Lambda->n();
    T *pE = eps->rawX();
    T *pLambda = Lambda->rawX();
    int *pL = L->rawX();
    bool vecL = false;
    bool vecEps = false;
    bool vecLambda = false;
    if (! given_L && ! given_eps && ! given_Lambda)
      throw("omp : You should either provide L, eps or lambda");
    int scalar_L = MIN(n,K);
    if(! given_L) 
      pL = &scalar_L;
    else if (sizeL > 1)
      vecL = true;
    if(! given_eps) {
      T scalar_eps = 0.;
      pE = &scalar_eps;
    } else if (sizeE > 1)
      vecEps = true;
    if(! given_Lambda) {
      T scalar_Lambda = 0.;
      pLambda = &scalar_Lambda;
    } else if(sizeLambda > 1)
      vecLambda = true;
    if(return_reg_path) {
      *path = new Matrix<T>(K,scalar_L);
      (*path)->setZeros();
      omp((Matrix<T> &)(*X),(Matrix<T> &)(*D),(SpMatrix<T> &)(*alpha),pL,pE,pLambda,vecL,vecEps,vecLambda,numThreads,*path);
    } else {
      *path = NULL;
      omp((Matrix<T> &)(*X),(Matrix<T> &)(*D),(SpMatrix<T> &)(*alpha),pL,pE,pLambda,vecL,vecEps,vecLambda,numThreads);
    }
    return alpha;
}

template <typename T>
SpMatrix<T> *_ompMask(Matrix<T> *X,Matrix<T> *D,Matrix<bool> *B,Matrix<T> **path,bool return_reg_path,bool given_L,Vector<int>*L,bool given_eps,Vector<T>*eps,bool given_Lambda,Vector<T>*Lambda,const int numThreads) throw(const char *){
  SpMatrix<T> *alpha = new SpMatrix<T>();
    int n = X->m();
    int M = X->n();
    int nD = D->m();
    int K = D->n();
    int nM = B->m();
    int mM = B->n();
    if (n != nD )
      throw("ompMask : incompatible matrix dimensions");
    if (nM != n || mM != M)
      throw("ompMask : Mash has non acceptable dimensions");
    int sizeL = L->n();
    int sizeE = eps->n();
    int sizeLambda = Lambda->n();
    T *pE = eps->rawX();
    T *pLambda = Lambda->rawX();
    int *pL = L->rawX();
    bool vecL = false;
    bool vecEps = false;
    bool vecLambda = false;
    if (! given_L && ! given_eps && ! given_Lambda)
      throw("omp : You should either provide L, eps or lambda");
    int scalar_L = MIN(n,K);
    if(! given_L) 
      pL = &scalar_L;
    else if (sizeL > 1)
      vecL = true;
    if(! given_eps) {
      T scalar_eps = 0.;
      pE = &scalar_eps;
    } else if (sizeE > 1)
      vecEps = true;
    if(! given_Lambda) {
      T scalar_Lambda = 0.;
      pLambda = &scalar_Lambda;
    } else if(sizeLambda > 1)
      vecLambda = true;
    if(return_reg_path) {
      *path = new Matrix<T>(K,scalar_L);
      (*path)->setZeros();
      omp_mask((Matrix<T> &)(*X),(Matrix<T> &)(*D),(SpMatrix<T> &)(*alpha),(Matrix<bool> &)(*B),pL,pE,pLambda,vecL,vecEps,vecLambda,numThreads,*path);
    } else {
      *path = NULL;
      omp_mask((Matrix<T> &)(*X),(Matrix<T> &)(*D),(SpMatrix<T> &)(*alpha),(Matrix<bool> &)(*B),pL,pE,pLambda,vecL,vecEps,vecLambda,numThreads);
    }
    return alpha;
}
template <typename T>
SpMatrix<T> *_cd(Matrix<T> *X,Matrix<T> *D,SpMatrix<T>*alpha,T lambda1, constraint_type mode, int itermax, T tol,int numThreads) throw(const char *){
  int n = X->m();
  int M = X->n();
  int nD = D->m();
  int K = D->n();
  if (n != nD)
    throw("cd : incompatible X D matrices dimensions");
  int Ka = alpha->m();
  int Ma = alpha->n();
  if (Ma != M || Ka != K) 
    throw("cd : incompatible X D A0 matrices dimensions");
  SpMatrix<T> *alpha0 = new SpMatrix<T>();
  alpha0->copy((SpMatrix<T> &)(*alpha));
  ist((Matrix<T> &)(*X),(Matrix<T> &)(*D),(SpMatrix<T> &)(*alpha0),lambda1,mode,itermax,tol,numThreads);
  return alpha0;
}

template <typename T>
SpMatrix<T> *_somp(Matrix<T> *X,Matrix<T> *D,Vector<int> *groups,int LL, T eps, int numThreads) throw(const char *){
  int *list_groups = groups->rawX();
  int Ng = groups->n();
  int n = X->m();
  int M = X->n();
  int nD = D->m();
  T *prX = X->rawX();
  if (nD != n) throw("_somp : wrong size for argument 2");
  int K = D->n();
  Matrix<T>* Y = new Matrix<T>[Ng];
  if (list_groups[0] != 0)
    throw("somp : First group index should be zero");
  for (int i = 0; i<Ng-1; ++i) {
    if (list_groups[i] >= M) 
      throw("Size of groups is not consistent");
    if (list_groups[i] >= list_groups[i+1]) 
      throw("Group indices should be a strictly non-decreasing sequence");
    Y[i].setData(prX+list_groups[i]*n,n,list_groups[i+1]-list_groups[i]);
  }
  Y[Ng-1].setData(prX+list_groups[Ng-1]*n,n,M-list_groups[Ng-1]);
  SpMatrix<T>* spAlpha = new SpMatrix<T>[Ng];

  somp(Y,(Matrix<T> &)(*D),spAlpha,Ng,LL,eps,numThreads);
  int nzmax=0;
   for (int i = 0; i<Ng; ++i) {
     nzmax += spAlpha[i].nzmax();
   }
  SpMatrix<T> *alpha = new SpMatrix<T>(K,M,nzmax);
  T *Pr = alpha->v();
  int *Ir = alpha->r();
  int *Jc = alpha->pB();
     int count=0;
   int countcol=0;
   int offset=0;
   for (int i = 0; i<Ng; ++i) {
     const T* v = spAlpha[i].v();
     const int* r = spAlpha[i].r();
     const int* pB = spAlpha[i].pB();
     int nn = spAlpha[i].n();
     nzmax = spAlpha[i].nzmax();
     if (nn != 0) {
       for (int j = 0; j<pB[nn]; ++j) {
	 Pr[count]=static_cast<double>(v[j]);
	 Ir[count++]=static_cast<mwSize>(r[j]);
       }
       for (int j = 0; j<=nn; ++j) 
	 Jc[countcol++]=static_cast<mwSize>(offset+pB[j]);
       --countcol;
       offset = Jc[countcol];
     }
   }
   delete[] Y;
   delete[] spAlpha;
   return alpha;
}
template <typename T>
void _l1L2BCD(Matrix<T> *X,Matrix<T> *D,Matrix<T>*alpha0,Vector<int> *groups,T lambda1, constraint_type mode,int itermax,T tol,int numThreads) throw(const char *){
  int n = X->m();
  int M = X->n();
  int nD = D->m();
  int K = D->n();
  T *prX = X->rawX();
  T *pr_alpha = alpha0->rawX();
  if(nD != n)
    throw("l1L2BCD : wrong size for argument 2");
  int Ka = alpha0->m();
  int Ma = alpha0->n();
  if (Ma != M || Ka != K)
    throw("l1L2BCD : wrong size for argument 3");
  int Ng = groups->n();
  int *list_groups = groups->rawX();
  Matrix<T>* Y = new Matrix<T>[Ng];
  Matrix<T>* alpha = new Matrix<T>[Ng];
  if (list_groups[0] != 0)
    throw("l1L2BCD : First group index should be zero");
  for (int i = 0; i<Ng-1; ++i) {
    if (list_groups[i] >= M) 
      throw("l1L2BCD : size of groups is not consistent");
    if (list_groups[i] >= list_groups[i+1]) 
      throw("l1L2BCD : group indices should be a strictly non-decreasing sequence");
    Y[i].setData(prX+list_groups[i]*n,n,list_groups[i+1]-list_groups[i]);
    alpha[i].setData(pr_alpha+list_groups[i]*K,K,list_groups[i+1]-list_groups[i]);
  }
  Y[Ng-1].setData(prX+list_groups[Ng-1]*n,n,M-list_groups[Ng-1]);
  alpha[Ng-1].setData(pr_alpha+list_groups[Ng-1]*K,K,M-list_groups[Ng-1]);
  ist_groupLasso<T>(Y,(Matrix<T> &)(*D),alpha,Ng,lambda1,mode,itermax,tol,numThreads);
  delete[] (Y);
  delete[](alpha);
}

/* end decomp */

/* from prox */
#ifdef SWIGPYTHON
#include <Python.h>
#include <numpy/arrayobject.h>
#endif
template<typename T> 
Matrix<T> *_fistaFlat(Matrix<T> *X,AbstractMatrixB<T> *D,Matrix<T> *alpha0,
	     Matrix<T> *alpha,  
	     Vector<int> *groups, // params
	     int num_threads,
	     int max_it,
	     T L0,
	     bool fixed_step,
	     T gamma,
	     T _lambda,
	     T delta,
	     T lambda2,
	     T lambda3,
	     T a,
	     T b,
	     T c,
	     T tol,
	     int it0,
	     int max_iter_backtracking,
	     bool compute_gram,
	     bool lin_admm,
	     bool admm,
	     bool intercept,
	     bool resetflow,
	     char* name_regul,
	     char* name_loss,
	     bool verbose,
	     bool pos,
	     bool clever,
	     bool log,
	     bool ista,
	     bool subgrad,
	     char* logName,
	     bool is_inner_weights,
	     Vector<T> *inner_weights,
	     int size_group,
	     bool sqrt_step,
	     bool transpose
)
throw(const char *) 
{
using namespace FISTA;
 int mD = D->m();
 int p = D->n();
 int m = X->m();
 int n = X->n();
 int pAlpha = alpha0->m();
 int nAlpha = alpha0->n();
  FISTA::ParamFISTA<T> param;
  param.max_it = max_it;
  param.L0 = L0;
  param.fixed_step = fixed_step;
  param.gamma = gamma;
  param.lambda = _lambda;
  param.delta = delta;
  param.lambda2 = lambda2;
  param.lambda3 = lambda3;
  param.a = a;
  param.b = b;
  param.c = c;
  param.tol = tol;
  param.it0 = it0;
  param.max_iter_backtracking = max_iter_backtracking;
  param.loss = loss_from_string(name_loss);
  if (param.loss==INCORRECT_LOSS)
    throw("fistaFlat: Unknown loss");
  param.compute_gram = compute_gram;
  param.lin_admm = lin_admm;
  param.admm = admm;
  param.intercept = intercept;
  param.resetflow = resetflow;
  param.regul = regul_from_string(name_regul);

  if (param.regul==INCORRECT_REG) {
      throw("fistaFlat: Unknown regularization.\n  For valid names see source code of regul_from_string in spams/src/spams/prox/fista.h\n");
  }
  strncpy(param.name_regul,name_regul,param.length_names);
  strncpy(param.name_loss,name_loss,param.length_names);
  param.verbose = verbose;
  param.pos = pos;
  param.clever = clever;

  if(groups->n() == 0) { // groups is not given
      param.size_group = size_group;
  } else {
    param.ngroups = groups->n();
    if (param.ngroups != pAlpha)
      throw("fistaFlat : Wrong size of param.groups");
    param.groups = groups->rawX();
  }
  if(param.log = log) {
    int n = strlen(logName);
    if(n == 0) 
      throw("fistaFlat : missing field logName");
    param.logName = new char[n+1];
    strcpy(param.logName,logName);
  }
  param.ista = ista;
  param.subgrad = subgrad;
  param.is_inner_weights = is_inner_weights;

  if(is_inner_weights) {
    if(inner_weights == NULL)
      throw("fistaFlat : missing inner_heights ");
    param.inner_weights = inner_weights->rawX();
  }

  param.eval = false;
  param.sqrt_step = sqrt_step;
  param.transpose = transpose;

  if ((param.loss != CUR && param.loss != MULTILOG) && (pAlpha != p || nAlpha != n || mD != m)) { 
      throw("fistaFlat : Argument sizes are not consistent");
   } else if (param.loss == MULTILOG) {
    Vector<T> Xv;
    X->toVect(Xv);
    int maxval = static_cast<int>(Xv.maxval());
    int minval = static_cast<int>(Xv.minval());
    if (minval != 0)
      throw("fistaFlat : smallest class should be 0");
    if (maxval*X->n() > nAlpha || mD != m) {
      cerr << "Number of classes: " << maxval << endl;
      //cerr << "Alpha: " << pAlpha << " x " << nAlpha << endl;
         //cerr << "X: " << X.m() << " x " << X.n() << endl;
      throw("fistaFlat : Argument sizes are not consistent");
    }
  } else if (param.loss == CUR && (pAlpha != D->n() || nAlpha != D->m())) {
      throw("fistaFlat : Argument sizes are not consistent");
   }
   if (param.num_threads == -1) {
      param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   }

   if (param.regul==GRAPH || param.regul==GRAPHMULT) 
    throw("Error: fistaGraph should be used instead");
  if (param.regul==TREE_L0 || param.regul==TREEMULT || param.regul==TREE_L2 || param.regul==TREE_LINF) 
      throw("Error: fistaFlat should be used instead");

  Matrix<T> *optim_info = new Matrix<T>();
  FISTA::solver((Matrix<T> &)(*X),(AbstractMatrixB<T> &)(*D),(Matrix<T> &)(*alpha0),(Matrix<T> &)(*alpha),param,(Matrix<T> &)(*optim_info));
  if (param.log) delete[](param.logName);
  return optim_info;
}

template<typename T> 
Matrix<T> *_fistaTree(
	     Matrix<T> *X,AbstractMatrixB<T> *D,Matrix<T> *alpha0,
	     Matrix<T> *alpha, // tree :
	     Vector<double> *eta_g,SpMatrix<bool> *groups,Vector<int> *own_variables,
	     Vector<int> *N_own_variables, // params :
	     int num_threads,
	     int max_it,
	     T L0,
	     bool fixed_step,
	     T gamma,
	     T _lambda,
	     T delta,
	     T lambda2,
	     T lambda3,
	     T a,
	     T b,
	     T c,
	     T tol,
	     int it0,
	     int max_iter_backtracking,
	     bool compute_gram,
	     bool lin_admm,
	     bool admm,
	     bool intercept,
	     bool resetflow,
	     char* name_regul,
	     char* name_loss,
	     bool verbose,
	     bool pos,
	     bool clever,
	     bool log,
	     bool ista,
	     bool subgrad,
	     char* logName,
	     bool is_inner_weights,
	     Vector<T> *inner_weights,
	     int size_group,
	     bool sqrt_step,
	     bool transpose
)
throw(const char *) 
{
using namespace FISTA;
 int mD = D->m();
 int p = D->n();
 int m = X->m();
 int n = X->n();
 int pAlpha = alpha0->m();
 int nAlpha = alpha0->n();
  FISTA::ParamFISTA<T> param;
  param.max_it = max_it;
  param.L0 = L0;
  param.fixed_step = fixed_step;
  param.gamma = gamma;
  param.lambda = _lambda;
  param.delta = delta;
  param.lambda2 = lambda2;
  param.lambda3 = lambda3;
  param.a = a;
  param.b = b;
  param.c = c;
  param.tol = tol;
  param.it0 = it0;
  param.max_iter_backtracking = max_iter_backtracking;
  param.loss = loss_from_string(name_loss);
  if (param.loss==INCORRECT_LOSS)
    throw("fistaTree: Unknown loss");
  param.compute_gram = compute_gram;
  param.lin_admm = lin_admm;
  param.admm = admm;
  param.intercept = intercept;
  param.resetflow = resetflow;
  param.regul = regul_from_string(name_regul);

  if (param.regul==INCORRECT_REG) {
      throw("fistaTree: Unknown regularization.\n  For valid names see source code of regul_from_string in spams/src/spams/prox/fista.h\n");
  }
  strncpy(param.name_regul,name_regul,param.length_names);
  strncpy(param.name_loss,name_loss,param.length_names);
  param.verbose = verbose;
  param.pos = pos;
  param.clever = clever;

  if(param.log = log) {
    int n = strlen(logName);
    if(n == 0) 
      throw("fistaTree : missing field logName");
    param.logName = new char[n+1];
    strcpy(param.logName,logName);
  }
  param.ista = ista;
  param.subgrad = subgrad;
  param.is_inner_weights = is_inner_weights;

  if(is_inner_weights) {
    if(inner_weights == NULL)
      throw("fistaTree : missing inner_heights ");
    param.inner_weights = inner_weights->rawX();
  }

  param.eval = false;
  param.size_group = size_group;
  param.sqrt_step = sqrt_step;
  param.transpose = transpose;

  if ((param.loss != CUR && param.loss != MULTILOG) && (pAlpha != p || nAlpha != n || mD != m)) { 
      throw("fistaTree : Argument sizes are not consistent");
   } else if (param.loss == MULTILOG) {
    Vector<T> Xv;
    X->toVect(Xv);
    int maxval = static_cast<int>(Xv.maxval());
    int minval = static_cast<int>(Xv.minval());
    if (minval != 0)
      throw("fistaTree : smallest class should be 0");
    if (maxval*X->n() > nAlpha || mD != m) {
      cerr << "Number of classes: " << maxval << endl;
      //cerr << "Alpha: " << pAlpha << " x " << nAlpha << endl;
         //cerr << "X: " << X.m() << " x " << X.n() << endl;
      throw("fistaTree : Argument sizes are not consistent");
    }
  } else if (param.loss == CUR && (pAlpha != D->n() || nAlpha != D->m())) {
      throw("fistaTree : Argument sizes are not consistent");
   }
   if (param.num_threads == -1) {
      param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   }

   if (param.regul==GRAPH || param.regul==GRAPHMULT) 
    throw("Error: fistaGraph should be used instead");
  if (param.regul==TREEMULT && abs<T>(param.lambda2 - 0) < 1e-20) 
      throw("fistaTree error: with multi-task-tree, lambda2 should be > 0");
  TreeStruct<T> tree;
  tree.Nv=0;
  int num_groups = own_variables->n();
  if (num_groups != N_own_variables->n()) {
    throw("fistaTree error: in tree,  own_variables and N_own_variables must have same dimension");
  }
  int *pr_N_own_variables = N_own_variables->rawX();
  int num_var = 0;
  for (int i = 0; i<num_groups; ++i)
    num_var+=pr_N_own_variables[i];
  if (pAlpha < num_var) 
    throw("fistaTree error: Input alpha is too small");
  if(num_groups != eta_g->n())
    throw("fistaTree error: in tree, nb of groups incompatible with eta_g size");
  if((num_groups != groups->n()) || (num_groups != groups->m()))
    throw("fistaTree error: in tree, nb of groups incompatible with groups size");
  for (int i = 0; i<num_groups; ++i) tree.Nv+=pr_N_own_variables[i]; 
   tree.Ng=num_groups;
   tree.weights= eta_g->rawX();
   tree.own_variables= own_variables->rawX();
   tree.N_own_variables=pr_N_own_variables;
   tree.groups_ir= groups->r();
   tree.groups_jc= groups->pB();

  Matrix<T> *optim_info = new Matrix<T>();
  FISTA::solver<T>((Matrix<T> &)(*X),(AbstractMatrixB<T> &)(*D),(Matrix<T> &)(*alpha0),(Matrix<T> &)(*alpha),param,(Matrix<T> &)(*optim_info),NULL,&tree);
  if (param.log) delete[](param.logName);
  return optim_info;
}

template<typename T> 
Matrix<T> *_fistaGraph(
	     Matrix<T> *X,AbstractMatrixB<T> *D,Matrix<T> *alpha0,
	     Matrix<T> *alpha, // tree :
	     Vector<double> *eta_g,SpMatrix<bool> *groups,SpMatrix<bool> *groups_var, // params :
	     int num_threads,
	     int max_it,
	     T L0,
	     bool fixed_step,
	     T gamma,
	     T _lambda,
	     T delta,
	     T lambda2,
	     T lambda3,
	     T a,
	     T b,
	     T c,
	     T tol,
	     int it0,
	     int max_iter_backtracking,
	     bool compute_gram,
	     bool lin_admm,
	     bool admm,
	     bool intercept,
	     bool resetflow,
	     char* name_regul,
	     char* name_loss,
	     bool verbose,
	     bool pos,
	     bool clever,
	     bool log,
	     bool ista,
	     bool subgrad,
	     char* logName,
	     bool is_inner_weights,
	     Vector<T> *inner_weights,
	     int size_group,
	     bool sqrt_step,
	     bool transpose
)
throw(const char *) 
{
  using namespace FISTA;
  int mD = D->m();
  int p = D->n();
  int m = X->m();
  int n = X->n();
  int pAlpha = alpha0->m();
  int nAlpha = alpha0->n();
  FISTA::ParamFISTA<T> param;
  param.max_it = max_it;
  param.L0 = L0;
  param.fixed_step = fixed_step;
  param.gamma = gamma;
  param.lambda = _lambda;
  param.delta = delta;
  param.lambda2 = lambda2;
  param.lambda3 = lambda3;
  param.a = a;
  param.b = b;
  param.c = c;
  param.tol = tol;
  param.it0 = it0;
  param.max_iter_backtracking = max_iter_backtracking;
  param.loss = loss_from_string(name_loss);
  if (param.loss==INCORRECT_LOSS)
    throw("fistaGraph: Unknown loss");
  param.compute_gram = compute_gram;
  param.lin_admm = lin_admm;
  param.admm = admm;
  param.intercept = intercept;
  param.resetflow = resetflow;
  param.regul = regul_from_string(name_regul);

  if (param.regul==INCORRECT_REG) {
      throw("fistaGraph: Unknown regularization.\n  For valid names see source code of regul_from_string in spams/src/spams/prox/fista.h\n");
  }
  strncpy(param.name_regul,name_regul,param.length_names);
  strncpy(param.name_loss,name_loss,param.length_names);
  param.verbose = verbose;
  param.pos = pos;
  param.clever = clever;

  if(param.log = log) {
    int n = strlen(logName);
    if(n == 0) 
      throw("fistaGraph : missing field logName");
    param.logName = new char[n+1];
    strcpy(param.logName,logName);
  }
  param.ista = ista;
  param.subgrad = subgrad;
  param.is_inner_weights = is_inner_weights;

  if(is_inner_weights) {
    if(inner_weights == NULL)
      throw("fistaGraph : missing inner_heights ");
    param.inner_weights = inner_weights->rawX();
  }

  param.eval = false;
  param.size_group = size_group;
  param.sqrt_step = sqrt_step;
  param.transpose = transpose;

  if ((param.loss != CUR && param.loss != MULTILOG) && (pAlpha != p || nAlpha != n || mD != m)) { 
      throw("fistaGraph : Argument sizes are not consistent");
   } else if (param.loss == MULTILOG) {
    Vector<T> Xv;
    X->toVect(Xv);
    int maxval = static_cast<int>(Xv.maxval());
    int minval = static_cast<int>(Xv.minval());
    if (minval != 0)
      throw("fistaGraph : smallest class should be 0");
    if (maxval*X->n() > nAlpha || mD != m) {
      cerr << "Number of classes: " << maxval << endl;
      //cerr << "Alpha: " << pAlpha << " x " << nAlpha << endl;
         //cerr << "X: " << X.m() << " x " << X.n() << endl;
      throw("fistaGraph : Argument sizes are not consistent");
    }
  } else if (param.loss == CUR && (pAlpha != D->n() || nAlpha != D->m())) {
      throw("fistaGraph : Argument sizes are not consistent");
   }
   if (param.num_threads == -1) {
      param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   }

   if (param.regul==TREE_L0 || param.regul==TREEMULT || param.regul==TREE_L2 || param.regul==TREE_LINF) 
      throw("fistaGraph error: fistaTree should be used instead");

  if (param.regul==GRAPHMULT && abs<T>(param.lambda2 - 0) < 1e-20) 
      throw("fistaGraph error: with multi-task-graph, lambda2 should be > 0");
  if(groups->m() != groups->n())
    throw("fistaGraph error: size of field groups is not consistent");
  GraphStruct<T> graph;
  graph.Nv = groups_var->m();
  graph.Ng = groups_var->n();
  graph.weights = eta_g->rawX();
  graph.gg_ir = groups->r();
  graph.gg_jc = groups->pB();
  graph.gv_ir = groups_var->r();
  graph.gv_jc = groups_var->pB();
  if (graph.Nv <= 0 || graph.Ng != groups->n())
    throw("fistaGraph error: size of field groups_var is not consistent");
  if (eta_g->n() != groups_var->n())
    throw("fistaGraph error: size of field eta_g is not consistent");

  Matrix<T> *optim_info = new Matrix<T>();
  FISTA::solver<T>((Matrix<T> &)(*X),(AbstractMatrixB<T> &)(*D),(Matrix<T> &)(*alpha0),(Matrix<T> &)(*alpha),param,(Matrix<T> &)(*optim_info),&graph);
  if (param.log) delete[](param.logName);
  return optim_info;
}

template<typename T> 
Vector<T> *_proximalFlat(Matrix<T> *alpha0,Matrix<T> *alpha, 
		 Vector<int> *groups,  // params
		int num_threads,
		T lambda1,
		T lambda2,
		T lambda3,
		bool intercept,
		bool resetflow,
		char* name_regul,
		bool verbose,
		bool pos,
		bool clever,
		bool eval,
		int size_group,
		bool transpose
		) 
throw(const char *) 
{
using namespace FISTA;
  FISTA::ParamFISTA<T> param;
  param.regul = regul_from_string(name_regul);
  if (param.regul==INCORRECT_REG)
    throw("proximalFlat : Unknown regularization.\n  For valid names see source code of regul_from_string in spams/src/spams/prox/fista.h\n");
  strncpy(param.name_regul,name_regul,param.length_names);
  if (param.regul==GRAPH || param.regul==GRAPHMULT) 
    throw("proximalFlat : proximalGraph should be used instead");
  param.num_threads = (num_threads < 0) ? 1 : num_threads;
  param.lambda = lambda1;
  param.lambda2 = lambda2;
  param.lambda3 = lambda3;
  param.intercept = intercept;
  param.resetflow = resetflow;
  param.verbose = verbose;
  param.pos = pos;
  param.clever = clever;
  param.eval = eval;
  param.transpose = transpose;
  if (param.num_threads == -1) {
    param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   }
  if(groups->n() == 0) { // groups is not given
    param.size_group = size_group;
  } else {
    int pAlpha = alpha0->m();
    param.ngroups = groups->n();
    if (param.ngroups != pAlpha)
      throw("fistaFlat : Wrong size of param.groups");
    param.groups = groups->rawX();
  }
  
  Vector<T> *val_loss = new Vector<T>();
  FISTA::PROX((Matrix<T> &)(*alpha0),(Matrix<T> &)(*alpha),param,(Vector<T> &)(*val_loss));
  return val_loss;
}

template<typename T> 
Vector<T> *_proximalTree(Matrix<T> *alpha0,Matrix<T> *alpha, // tree
		Vector<double> *eta_g,SpMatrix<bool> *groups,Vector<int> *own_variables,
		Vector<int> *N_own_variables, // params :	 
		int num_threads,
		T lambda1,
		T lambda2,
		T lambda3,
		bool intercept,
		bool resetflow,
		char* name_regul,
		bool verbose,
		bool pos,
		bool clever,
		bool eval,
		int size_group,
		bool transpose
		) 
throw(const char *) 
{
using namespace FISTA;
  FISTA::ParamFISTA<T> param;
  param.regul = regul_from_string(name_regul);
  param.num_threads = (num_threads < 0) ? 1 : num_threads;
  param.lambda = lambda1;
  param.lambda2 = lambda2;
  param.lambda3 = lambda3;
  param.intercept = intercept;
  param.resetflow = resetflow;
  param.verbose = verbose;
  param.pos = pos;
  param.clever = clever;
  param.eval = eval;
  param.size_group = size_group;
  param.transpose = transpose;
  if (param.num_threads == -1) {
    param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   }
  int pAlpha = alpha0->m();

  if (param.regul==INCORRECT_REG)
    throw("proximalTree : Unknown regularization.\n  For valid names see source code of regul_from_string in spams/src/spams/prox/fista.h\n");
  strncpy(param.name_regul,name_regul,param.length_names);
  if (param.regul==GRAPH || param.regul==GRAPHMULT) 
    throw("proximalTree : proximalGraph should be used instead");
  if (param.regul==TREEMULT && abs<T>(param.lambda2 - 0) < 1e-20) {
    throw("proximalTree error: with multi-task-tree, lambda2 should be > 0");
  }

  TreeStruct<T> tree;
  tree.Nv=0;
  int num_groups = own_variables->n();
  if (num_groups != N_own_variables->n()) {
    throw("proximalTree error: in tree,  own_variables and N_own_variables must have same dimension");
  }
  int *pr_N_own_variables = N_own_variables->rawX();
  int num_var = 0;
  for (int i = 0; i<num_groups; ++i)
    num_var+=pr_N_own_variables[i];
  if (pAlpha < num_var) 
    throw("proximalTree error: Input alpha is too small");
  if(num_groups != eta_g->n())
    throw("proximalTree error: in tree, nb of groups incompatible with eta_g size");
  if((num_groups != groups->n()) || (num_groups != groups->m()))
    throw("proximalTree error: in tree, nb of groups incompatible with groups size");
  for (int i = 0; i<num_groups; ++i) tree.Nv+=pr_N_own_variables[i]; 
   tree.Ng=num_groups;
   tree.weights= eta_g->rawX();
   tree.own_variables= own_variables->rawX();
   tree.N_own_variables=pr_N_own_variables;
   tree.groups_ir= groups->r();
   tree.groups_jc= groups->pB();
  Vector<T> *val_loss = new Vector<T>();
  FISTA::PROX<T>((Matrix<T> &)(*alpha0),(Matrix<T> &)(*alpha),param,(Vector<T> &)(*val_loss),NULL,&tree);
  return val_loss;
}

template<typename T> 
Vector<T> *_proximalGraph(Matrix<T> *alpha0,Matrix<T> *alpha, // graph
		Vector<double> *eta_g,SpMatrix<bool> *groups,SpMatrix<bool> *groups_var, // params :	 
		int num_threads,
		T lambda1,
		T lambda2,
		T lambda3,
		bool intercept,
		bool resetflow,
		char* name_regul,
		bool verbose,
		bool pos,
		bool clever,
		bool eval,
		int size_group,
		bool transpose
		) 
throw(const char *) 
{
using namespace FISTA;
  FISTA::ParamFISTA<T> param;
  param.regul = regul_from_string(name_regul);
  if (param.regul==INCORRECT_REG)
    throw("proximalGraph : Unknown regularization.\n  For valid names see source code of regul_from_string in spams/src/spams/prox/fista.h\n");
  strncpy(param.name_regul,name_regul,param.length_names);
  if (param.regul==TREE_L0 || param.regul==TREEMULT || param.regul==TREE_L2 || param.regul==TREE_LINF) 
    throw("proximalGraph : proximalTree should be used instead");
  if (param.regul==TREEMULT && abs<T>(param.lambda2 - 0) < 1e-20) {
      throw("proximalGraph error: with multi-task-graph, lambda2 should be > 0");
  }
  param.num_threads = (num_threads < 0) ? 1 : num_threads;
  param.lambda = lambda1;
  param.lambda2 = lambda2;
  param.lambda3 = lambda3;
  param.intercept = intercept;
  param.resetflow = resetflow;
  param.verbose = verbose;
  param.pos = pos;
  param.clever = clever;
  param.eval = eval;
  param.size_group = size_group;
  param.transpose = transpose;
  if (param.num_threads == -1) {
    param.num_threads=1;
#ifdef _OPENMP
      param.num_threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   }
  int pAlpha = alpha0->m();
  if(groups->m() != groups->n())
    throw("proximalGraph error: size of field groups is not consistent");
  GraphStruct<T> graph;
  graph.Nv = groups_var->m();
  graph.Ng = groups_var->n();
  graph.weights = eta_g->rawX();
  graph.gg_ir = groups->r();
  graph.gg_jc = groups->pB();
  graph.gv_ir = groups_var->r();
  graph.gv_jc = groups_var->pB();
  if (graph.Nv <= 0 || graph.Ng != groups->n())
    throw("proximalGraph error: size of field groups_var is not consistent");
  if (eta_g->n() != groups_var->n())
    throw("proximalGraph error: size of field eta_g is not consistent");

  Vector<T> *val_loss = new Vector<T>();
  FISTA::PROX<T>((Matrix<T> &)(*alpha0),(Matrix<T> &)(*alpha),param,(Vector<T> &)(*val_loss),&graph);
  return val_loss;
}

/* end prox */

/* from dictLearn */
template<typename T> 
Matrix<T> *_alltrainDL(Data<T> *X,bool in_memory, Matrix<T> **omA,Matrix<T> **omB,Vector<int> **omiter,bool return_model,Matrix<double> *m_A,Matrix<double> *m_B,int m_iter,
		    Matrix<T> *D1,
		    int num_threads,
		    int batch_size,
		    int K,
		    double lambda1,
		    double lambda2,
		    int iter,
		    double t0, 
		    constraint_type mode,
		    bool posAlpha,
		    bool posD,
		    bool expand,
		    constraint_type_D modeD,
		    bool whiten,
		    bool clean,
		    bool verbose,
		    double gamma1,
		    double gamma2,
		    T rho,
		    int iter_updateD,
		    bool stochastic,
		    int modeParam,
		    bool batch,
		    bool log,
		    char *logName
		    )  throw(const char *){
#ifdef _OPENMP
  num_threads = num_threads <= 0 ? omp_get_num_procs() : num_threads;
#else
  num_threads = 1;
#endif
  if (in_memory) return_model = false;
  if (batch_size < 0) batch_size = 256 * (num_threads + 1);
  int n = X->m();
  int M = X->n();
  Trainer<T>* trainer;
  if(D1->n() == 0) { // D1 is not given
    if (K < 0)
      throw("trainDL : bad parameter K\n");
    trainer = new Trainer<T>(K,batch_size,num_threads);
  } else {
    int nD = D1->m();
    K = D1->n();
    if (n != nD)
      throw("trainDL : sizes of D are not consistent\n");
    if ((m_A->n() == 0) || in_memory) {
      trainer = new Trainer<T>((Matrix<T> &)(*D1),batch_size,num_threads);
    } else {  // model given
      trainer = new Trainer<T>((Matrix<T> &)(*m_A),(Matrix<T> &)(*m_B),(Matrix<T> &)(*D1),m_iter,batch_size,num_threads);
    }
  }
  ParamDictLearn<T> param;
  param.lambda = lambda1;
  param.lambda2 = lambda2;
  param.iter = iter;
  param.t0 = t0;
  param.mode = mode;
  param.posAlpha = posAlpha;
  param.posD = posD;
  param.expand = expand;
  param.modeD = modeD;
  param.whiten = whiten;
  param.clean = clean;
  param.verbose = verbose;
  param.gamma1 = gamma1;
  param.gamma2 = gamma2;
  param.rho = rho;
  param.stochastic = stochastic;
  param.modeParam = static_cast<mode_compute>(modeParam);
  param.batch = batch;
  param.iter_updateD = iter_updateD;
  if(param.log = log) {
    int n = strlen(logName);
    if(n == 0) 
      throw("trainDL : missing field logName");
    param.logName = new char[n+1];
    strcpy(param.logName,logName);
  }
  if (in_memory)
    trainer->trainOffline(*X,param);
  else
    trainer->train(*X,param);
  if (param.log) delete[](param.logName);
  Matrix<T> *D = new Matrix<T>();
  trainer->getD((Matrix<T> &)(*D));
  K = D->n();
  if (return_model) {
    *omA = new Matrix<T>(K,K);
    trainer->getA((Matrix<T> &)(**omA));
    *omB = new Matrix<T>(n,K);
    trainer->getB((Matrix<T> &)(**omB));
    *omiter = new Vector<int>(1);
    int *p = (*omiter)->rawX();
    *p = trainer->getIter();
  } else {
    *omA = NULL;
    *omB = NULL;
    *omiter = NULL;
  }
  delete(trainer);
  return D;
}

/* end  dictLearn */
/* utility : equivalent of matlab im2col in 'sliding' mode */
/* 
   input:
     A : image as matrix of mm lines of nn values (for rgb nn = 3 * dimx)
     RGB : true if image is true color (3 values per pixel)
     m,n : size of chunks (typically 8x8)
  output:
     B : matrix of all possible mxn blocs, size = m*n lines of (mm - m + 1) * (nn -n + 1) values;
     stored by columns.
*/
void im2col_sliding(Matrix<double>  *A,Matrix<double>  *B,int m, int n,bool RGB)  throw(const char *){
  /* if RGB is true A has 3*n columns, R G B columns are consecutives 
   */
  int mm = A->m();
  int nn = A->n();
  int nn1 = RGB ? nn / 3 : nn;
  int M = m * n;
  int N = (mm - m + 1) * (nn -n + 1);
  if (M != B->m() || N != B->n())
    throw("im2col_sliding : incompatible dimensions for output matrix\n");
  double *po = B->rawX();
  double *pi = A->rawX();
  for(int j = 0; j <= nn - n;j++) {
    for(int i = 0;i <= mm - m; i++) {
      for(int kj = j;kj < j + n;kj++) {
	int kj1 = kj;
	if (RGB) {
	  int numpl = kj / nn1;
	  int kp = kj % nn1;
	  kj1 = 3 * kp + numpl;
	}
	for(int ki = i;ki < i + m;ki++) {
	  *po++ = *(pi + ki + kj1 * mm);
	}
      }
    }
  }

}


#endif /* SPAMS_H */
