/*!
/* Software SPAMS v2.2 - Copyright 2009-2011 Julien Mairal 
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
 *
 *
 * \file
 *                toolbox decomp
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File decomp.h
 * \brief Contains sparse decomposition algorithms 
 * It requires the toolbox linalg */

#ifndef DECOMP_H
#define DECOMP_H

#include <utils.h>
/* **************************
 * Greedy Forward Selection 
 * **************************/

/// Forward Selection (or Orthogonal matching pursuit) 
/// Address the problem of:
/// \forall i, \min_{\alpha_i} ||X_i-D\alpha_i||_2^2 
///                        s.t. ||\alphai||_0 <= L or
/// \forall i, \min_{\alpha_i} ||\alpha_i||_0 
///                        s.t. ||\X_i-D\alpha_i||_2^2 <= epsilon
/// This function is 
///   * based on Cholesky decompositions
///   * parallel
///   * optimized for a large number of signals (precompute the Gramm matrix

template <typename T>
void omp(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha, 
      const int *L, const T* eps, const T* lambda, const bool vecL = false,
      const bool vecEps = false, const bool Lambda=false, const int numThreads=-1,
      Matrix<T>* path = NULL);

template <typename T>
void omp_mask(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha, const Matrix<bool>& mask,
      const int *L, const T* eps, const T* lambda, const bool vecL = false,
      const bool vecEps = false, const bool Lambda=false, const int numThreads=-1,
      Matrix<T>* path = NULL);

/// Auxiliary function of omp
template <typename T>
void coreORMP(Vector<T>& scores, Vector<T>& norm, Vector<T>& tmp, 
      Matrix<T>& Un, Matrix<T>& Undn, Matrix<T>& Unds, Matrix<T>& Gs, 
      Vector<T>& Rdn, const AbstractMatrix<T>& G, Vector<int>& ind, 
      Vector<T>& RUn, T& normX, const T* eps, const int* L, const T* lambda,
      T* path = NULL);


/// Auxiliary function of omp
template <typename T>
void coreORMPB(Vector<T>& RtD, const AbstractMatrix<T>& G, Vector<int>& ind, 
      Vector<T>& coeffs, T& normX, const int L, const T eps, const T lambda = 0);

/* **************
 * LARS - Lasso 
 * **************/

/// Defines different types of problem,
///       - constraint on the l1 norm of the coefficients
///       - constraint on the reconstruction error
///       - l1-sparsity penalty 
enum constraint_type { L1COEFFS, L2ERROR, PENALTY, SPARSITY, L2ERROR2, PENALTY2};

/// Implementation of LARS-Lasso for solving
/// \forall i, \min_{\alpha_i} ||X_i-D\alpha_i||_2^2 
///                        s.t. ||\alphai||_1 <= constraint or
/// \forall i, \min_{\alpha_i} ||\alpha_i||_1 
///                        s.t. ||\X_i-D\alpha_i||_2^2 <= constraint or
/// \forall i, \min_{\alpha_i} constraint*||\alpha_i||_1 + ...
///                        ... ||\X_i-D\alpha_i||_2^2 <= T
/// Optionally, the solution might be positive (boolean pos), and a 
/// Least-Square can be solved as a post-processing step.
/// L is a maximum number of coefficients.
/// This function is 
///   * efficient (Cholesky-based)
///   * parallel
///   * optimized for a big number of signals (precompute the Gramm matrix
template <typename T>
void lasso(const Matrix<T>& X, const Matrix<T>& D, 
      SpMatrix<T>& spalpha, 
      int L, const T constraint, const T lambda2 = 0, constraint_type mode = PENALTY,
      const bool pos = false, const bool ols = false, const int numThreads=-1,
      Matrix<T>* path = NULL, const int length_path=-1);

template <typename T>
void lasso(const Data<T>& X, const AbstractMatrix<T>& G, const AbstractMatrix<T>& DtX,
      SpMatrix<T>& spalpha, 
      int L, const T constraint, constraint_type mode = PENALTY,
      const bool pos = false, const bool ols = false, const int numThreads=-1,
      Matrix<T>* path = NULL, const int length_path=-1);

/// second implementation using matrix inversion lemma
template <typename T>
void lasso2(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha,
      int L, const T constraint,const T lambda2=0, constraint_type mode = PENALTY, const bool pos = false,
      const int numThreads = -1, Matrix<T>* path = NULL, const int length_path=-1);

template <typename T>
void lasso2(const Data<T>& X, const AbstractMatrix<T>& G, const AbstractMatrix<T>& DtX,
      SpMatrix<T>& spalpha,
      int L, const T constraint, constraint_type mode = PENALTY, const bool pos = false,
      const int numThreads = -1, Matrix<T>* path = NULL, const int length_path=-1);

/// second implementation using matrix inversion lemma
template <typename T>
void lasso_mask(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha, const Matrix<bool>& mask,
      int L, const T constraint,const T lambda2=0, constraint_type mode = PENALTY, const bool pos = false,
      const int numThreads = -1);

/// second implementation using matrix inversion lemma
template <typename T>
void lassoReweighted(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha,
      int L, const T constraint, constraint_type mode, const bool pos,
      const T sigma,
      const int numThreads = -1);

/// Auxiliary function for lasso
template <typename T>
void coreLARS(Vector<T>& Rdn, Vector<T>& Xdn, Vector<T>& A, 
      Vector<T>& u, Vector<T>& sig,
      Vector<T>& av, Vector<T>& RUn, Matrix<T>& Un, 
      Matrix<T>& Unds, Matrix<T>& Gs,
      Matrix<T>& Gsa, Matrix<T>& workT, Matrix<T>& R,
      const AbstractMatrix<T>& G,T& normX, 
      Vector<int>& ind,Vector<T>& coeffs,const T constraint,
      const bool ols = false,
      const bool pos =false, 
      constraint_type mode = L1COEFFS,
      T* path = NULL, int length_path=-1);

template <typename T>
void coreLARS2(Vector<T>& DtR, const AbstractMatrix<T>& G,
      Matrix<T>& Gs,
      Matrix<T>& Ga,
      Matrix<T>& invGs,
      Vector<T>& u,
      Vector<T>& coeffs,
      Vector<int>& ind,
      Matrix<T>& work,
      T& normX,
      const constraint_type mode,
      const T constraint, const bool pos = false,
      T* pr_path = NULL, int length_path = -1);

template <typename T>
void coreLARS2W(Vector<T>& DtR, AbstractMatrix<T>& G,
      Matrix<T>& Gs,
      Matrix<T>& Ga,
      Matrix<T>& invGs,
      Vector<T>& u,
      Vector<T>& coeffs,
      const Vector<T>& weights,
      Vector<int>& ind,
      Matrix<T>& work,
      T& normX,
      const constraint_type mode,
      const T constraint, const bool pos = false);

/// Auxiliary functoni for coreLARS (Cholesky downdate)
template <typename T>
void downDateLasso(int& j,int& minBasis,T& normX,const bool ols,
      const bool pos, Vector<T>& Rdn, int* ind,
      T* coeffs, Vector<T>& sig, Vector<T>& av,
      Vector<T>& Xdn, Vector<T>& RUn,Matrix<T>& Unm, Matrix<T>& Gsm,
      Matrix<T>& Gsam, Matrix<T>& Undsm, Matrix<T>& Rm);


/* ************************
 * Iterative thresholding
 * ************************/

/// Implementation of IST for solving
/// \forall i, \min_{\alpha_i} ||\alpha_i||_1 
///                        s.t. ||\X_i-D\alpha_i||_2^2 <= constraint or
/// \forall i, \min_{\alpha_i} constraint*||\alpha_i||_1 + ...
///                        ... ||\X_i-D\alpha_i||_2^2 <= T
template <typename T>
void ist(const Matrix<T>& X, const Matrix<T>& D, 
      SpMatrix<T>& spalpha, T lambda, constraint_type mode,
      const int itermax=500, 
      const T tol = 0.5, const int numThreads = -1);
template <typename T>
void ist(const Matrix<T>& X, const Matrix<T>& D, 
      Matrix<T>& spalpha, T lambda, constraint_type mode,
      const int itermax=500, 
      const T tol = 0.5, const int numThreads=-1);


/// coreIST
template <typename T>
void coreIST(const AbstractMatrix<T>& G, Vector<T>& DtR, Vector<T>& coeffs,
      const T thrs, const int itermax = 500, 
      const T tol = 0.5);

/// coreIST constrained
template <typename T>
void coreISTconstrained(const AbstractMatrix<T>& G, Vector<T>& DtR, Vector<T>& coeffs,
      const T normX2,
      const T thrs, const int itermax = 500, 
      const T tol = 0.5);

/// ist for group Lasso
template <typename T>
void ist_groupLasso(const Matrix<T>* XT, const Matrix<T>& D,
      Matrix<T>* alphaT, const int Ngroups, 
      const T lambda, const constraint_type mode,
      const int itermax = 500,
      const T tol = 0.5, const int numThreads = -1);

/// Auxiliary function for ist_groupLasso
template <typename T>
void coreGroupIST(const Matrix<T>& G, Matrix<T>& RtD,
      Matrix<T>& alphat,
      const T thrs,
      const int itermax=500,
      const T tol = 0.5);


/// Auxiliary function for ist_groupLasso
template <typename T>
void coreGroupISTConstrained(const Matrix<T>& G, Matrix<T>& RtD,
      Matrix<T>& alphat, const T normR,
      const T eps,
      const int itermax=500,
      const T tol = 0.5);

/// auxiliary function for ist_groupLasso
template <typename T>
T computeError(const T normX2,const Vector<T>& norms,
      const Matrix<T>& G,const Matrix<T>& RtD,const Matrix<T>& alphat);

/// auxiliary function for ist_groupLasso
template <typename T>
T computeError(const T normX2,
      const Matrix<T>& G,const Vector<T>& DtR,const Vector<T>& coeffs,
      SpVector<T>& coeffs_tmp);

/* ******************
 * Simultaneous OMP 
 * *****************/
template <typename T>
void somp(const Matrix<T>* X, const Matrix<T>& D, SpMatrix<T>* spalpha, 
      const int Ngroups, const int L, const T* pr_eps, const bool adapt=false,
      const int numThreads=-1);

template <typename T>
void somp(const Matrix<T>* X, const Matrix<T>& D, SpMatrix<T>* spalpha, 
      const int Ngroups, const int L, const T eps, const int numThreads=-1);


template <typename T>
void coreSOMP(const Matrix<T>& X, const Matrix<T>& D, const Matrix<T>& G,
      Matrix<T>& vM,
      Vector<int>& rv, const int L, const T eps);

/* *********************
 * Implementation of OMP
 * *********************/

/// Forward Selection (or Orthogonal matching pursuit) 
/// Address the problem of:
/// \forall i, \min_{\alpha_i} ||X_i-D\alpha_i||_2^2 
///                        s.t. ||\alphai||_0 <= L or
/// \forall i, \min_{\alpha_i} ||\alpha_i||_0 
///                        s.t. ||\X_i-D\alpha_i||_2^2 <= epsilon
/// This function is 
///   * efficient (Cholesky-based)
///   * parallel
///   * optimized for a big number of signals (precompute the Gramm matrix

template <typename T>
void omp(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha, 
      const int* pL, const T* peps, const T* pLambda, 
      const bool vecL, const bool vecEps,
      const bool vecLambda, const int numThreads, Matrix<T>* path) {
   int L;
   if (!vecL) {
      L=*pL;
   } else {
      Vector<int> vL(const_cast<int*>(pL),X.n());
      L=vL.maxval();
   }
   spalpha.clear();
   if (L <= 0) return;
   const int M = X.n();
   const int K = D.n();
   L = MIN(X.m(),MIN(L,K));
   Matrix<T> vM(L,M);
   Matrix<int> rM(L,M);

   ProdMatrix<T> G(D, K < 25000 && M > 10);

   int NUM_THREADS=init_omp(numThreads);

   Vector<T>* scoresT=new Vector<T>[NUM_THREADS];
   Vector<T>* normT=new Vector<T>[NUM_THREADS];
   Vector<T>* tmpT=new Vector<T>[NUM_THREADS];
   Vector<T>* RdnT=new Vector<T>[NUM_THREADS];
   Matrix<T>* UnT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* UndnT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* UndsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GsT=new Matrix<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      scoresT[i].resize(K);
      normT[i].resize(K);
      tmpT[i].resize(K);
      RdnT[i].resize(K);
      UnT[i].resize(L,L);
      UnT[i].setZeros();
      UndnT[i].resize(K,L);
      UndsT[i].resize(L,L);
      GsT[i].resize(K,L);
   }

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      Vector<T> Xi;
      X.refCol(i,Xi);
      T normX = Xi.nrm2sq();

      Vector<int> ind;
      rM.refCol(i,ind);
      ind.set(-1);

      Vector<T> RUn;
      vM.refCol(i,RUn);

      Vector<T>& Rdn=RdnT[numT];
      D.multTrans(Xi,Rdn);
      coreORMP(scoresT[numT],normT[numT],tmpT[numT],UnT[numT],UndnT[numT],UndsT[numT],
            GsT[numT],Rdn,G,ind,RUn, normX, vecEps ? peps+i : peps,
            vecL ? pL+i : pL, vecLambda ? pLambda+i : pLambda, 
            path && i==0 ? path->rawX() : NULL);
   }

   delete[](scoresT);
   delete[](normT);
   delete[](tmpT);
   delete[](RdnT);
   delete[](UnT);
   delete[](UndnT);
   delete[](UndsT);
   delete[](GsT);

   /// convert the sparse matrix into a proper format
   spalpha.convert(vM,rM,K);
};

template <typename T>
void omp_mask(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha, const Matrix<bool>& mask,
      const int *pL, const T* peps, const T* pLambda, const bool vecL,
      const bool vecEps, const bool vecLambda, const int numThreads,
      Matrix<T>* path) {
   int L;
   if (!vecL) {
      L=*pL;
   } else {
      Vector<int> vL(const_cast<int*>(pL),X.n());
      L=vL.maxval();
   }
   spalpha.clear();
   if (L <= 0) return;
   const int M = X.n();
   const int K = D.n();
   L = MIN(X.m(),MIN(L,K));
   Matrix<T> vM(L,M);
   Matrix<int> rM(L,M);

   ProdMatrix<T> G(D, K < 25000 && M > 10);

   int NUM_THREADS=init_omp(numThreads);

   Vector<T>* scoresT=new Vector<T>[NUM_THREADS];
   Vector<T>* normT=new Vector<T>[NUM_THREADS];
   Vector<T>* tmpT=new Vector<T>[NUM_THREADS];
   Vector<T>* RdnT=new Vector<T>[NUM_THREADS];
   Matrix<T>* UnT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* UndnT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* UndsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GsT=new Matrix<T>[NUM_THREADS];
   ProdMatrix<T>* GT=new ProdMatrix<T>[NUM_THREADS];
   Matrix<T>* DmaskT=new Matrix<T>[NUM_THREADS];
   Vector<T>* XmaskT=new Vector<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      DmaskT[i].resize(D.m(),D.n());
      XmaskT[i].resize(X.m());
      scoresT[i].resize(K);
      normT[i].resize(K);
      tmpT[i].resize(K);
      RdnT[i].resize(K);
      UnT[i].resize(L,L);
      UnT[i].setZeros();
      UndnT[i].resize(K,L);
      UndsT[i].resize(L,L);
      GsT[i].resize(K,L);
   }

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      Vector<T> Xi;
      X.refCol(i,Xi);

      Vector<int> ind;
      rM.refCol(i,ind);
      ind.set(-1);

      Vector<T> RUn;
      vM.refCol(i,RUn);

      Vector<bool> maski;
      mask.refCol(i,maski);
      Vector<T>& Rdn=RdnT[numT];
      if (maski.allfalse()) continue;
      if (maski.alltrue()) {
         D.multTrans(Xi,Rdn);
         T normX = Xi.nrm2sq();
         coreORMP(scoresT[numT],normT[numT],tmpT[numT],UnT[numT],UndnT[numT],UndsT[numT],
               GsT[numT],Rdn,G,ind,RUn, normX, vecEps ? peps+i : peps,
               vecL ? pL+i : pL, vecLambda ? pLambda+i : pLambda, 
               path && i==0 ? path->rawX() : NULL);
      } else {
         D.copyMask(DmaskT[numT],maski);
         Xi.copyMask(XmaskT[numT],maski);
         T normX = XmaskT[numT].nrm2sq();
         DmaskT[numT].multTrans(XmaskT[numT],Rdn);
         GT[numT].setMatrices(DmaskT[numT],false);
         GT[numT].addDiag(T(1e-10));
         T eps_mask= (vecEps ? *(peps+i) : *peps)*XmaskT[numT].n()/Xi.n();
         coreORMP(scoresT[numT],normT[numT],tmpT[numT],
               UnT[numT],UndnT[numT],UndsT[numT],
               GsT[numT],Rdn,GT[numT],ind,RUn,
               normX, &eps_mask, vecL ? pL+i : pL, 
               vecLambda ? pLambda+i : pLambda, 
               path && i==0 ? path->rawX() : NULL);

         DmaskT[numT].setm(D.m());
         DmaskT[numT].setn(D.n());
         XmaskT[numT].setn(X.m());
      }
   }

   delete[](GT);
   delete[](XmaskT);
   delete[](DmaskT);
   delete[](scoresT);
   delete[](normT);
   delete[](tmpT);
   delete[](RdnT);
   delete[](UnT);
   delete[](UndnT);
   delete[](UndsT);
   delete[](GsT);

   /// convert the sparse matrix into a proper format
   spalpha.convert(vM,rM,K);
};

/// Auxiliary function of omp
template <typename T>
void coreORMPB(Vector<T>& RtD, const AbstractMatrix<T>& G, Vector<int>& ind, 
      Vector<T>& coeffs, T& normX, const int L, const T eps, const T lambda) {
   const int K = G.n();
   Vector<T> scores(K);
   Vector<T> norm(K);
   Vector<T> tmp(K);
   Matrix<T> Un(L,L);
   Matrix<T> Undn(K,L);
   Matrix<T> Unds(L,L);
   Matrix<T> Gs(K,L);
   ind.set(-1);
   coreORMP(scores,norm,tmp,Un,Undn,Unds,Gs,RtD,G,ind,coeffs,normX,&eps,&L,&lambda);
};

/// Auxiliary function of omp
template <typename T>
void coreORMP(Vector<T>& scores, Vector<T>& norm, Vector<T>& tmp, Matrix<T>& Un,
      Matrix<T>& Undn, Matrix<T>& Unds, Matrix<T>& Gs, Vector<T>& Rdn,
      const AbstractMatrix<T>& G,
      Vector<int>& ind, Vector<T>& RUn, 
       T& normX, const T* peps, const int* pL, const T* plambda,
      T* path) {
   const T eps = abs<T>(*peps);
   const int L = MIN(*pL,Gs.n());
   const T lambda=*plambda;
   if ((normX <= eps) || L == 0) return;
   const int K = scores.n();
   scores.copy(Rdn);
   norm.set(T(1.0));
   Un.setZeros();

   // permit unsafe low level access
   T* const prUn = Un.rawX();
   T* const prUnds = Unds.rawX();
   T* const prUndn = Undn.rawX();
   T* const prGs = Gs.rawX();
   T* const prRUn= RUn.rawX();
   if (path)
      memset(path,0,K*L*sizeof(T));

   int j;
   for (j = 0; j<L; ++j) {
      const int currentInd=scores.fmax();
      if (norm[currentInd] < 1e-8) {
         ind[j]=-1;
         break;
      }
      const T invNorm=T(1.0)/sqrt(norm[currentInd]);
      const T RU=Rdn[currentInd]*invNorm;
      const T delta = RU*RU;
      if (delta < 2*lambda) {
         break;
      }

      RUn[j]=RU;
      normX -= delta;
      ind[j]=currentInd;
      //for (int k = 0; k<j; ++k) prUn[j*L+k]=0.0;
      //prUn[j*L+j]=T(1.0);
      
      //    for (int k = 0; k<j; ++k) prUnds[k*L+j]=prUndn[k*K+currentInd];
      // MGS algorithm, Update Un 
      //      int iter = norm[currentInd] < 0.5 ? 2 : 1;
      //int iter=1;
      //     for (int k = 0; k<iter; ++k) {
      ///       for (int l = 0; l<j; ++l) {
      //         T scal=-cblas_dot<T>(j+1-l,prUn+j*L+l,1,prUnds+l*L+l,1);
      //        T scal = -prUnds[l*L+j];
      //         cblas_axpy<T>(l+1,scal,prUn+l*L,1,prUn+j*L,1);
      //       }
      //    }

      prUn[j*L+j]=-T(1.0);
      cblas_copy<T>(j,prUndn+currentInd,K,prUn+j*L,1);
      cblas_trmv<T>(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,j,prUn,L,prUn+j*L,1);
      cblas_scal<T>(j+1,-invNorm,prUn+j*L,1);
 
      if (j == L-1 || (normX <= eps)) {
         ++j;
         break;
      }

      if (path) {
         T* last_path=path+(L-1)*K;
         cblas_copy<T>(j+1,prRUn,1,last_path,1);
         cblas_trmv<T>(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,
               j+1,prUn,L,last_path,1); 
         for (int k = 0; k<=j; ++k) {
            path[j*K+ind[k]]=last_path[k];
         }
      }

      // update the variables Gs, Undn, Unds, Rdn, norm, scores
      Vector<T> Gsj;
      Gs.refCol(j,Gsj);
      G.copyCol(currentInd,Gsj);
      cblas_gemv<T>(CblasColMajor,CblasNoTrans,K,j+1,T(1.0),prGs,K,prUn+j*L,1,
            T(0.0),prUndn+j*K,1);
     // prUnds[j*L+j] = prUndn[j*K+currentInd];
      Vector<T> Undnj;
      Undn.refCol(j,Undnj);
      Rdn.add(Undnj,-RUn[j]);
      tmp.sqr(Undnj);
      norm.sub(tmp);
      scores.sqr(Rdn);
      scores.div(norm);
      for (int k = 0; k<=j; ++k) scores[ind[k]]=T();
   }
   // compute the final coefficients 
   cblas_trmv<T>(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,
         j,prUn,L,prRUn,1); 
   if (path) {
      memset(path+(L-1)*K,0,L*sizeof(T));
      for (int k = 0; k<j; ++k) {
         path[(j-1)*K+ind[k]]=prRUn[k];
      }
   }
};



/* **************
 * LARS - Lasso 
 * **************/

/// Implementation of LARS-Lasso for solving
/// \forall i, \min_{\alpha_i} ||X_i-D\alpha_i||_2^2 
///                        s.t. ||\alphai||_1 <= constraint or
/// \forall i, \min_{\alpha_i} ||\alpha_i||_1 
///                        s.t. ||\X_i-D\alpha_i||_2^2 <= constraint or
/// \forall i, \min_{\alpha_i} constraint*||\alpha_i||_1 + ...
///                        ... ||\X_i-D\alpha_i||_2^2 <= T
/// Optionally, the solution might be positive (boolean pos), and a 
/// Least-Square can be solved as a post-processing step.
/// L is a maximum number of coefficients.
/// This function is 
///   * efficient (Cholesky-based)
///   * parallel
///   * optimized for a big number of signals (precompute the Gramm matrix

template <typename T>
void lasso(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha, 
      int L, const T lambda, const T lambda2, constraint_type mode, 
      const bool pos, const bool ols, const int numThreads,
      Matrix<T>* path, const int length_path) {
   ProdMatrix<T> G(D, X.n() > 10 && D.n() < 50000);
   G.addDiag(MAX(lambda2,1e-10));
   ProdMatrix<T> DtX(D,X,false);
   lasso(X,G,DtX,spalpha,L,lambda,mode,pos,ols,numThreads,path,length_path);
}

template <typename T>
void lasso(const Data<T>& X, const AbstractMatrix<T>& G, 
      const AbstractMatrix<T>& DtX, SpMatrix<T>& spalpha, 
      int L, const T lambda, constraint_type mode, 
      const bool pos, const bool ols, const int numThreads,
      Matrix<T>* path, const int length_path) {

   spalpha.clear();
   const int M = X.n();
   const int K = G.n();
   Matrix<T> vM;
   Matrix<int> rM;
   vM.resize(L,M);
   rM.resize(L,M);

   if (L <= 0) return;
   if (path) path->setZeros();
   
   int NUM_THREADS=init_omp(numThreads);

   //ProdMatrix<T> G(D, K < 25000 && M > 10);

   Vector<T>* RdnT=new Vector<T>[NUM_THREADS];
   Vector<T>* XdnT =new Vector<T>[NUM_THREADS];
   Vector<T>* AT=new Vector<T>[NUM_THREADS];
   Vector<T>* uT=new Vector<T>[NUM_THREADS];
   Vector<T>* sigT=new Vector<T>[NUM_THREADS];
   Vector<T>* avT=new Vector<T>[NUM_THREADS];
   Vector<T>* RUnT = new Vector<T>[NUM_THREADS];
   Matrix<T>* UnT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* RT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* UndsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GsaT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* workT=new Matrix<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      RdnT[i].resize(K);
      if (ols) XdnT[i].resize(K);
      AT[i].resize(K);
      uT[i].resize(L);
      sigT[i].resize(L);
      avT[i].resize(L);
      if (ols) RUnT[i].resize(L);
      UnT[i].resize(L,L);
      UnT[i].setZeros();
      UndsT[i].resize(L,L);
      UndsT[i].setZeros();
      GsT[i].resize(K,L);
      GsaT[i].resize(L,L);
      workT[i].resize(K,2);
      RT[i].resize(L,L);
   }

   Vector<T> norms;
   X.norm_2sq_cols(norms);
   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      T normX = norms[i]; 

      Vector<int> ind;
      rM.refCol(i,ind);
      Vector<T> coeffs;
      vM.refCol(i,coeffs);
      coeffs.setZeros();

      Vector<T>& Rdn=RdnT[numT];
      DtX.copyCol(i,Rdn);
      coreLARS(Rdn,XdnT[numT], AT[numT], uT[numT], sigT[numT], avT[numT],
            RUnT[numT], UnT[numT], UndsT[numT], GsT[numT], GsaT[numT], 
            workT[numT],RT[numT],G,normX, ind,coeffs,lambda,ols,pos,
            mode,path && i==0 ? path->rawX() : NULL, length_path);
   }

   delete[](RdnT);
   delete[](XdnT);
   delete[](AT);
   delete[](uT);
   delete[](sigT);
   delete[](avT);
   delete[](RUnT);
   delete[](UnT);
   delete[](RT);
   delete[](UndsT);
   delete[](GsT);
   delete[](GsaT);
   delete[](workT);

   /// convert the sparse matrix into a proper format
   spalpha.convert(vM,rM,K);
};

/// Auxiliary function for lasso
template <typename T>
void coreLARS(Vector<T>& Rdnv, Vector<T>& Xdnv, Vector<T>& Av,
      Vector<T>& uv, Vector<T>& sigv, Vector<T>& avv, Vector<T>& RUnv,
      Matrix<T>& Unm, Matrix<T>& Undsm, Matrix<T>& Gsm,
      Matrix<T>& Gsam, Matrix<T>& workm, Matrix<T>& Rm, 
      const AbstractMatrix<T>& Gm,T& normX, 
      Vector<int>& indv,Vector<T>& coeffsv,const T constraint,
      const bool ols,const bool pos, constraint_type mode,
      T* path, int length_path) {
   if (mode == L2ERROR && normX < constraint) return;

   const int LL = Gsm.n();
   const int K = Gsm.m();
   const int L = MIN(LL,K);
   if (length_path <= 1) length_path=4*L;
   // permit unsafe fast low level access
   T* const Rdn = Rdnv.rawX();
   T* const Xdn = Xdnv.rawX();
   T* const A = Av.rawX();
   T* const u = uv.rawX();
   T* const sig = sigv.rawX();
   T* const av = avv.rawX();
   T* const RUn = RUnv.rawX();
   T* const Un = Unm.rawX();
   T* const Unds = Undsm.rawX();
   T* const Gs = Gsm.rawX();
   T* const Gsa = Gsam.rawX();
   T* const work = workm.rawX();
   //T* const G = Gm.rawX();
   T* const R = Rm.rawX();
   int* ind = indv.rawX();
   T* coeffs = coeffsv.rawX();

   coeffsv.setZeros();
   indv.set(-1);

   if (ols) Xdnv.copy(Rdnv);
   int currentInd= pos ? Rdnv.max() : Rdnv.fmax();
   bool newAtom=true;
   T Cmax;
   int iter=1;
   T thrs = 0.0;

   int* const ind_orig = ind;
   T* const coeffs_orig = coeffs;

   int j;
   for (j = 0; j<L; ++j) {
      if (newAtom) {
         ind[j]=currentInd;

         if (pos) {
            Cmax = Rdn[currentInd];
            sig[j]=1.0;
         } else {
            Cmax = abs<T>(Rdn[currentInd]);
            sig[j] = SIGN(Rdn[currentInd]);
         }
         for (int k = 0; k<=j; ++k) Un[j*L+k]=0.0;
         Un[j*L+j]=1.0;
         Gm.extract_rawCol(currentInd,Gs+K*j);
         for (int k = 0; k<j; ++k) Gs[K*j+ind[k]] *= sig[k];
         if (sig[j] < 0) {
            Rdn[currentInd]=-Rdn[currentInd];
            if (ols) Xdn[currentInd]=-Xdn[currentInd];
            cblas_scal<T>(K,sig[j],Gs+K*j,1);
            cblas_scal<T>(j+1,sig[j],Gs+currentInd,K);
         }
         cblas_copy<T>(j+1,Gs+currentInd,K,Gsa+j*L,1);
         for (int k = 0; k<j; ++k) Gsa[k*L+j]=Gsa[j*L+k];

         // <d_j,d_i>
         cblas_copy<T>(j,Gsa+j*L,1,Unds+j,L);
         // <U_j final,d_i>
         cblas_trmv<T>(CblasColMajor,CblasUpper,CblasTrans,CblasNonUnit,
               j+1,Un,L,Unds+j,L);
         // norm2
         T norm2=Gsa[j*L+j];
         for (int k = 0; k<j; ++k) norm2 -= Unds[k*L+j]*Unds[k*L+j];
         if (norm2 < 1e-15) {
            ind[j]=-1;
      //      cerr << "bad exit" << endl;
            break;
         }
      
      //   int iter2 = norm2 < 0.5 ? 2 : 1;
      //   for(int k = 0; k<iter2; ++k) {
      //      for (int l = 0; l<j; ++l) {
      //         T scal=-cblas_dot<T>(j+1-l,Un+j*L+l,1,Unds+l*L+l,1);
      //         cblas_axpy<T>(l+1,scal,Un+l*L,1,Un+j*L,1);
      //      }
      //   }
         Un[j*L+j]=-T(1.0);
         cblas_copy<T>(j,Unds+j,L,Un+j*L,1);
         cblas_trmv<T>(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,j,Un,L,Un+j*L,1);

         /// Un is the orthogonalized vectors in the D basis
         T invNorm=1.0/sqrt(norm2);
         cblas_scal<T>(j+1,-invNorm,Un+j*L,1);
         Unds[j*L+j]=cblas_dot<T>(j+1,Un+j*L,1,Gsa+j*L,1);
      }

      for (int k = 0; k<=j; ++k) u[k]=T(1.0);
      cblas_trmv<T>(CblasColMajor,CblasUpper,CblasTrans,CblasNonUnit,
            j+1,Un,L,u,1);

      T a = T(1.0)/cblas_nrm2<T>(j+1,u,1);

      cblas_trmv<T>(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,
            j+1,Un,L,u,1);
      cblas_scal<T>(j+1,a,u,1);

      cblas_gemv<T>(CblasColMajor,CblasNoTrans,K,j+1,T(1.0),Gs,K,u,1,T(0.0),A,1);

      T potentNorm=0.0;
      if (!ols) {
         for (int k = 0; k<=j; ++k)  potentNorm += Rdn[ind[k]]*u[k];
      }

      if (pos) {
         for (int k = 0; k<K; ++k) {
            T diff = a-A[k];
            work[k]= diff <= 0 ? INFINITY : (Cmax-Rdn[k])/diff;
         }
         for (int k = 0; k<=j; ++k) {
            work[ind[k]]=INFINITY; 
         }
         for (int k = 0; k<K; ++k) 
            if (work[k] <=0) work[k]=INFINITY;
         currentInd =cblas_iamin<T>(K,work,1);
      } else {
         memset(work,0,2*K*sizeof(T));
         for (int k = 0; k<=j; ++k) {
            const int index=2*ind[k];
            work[index]=INFINITY; 
            work[index+1]=INFINITY; 
         }
         for (int k = 0; k<K; ++k) {
            const int index=2*k;
            if (!work[index]) {
               const T diff1=a-A[k];
               work[index]= diff1 <= 0 ? INFINITY : (Cmax-Rdn[k])/diff1;
               const T diff2=a+A[k];
               work[index+1]=diff2 <= 0 ? INFINITY : (Cmax+Rdn[k])/diff2;
            }
         }
         currentInd =cblas_iamin<T>(2*K,work,1);
      }
      T gamma=work[currentInd];
      T gammaMin=0;
      int minBasis=0;

      //if (j == L-1) gamma=potentNorm;

      if (mode == PENALTY) {
         gamma=MIN(gamma,(Cmax-constraint)/a);
      }

//      if (j > 0) {
         vDiv<T>(j+1,coeffs,u,work);
         cblas_scal<T>(j+1,-T(1.0),work,1);
         /// voir pour petites valeurs
         for (int k=0; k<=j; ++k) 
            if (coeffs[k]==0 || work[k] <=0) work[k]=INFINITY;
         minBasis=cblas_iamin<T>(j+1,work,1);
         gammaMin=work[minBasis];
         if (gammaMin < gamma) gamma=gammaMin;
 //     }

      if (mode == L1COEFFS) {
         T Tu = 0.0;
         for (int k = 0; k<=j; ++k) Tu += u[k];

         if (Tu > EPSILON) 
            gamma= MIN(gamma,(constraint-thrs)/Tu);
         thrs+=gamma*Tu;
      }

      // compute the norm of the residdual

      if (ols == 0) {
         const T t = gamma*gamma - 2*gamma*potentNorm;
         if (t > 0 || isnan(t) || isinf(t)) {
      //      cerr << "bad bad exit" << endl;
     //       cerr << t << endl;
            ind[j]=-1;
            break;
         }
         normX += t;
      } else {
         // plan the last orthogonal projection
         if (newAtom) {
            RUn[j]=0.0;
            for (int k = 0; k<=j; ++k) RUn[j] += Xdn[ind[k]]*
               Un[j*L+k];
            normX -= RUn[j]*RUn[j];
         }
      }

      // Update the coefficients
      cblas_axpy<T>(j+1,gamma,u,1,coeffs,1);

      if (pos) {
         for (int k = 0; k<j+1; ++k)
            if (coeffs[k] < 0) coeffs[k]=0;
      }

      cblas_axpy<T>(K,-gamma,A,1,Rdn,1);
      if (!pos) currentInd/= 2;
      if (path) {
         for (int k = 0; k<=j; ++k) 
            path[iter*K+ind[k]]=coeffs[k]*sig[k];
      }

      if (gamma == gammaMin) {
         downDateLasso<T>(j,minBasis,normX,ols,pos,Rdnv,ind,coeffs,sigv,
               avv,Xdnv, RUnv, Unm, Gsm, Gsam,Undsm,Rm);
         newAtom=false;
         Cmax=abs<T>(Rdn[ind[0]]);
         --j;
      } else {
         newAtom=true;
      }
      ++iter;

      if (mode == PENALTY) {
         thrs=abs<T>(Rdn[ind[0]]);
      }

      if ((j == L-1) || 
            (mode == PENALTY && (thrs - constraint < 1e-15)) ||
            (mode == L1COEFFS && (thrs - constraint > -1e-15)) || 
            (newAtom && mode == L2ERROR && (normX - constraint < 1e-15)) ||
            (normX < 1e-15) ||
            (iter >= length_path)) {
     //       cerr << "exit" << endl;
     //       PRINT_F(thrs)
     //       PRINT_F(constraint)
     //       PRINT_F(normX)
         break;
      }

   }
   if (ols) {
      cblas_copy<T>(j+1,RUn,1,coeffs,1);
      cblas_trmv<T>(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,
            j+1,Un,L,coeffs,1);
   }
   vMul<T>(j+1,coeffs,sig,coeffs);
};

/// Auxiliary functoni for coreLARS (Cholesky downdate)
template <typename T>
inline void downDateLasso(int& j,int& minBasis,T& normX,const bool ols,
      const bool pos,
      Vector<T>& Rdnv, int* ind,
      T* coeffs, Vector<T>& sigv, Vector<T>& avv,
      Vector<T>& Xdnv, Vector<T>& RUnv,Matrix<T>& Unm, Matrix<T>& Gsm,
      Matrix<T>& Gsam, Matrix<T>& Undsm, Matrix<T>& Rm) {
   int k,l;
   const int L = Gsm.n();
   const int K = Gsm.m();
   T* const Rdn = Rdnv.rawX();
   T* const Xdn = Xdnv.rawX();
   T* const sig = sigv.rawX();
   T* const av = avv.rawX();
   T* const RUn = RUnv.rawX();
   T* const Un = Unm.rawX();
   T* const Unds = Undsm.rawX();
   T* const Gs = Gsm.rawX();
   T* const Gsa = Gsam.rawX();
   T* const R = Rm.rawX();

   int indB=ind[minBasis];

   if (!pos && sig[minBasis] < 0) {
      // Update Rdn
      Rdn[indB]=-Rdn[indB];
      if (ols) Xdn[indB]=-Xdn[indB];
   }

   int num=j-minBasis;
   for (int k = 0; k<num*num;++k) R[k]=0.0;
   for (int k = 0; k<num; ++k) R[k*num+k]=1.0;
   // Update Un
   for (int k = minBasis+1; k<=j; ++k) {
      T a = -Un[k*L+minBasis]/Un[minBasis*L+minBasis];
      av[k-minBasis-1] = a;
      cblas_axpy<T>(minBasis,a,Un+minBasis*L,1,Un+k*L,1);
   }
   for (int k = minBasis+1; k<=j; ++k) {
      cblas_copy<T>(minBasis,Un+k*L,1,Un+(k-1)*L,1);
      cblas_copy<T>(num,Un+k*L+minBasis+1,1,Un+(k-1)*L+minBasis,1);
   }
   T alpha=1.0;
   T alphab,gamma,lambda;
   for (int k = 0; k<num; ++k) {
      alphab=alpha+av[k]*av[k];
      R[k*num+k]=sqrt(alphab/alpha);
      gamma=av[k]*R[k*num+k]/alphab;
      alpha=alphab;
      cblas_copy<T>(num-k-1,av+k+1,1,R+k*num+k+1,1);
      cblas_scal<T>(num-k-1,gamma,R+k*num+k+1,1);
   }
   if (num > 0) {
      trtri<T>(low,nonUnit,num,R,num);
      cblas_trmm<T>(CblasColMajor,CblasRight,CblasLower,CblasTrans,CblasNonUnit,
            j,num,T(1.0),R,num,Un+minBasis*L,L);
   }

   // Update Unds
   for (int k = minBasis+1; k<=j; ++k) 
      cblas_axpy<T>(j-minBasis,av[k-minBasis-1],Unds+minBasis*L+minBasis+1,1,
            Unds+k*L+minBasis+1,1);
   for (int k = 0; k<minBasis; ++k) 
      for (int l = minBasis+1; l<=j; ++l) 
         Unds[k*L+l-1]=Unds[k*L+l];
   for (int k = minBasis+1; k<=j; ++k) 
      cblas_copy<T>(j-minBasis,Unds+k*L+minBasis+1,1,Unds+(k-1)*L+minBasis,1);
   if (num > 0)
      cblas_trmm<T>(CblasColMajor,CblasRight,CblasLower,CblasTrans,CblasNonUnit,
            j-minBasis,num,T(1.0),R,num,Unds+minBasis*L+minBasis,L);
   for (int k = minBasis+1; k<=j; ++k)
      for (int l = 0; l<k; ++l) Unds[k*L+l]=0.0;

   // Update Gs
   for (int k = minBasis+1; k<=j; ++k) {
      cblas_copy<T>(K,Gs+k*K,1,Gs+(k-1)*K,1);
   }
   if (!pos && sig[minBasis] < T(0.0)) cblas_scal<T>(j,T(-1.0),Gs+indB,K);
   // Update Gsa
   for (int k = minBasis+1; k<=j; ++k) {
      cblas_copy<T>(minBasis,Gsa+k*L,1,Gsa+(k-1)*L,1);
      cblas_copy<T>(j-minBasis,Gsa+k*L+minBasis+1,1,Gsa+(k-1)*L+minBasis,1);
   }
   for (int k = 0; k<minBasis; ++k) {
      for (int l = minBasis+1; l<=j; ++l) Gsa[k*L+l-1]=Gsa[k*L+l];
   }

   // Update sig
   for (int k = minBasis+1; k<=j && !pos; ++k) sig[k-1]=sig[k];
   // Update ind
   for (int k = minBasis+1; k<=j; ++k) ind[k-1]=ind[k];
   ind[j]=-1;

   for (int k = minBasis+1; k<=j; ++k) coeffs[k-1]=coeffs[k];
   coeffs[j]=0.0;

   if (ols) {
      // Update RUn and normX
      for (int k = minBasis; k<=j; ++k)
         normX += RUn[k]*RUn[k];
      for (int k = minBasis; k<j; ++k) {
         RUn[k]=0.0;
         for (int l = 0; l<=k; ++l) RUn[k] += Xdn[ind[l]]*
            Un[k*L+l];
         normX -= RUn[k]*RUn[k];
      }
   }

   // Update j
   --j;
}

/// second implementation using matrix inversion lemma
template <typename T>
void lassoReweighted(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha,
      int L, const T constraint, constraint_type mode, const bool pos,
      const T sigma,
      const int numThreads) {
   spalpha.clear();
   const int M = X.n();
   const int K = D.n();
   Matrix<T> vM;
   Matrix<int> rM;
   vM.resize(L,M);
   rM.resize(L,M);
   const int iterR = 30;
   
   if (L <= 0) return;

   int NUM_THREADS=init_omp(numThreads);

   //ProdMatrix<T> G(D, K < 25000 && M > 10);
   ProdMatrix<T> G(D, K < 50000);
   //Matrix<T> G;
   //D.XtX(G);
   G.addDiag(1e-10);

   Vector<T>* DtRT=new Vector<T>[NUM_THREADS];
   Vector<T>* DtRRT=new Vector<T>[NUM_THREADS];
   Vector<T>* uT=new Vector<T>[NUM_THREADS];
   Vector<T>* weightsT=new Vector<T>[NUM_THREADS];
   Vector<int>* inddT=new Vector<int>[NUM_THREADS];
   Matrix<T>* GsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GaT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* invGsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* workT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GT=new Matrix<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      DtRT[i].resize(K);
      DtRRT[i].resize(K);
      uT[i].resize(K);
      weightsT[i].resize(K);
      GT[i].resize(K,K);
      inddT[i].resize(K);
      GsT[i].resize(L,L);
      invGsT[i].resize(L,L);
      GaT[i].resize(K,L);
      workT[i].resize(K,3);
      workT[i].setZeros();
   }

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      Vector<T> Xi;
      X.refCol(i,Xi);
      T normXo = Xi.nrm2sq();
      T normX = normXo;

      Vector<int> ind;
      rM.refCol(i,ind);
      Vector<T> coeffs;
      vM.refCol(i,coeffs);
      Vector<T>& DtR=DtRT[numT];
      Vector<T>& DtRR = DtRRT[numT];
      D.multTrans(Xi,DtR);
      DtRR.copy(DtR);
      coreLARS2(DtRR,G,GsT[numT],GaT[numT],invGsT[numT],uT[numT],coeffs,
            ind,workT[numT],normX,mode,constraint,pos);
      //Matrix<T>& GG = GT[numT];
      Vector<T>& weights = weightsT[numT];
      //Vector<int>& indd = inddT[numT];
      for (int j = 0; j<iterR; ++j) {
         const T sig = sigma*pow(0.7,iterR-1-j);
         weights.set(sig);
         for (int k = 0; k<K; ++k) {
            if (ind[k] != -1) {
               weights[ind[k]] = MAX(1e-4,sig*exp(-sig*abs<T>(coeffs[k])));
            } else {
               break;
            }
         }
         DtRR.copy(DtR);
         normX=normXo;
         coreLARS2W(DtRR,G,GsT[numT],GaT[numT],invGsT[numT],uT[numT],coeffs,weights,
               ind,workT[numT],normX,mode,constraint,pos);
      }
   }

   delete[](DtRT);
   delete[](DtRRT);
   delete[](inddT);
   delete[](uT);
   delete[](weightsT);
   delete[](GsT);
   delete[](GT);
   delete[](GaT);
   delete[](invGsT);
   delete[](workT);

   /// convert the sparse matrix into a proper format
   spalpha.convert(vM,rM,K);

}

template <typename T>
void lassoWeight(const Matrix<T>& X, const Matrix<T>& D, const Matrix<T>& weights,
      SpMatrix<T>& spalpha, 
      int L, const T constraint, constraint_type mode, const bool pos,
      const int numThreads) {

   spalpha.clear();
   const int M = X.n();
   const int K = D.n();
   Matrix<T> vM;
   Matrix<int> rM;
   vM.resize(L,M);
   rM.resize(L,M);
   
   if (L <= 0) return;

   int NUM_THREADS=init_omp(numThreads);

   //ProdMatrix<T> G(D, K < 25000 && M > 10);
   ProdMatrix<T> G(D, K < 50000);
   //Matrix<T> G;
   //D.XtX(G);
   G.addDiag(1e-10);

   Vector<T>* DtRT=new Vector<T>[NUM_THREADS];
   Vector<T>* uT=new Vector<T>[NUM_THREADS];
   Matrix<T>* GsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GaT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* invGsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* workT=new Matrix<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      DtRT[i].resize(K);
      uT[i].resize(K);
      uT[i].setZeros();
      GsT[i].resize(L,L);
      invGsT[i].resize(L,L);
      GaT[i].resize(K,L);
      workT[i].resize(K,3);
      workT[i].setZeros();
   }

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      Vector<T> Xi;
      X.refCol(i,Xi);
      T normX = Xi.nrm2sq();

      Vector<int> ind;
      rM.refCol(i,ind);
      Vector<T> coeffs;
      vM.refCol(i,coeffs);

      Vector<T>& DtR=DtRT[numT];
      D.multTrans(Xi,DtR);
      Vector<T> we;
      weights.refCol(i,we);

      coreLARS2W(DtR,G,GsT[numT],GaT[numT],invGsT[numT],uT[numT],coeffs,we,
            ind,workT[numT],normX,mode,constraint,pos);
   }

   delete[](DtRT);
   delete[](uT);
   delete[](GsT);
   delete[](GaT);
   delete[](invGsT);
   delete[](workT);

   /// convert the sparse matrix into a proper format
   spalpha.convert(vM,rM,K);
};

template <typename T>
void lassoWeightPreComputed(const Matrix<T>& X, const Matrix<T>& G, const Matrix<T>& DtR, const Matrix<T>& weights,
      SpMatrix<T>& spalpha, 
      int L, const T constraint, constraint_type mode, const bool pos,
      const int numThreads) {

   spalpha.clear();
   const int M = X.n();
   const int K = G.n();
   Matrix<T> vM;
   Matrix<int> rM;
   vM.resize(L,M);
   rM.resize(L,M);
   
   if (L <= 0) return;

   int NUM_THREADS=init_omp(numThreads);

   Vector<T>* DtRT=new Vector<T>[NUM_THREADS];
   Vector<T>* uT=new Vector<T>[NUM_THREADS];
   Matrix<T>* GsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GaT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* invGsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* workT=new Matrix<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      DtRT[i].resize(K);
      uT[i].resize(K);
      uT[i].setZeros();
      GsT[i].resize(L,L);
      invGsT[i].resize(L,L);
      GaT[i].resize(K,L);
      workT[i].resize(K,3);
      workT[i].setZeros();
   }

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      Vector<T> Xi;
      X.refCol(i,Xi);
      T normX = Xi.nrm2sq();

      Vector<int> ind;
      rM.refCol(i,ind);
      Vector<T> coeffs;
      vM.refCol(i,coeffs);

      Vector<T>& DtRi=DtRT[numT];
      DtR.copyCol(i,DtRi);
      Vector<T> we;
      weights.refCol(i,we);

      coreLARS2W(DtRi,G,GsT[numT],GaT[numT],invGsT[numT],uT[numT],coeffs,we,
            ind,workT[numT],normX,mode,constraint,pos);
   }

   delete[](DtRT);
   delete[](uT);
   delete[](GsT);
   delete[](GaT);
   delete[](invGsT);
   delete[](workT);

   /// convert the sparse matrix into a proper format
   spalpha.convert(vM,rM,K);
};

/// second implementation using matrix inversion lemma
template <typename T>
void lasso_mask(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha, const Matrix<bool>& mask,
      int L, const T constraint,const T lambda2, constraint_type mode, const bool pos,
      const int numThreads) {
   spalpha.clear();
   const int M = X.n();
   const int K = D.n();
   Matrix<T> vM;
   Matrix<int> rM;
   vM.resize(L,M);
   rM.resize(L,M);

   if (L <= 0) return;

   int NUM_THREADS=init_omp(numThreads);

   ProdMatrix<T> G(D,K < 25000 && M > 10);
   G.addDiag(MAX(lambda2,1e-10));

   Vector<T>* DtRT=new Vector<T>[NUM_THREADS];
   Vector<T>* uT=new Vector<T>[NUM_THREADS];
   Vector<T>* XmaskT=new Vector<T>[NUM_THREADS];
   Matrix<T>* GsT=new Matrix<T>[NUM_THREADS];
   ProdMatrix<T>* GT=new ProdMatrix<T>[NUM_THREADS];
   Matrix<T>* DmaskT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GaT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* invGsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* workT=new Matrix<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      DmaskT[i].resize(D.m(),D.n());
      DtRT[i].resize(K);
      uT[i].resize(K);
      XmaskT[i].resize(X.m());
      uT[i].setZeros();
      GsT[i].resize(L,L);
      invGsT[i].resize(L,L);
      GaT[i].resize(K,L);
      workT[i].resize(K,3);
      workT[i].setZeros();
   }

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      Vector<T> Xi;
      X.refCol(i,Xi);
      Vector<bool> maski;
      mask.refCol(i,maski);
      Vector<int> ind;
      rM.refCol(i,ind);
      Vector<T> coeffs;
      vM.refCol(i,coeffs);
      Vector<T>& DtR=DtRT[numT];

      if (maski.allfalse()) continue;
      if (maski.alltrue()) {
         T normX = Xi.nrm2sq();
         D.multTrans(Xi,DtR);
         coreLARS2(DtR,G,GsT[numT],GaT[numT],invGsT[numT],uT[numT],coeffs,
               ind,workT[numT],normX,mode,constraint,pos);
      } else {
         D.copyMask(DmaskT[numT],maski);
         Xi.copyMask(XmaskT[numT],maski);
         T constraint_mask = mode == PENALTY || mode == L2ERROR ? constraint*XmaskT[numT].n()/Xi.n() : constraint;
         T normX = XmaskT[numT].nrm2sq();
         DmaskT[numT].multTrans(XmaskT[numT],DtR);
         GT[numT].setMatrices(DmaskT[numT],false);
         GT[numT].addDiag(MAX(lambda2,T(1e-10)));
         coreLARS2(DtR,GT[numT],
               GsT[numT],GaT[numT],invGsT[numT],uT[numT],coeffs,
               ind,workT[numT],normX,mode,constraint_mask,pos);
         DmaskT[numT].setm(D.m());
         DmaskT[numT].setn(D.n());
         XmaskT[numT].setn(X.m());
      }
   }

   delete[](GT);
   delete[](XmaskT);
   delete[](DmaskT);
   delete[](DtRT);
   delete[](uT);
   delete[](GsT);
   delete[](GaT);
   delete[](invGsT);
   delete[](workT);

   /// convert the sparse matrix into a proper format
   spalpha.convert(vM,rM,K);

};

template <typename T>
void lasso2(const Matrix<T>& X, const Matrix<T>& D, SpMatrix<T>& spalpha, 
      int L, const T constraint, const T lambda2, constraint_type mode, const bool pos,
      const int numThreads, Matrix<T>* path, int length_path) {
   ProdMatrix<T> G(D,X.n() > 10 && D.n() < 50000);
   ProdMatrix<T> DtX(D,X,false);
   G.addDiag(MAX(lambda2,1e-10));
   lasso2(X,G,DtX,spalpha,L,constraint,mode,pos,numThreads,path, length_path);
}


template <typename T>
void lasso2(const Data<T>& X, const AbstractMatrix<T>& G, const AbstractMatrix<T>& DtX,
      SpMatrix<T>& spalpha, 
      int L, const T constraint, constraint_type mode, const bool pos,
      const int numThreads, Matrix<T>* path, int length_path) {
   spalpha.clear();
   const int M = X.n();
   const int K = G.n();
   Matrix<T> vM;
   Matrix<int> rM;
   vM.resize(L,M);
   rM.resize(L,M);

   if (L <= 0) return;
   if (path) path->setZeros();

   int NUM_THREADS=init_omp(numThreads);

   Vector<T>* DtRT=new Vector<T>[NUM_THREADS];
   Vector<T>* uT=new Vector<T>[NUM_THREADS];
   Matrix<T>* GsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* GaT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* invGsT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* workT=new Matrix<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      DtRT[i].resize(K);
      uT[i].resize(K);
      uT[i].setZeros();
      GsT[i].resize(L,L);
      invGsT[i].resize(L,L);
      GaT[i].resize(K,L);
      workT[i].resize(K,3);
      workT[i].setZeros();
   }
   int i;
   Vector<T> norms;
   X.norm_2sq_cols(norms);
#pragma omp parallel for private(i) 
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
    //  Vector<T> Xi;
    //  X.refCol(i,Xi);
    //  T normX = Xi.nrm2sq();
      T normX = norms[i];

      Vector<int> ind;
      rM.refCol(i,ind);
      Vector<T> coeffs;
      vM.refCol(i,coeffs);

      Vector<T>& DtR=DtRT[numT];
      DtX.copyCol(i,DtR);
      //D.multTrans(Xi,DtR);
      coreLARS2(DtR,G,GsT[numT],GaT[numT],invGsT[numT],
            uT[numT],coeffs,
            ind,workT[numT],normX,mode,constraint,pos,
            path && i==0 ? path->rawX() : NULL,length_path);
   }

   delete[](DtRT);
   delete[](uT);
   delete[](GsT);
   delete[](GaT);
   delete[](invGsT);
   delete[](workT);

   /// convert the sparse matrix into a proper format
   spalpha.convert(vM,rM,K);
};



/// Auxiliary function for lasso 
template <typename T>
void coreLARS2(Vector<T>& DtR, const AbstractMatrix<T>& G,
      Matrix<T>& Gs,
      Matrix<T>& Ga,
      Matrix<T>& invGs,
      Vector<T>& u,
      Vector<T>& coeffs,
      Vector<int>& ind,
      Matrix<T>& work,
      T& normX,
      const constraint_type mode,
      const T constraint,
      const bool pos,
      T* path, int length_path) {
   const int LL = Gs.n();
   const int K = G.n();
   const int L = MIN(LL,K);
   if (length_path <= 1) length_path=4*L;

   coeffs.setZeros();
   ind.set(-1);

   T* const pr_Gs = Gs.rawX();
   T* const pr_invGs = invGs.rawX();
   T* const pr_Ga = Ga.rawX();
   T* const pr_work = work.rawX();
   T* const pr_u = u.rawX();
   T* const pr_DtR = DtR.rawX();
   T* const pr_coeffs = coeffs.rawX();
   int* const pr_ind = ind.rawX();

   // Find the most correlated element
   int currentInd = pos ? DtR.max() : DtR.fmax();
   if (mode == PENALTY && abs(DtR[currentInd]) < constraint) return;
   if (mode == L2ERROR && normX < constraint) return;
   bool newAtom=true;

   int i;
   int iter=0;
   T thrs = 0;
   for (i = 0; i<L; ++i) {
      ++iter;
      if (newAtom) {
         pr_ind[i]=currentInd;
     //    cerr << "Add " << currentInd << endl;
         G.extract_rawCol(pr_ind[i],pr_Ga+i*K);
         for (int j = 0; j<=i; ++j)
            pr_Gs[i*LL+j]=pr_Ga[i*K+pr_ind[j]];

         // Update inverse of Gs
         if (i == 0) {
            pr_invGs[0]=T(1.0)/pr_Gs[0];
         } else {
            cblas_symv<T>(CblasColMajor,CblasUpper,i,T(1.0),
                  pr_invGs,LL,pr_Gs+i*LL,1,T(0.0),pr_u,1);
            const T schur =
               T(1.0)/(pr_Gs[i*LL+i]-cblas_dot<T>(i,pr_u,1,pr_Gs+i*LL,1));
            pr_invGs[i*LL+i]=schur;
            cblas_copy<T>(i,pr_u,1,pr_invGs+i*LL,1);
            cblas_scal<T>(i,-schur,pr_invGs+i*LL,1);
            cblas_syr<T>(CblasColMajor,CblasUpper,i,schur,pr_u,1,
                  pr_invGs,LL);
         }
      }

      // Compute the path direction 
      for (int j = 0; j<=i; ++j)
         pr_work[j]= pr_DtR[pr_ind[j]] > 0 ? T(1.0) : T(-1.0);
      cblas_symv<T>(CblasColMajor,CblasUpper,i+1,T(1.0),pr_invGs,LL,
            pr_work,1,T(0.0),pr_u,1);

      // Compute the step on the path
      T step_max = INFINITY;
      int first_zero = -1;
      for (int j = 0; j<=i; ++j) {
         T ratio = -pr_coeffs[j]/pr_u[j];
         if (ratio > 0 && ratio <= step_max) {
            step_max=ratio;
            first_zero=j;
         }
      }
 //     PRINT_F(step_max)

      T current_correlation = abs<T>(pr_DtR[pr_ind[0]]);
      cblas_gemv<T>(CblasColMajor,CblasNoTrans,K,i+1,T(1.0),pr_Ga,
            K,pr_u,1,T(0.0),pr_work+2*K,1);
      cblas_copy<T>(K,pr_work+2*K,1,pr_work+K,1);
      cblas_copy<T>(K,pr_work+2*K,1,pr_work,1);

     for (int j = 0; j<=i; ++j) {
         pr_work[pr_ind[j]]=INFINITY;
         pr_work[pr_ind[j]+K]=INFINITY;
      }
      for (int j = 0; j<K; ++j) {
         pr_work[j] = ((pr_work[j] < INFINITY) && (pr_work[j] > T(-1.0))) ? (pr_DtR[j]+current_correlation)/(T(1.0)+pr_work[j]) : INFINITY;
      }
 //     work.print("work");
      for (int j = 0; j<K; ++j) {
         pr_work[j+K] = ((pr_work[j+K] < INFINITY) && (pr_work[j+K] < T(1.0))) ? (current_correlation-pr_DtR[j])/(T(1.0)-pr_work[j+K]) : INFINITY;
      }
 //     work.print("work");

      if (pos) {
         for (int j = 0; j<K; ++j) {
            pr_work[j]=INFINITY;
         }
      }
 //     work.print("work");
 //     coeffs.print("coeffs");
      int index = cblas_iamin<T>(2*K,pr_work,1);
      T step = pr_work[index];

      // Choose next element
      currentInd = index % K;

      // compute the coefficients of the polynome representing normX^2
      T coeff1 = 0;
      for (int j = 0; j<=i; ++j)
         coeff1 += pr_DtR[pr_ind[j]] > 0 ? pr_u[j] : -pr_u[j];
      T coeff2 = 0;
      for (int j = 0; j<=i; ++j)
         coeff2 += pr_DtR[pr_ind[j]]*pr_u[j];
      T coeff3 = normX-constraint;


      T step_max2;
      if (mode == PENALTY) {
         step_max2 = current_correlation-constraint;
      } else if (mode == L2ERROR) {
         /// L2ERROR
         const T delta = coeff2*coeff2-coeff1*coeff3;
         step_max2 = delta < 0 ? INFINITY : (coeff2-sqrt(delta))/coeff1;
         step_max2 = MIN(current_correlation,step_max2);
      } else {
         /// L1COEFFS
         step_max2 = coeff1 < 0 ? INFINITY : (constraint-thrs)/coeff1;
         step_max2 = MIN(current_correlation,step_max2);
      }
      step = MIN(MIN(step,step_max2),step_max);
      if (step == INFINITY) break; // stop the path

      // Update coefficients
      cblas_axpy<T>(i+1,step,pr_u,1,pr_coeffs,1);

      if (pos) {
         for (int j = 0; j<i+1; ++j)
            if (pr_coeffs[j] < 0) pr_coeffs[j]=0;
      }

      // Update correlations
      cblas_axpy<T>(K,-step,pr_work+2*K,1,pr_DtR,1);

      // Update normX
      normX += coeff1*step*step-2*coeff2*step;

      // Update norm1
      thrs += step*coeff1;

      if (path) {
         for (int k = 0; k<=i; ++k) 
            path[iter*K+ind[k]]=pr_coeffs[k];
      }

      // Choose next action

      if (step == step_max) {
      //   cerr << "Remove " << pr_ind[first_zero] << endl;
         /// Downdate, remove first_zero
         /// Downdate Ga, Gs, invGs, ind, coeffs
         for (int j = first_zero; j<i; ++j) {
            cblas_copy<T>(K,pr_Ga+(j+1)*K,1,pr_Ga+j*K,1);
            pr_ind[j]=pr_ind[j+1];
            pr_coeffs[j]=pr_coeffs[j+1];
         }
         pr_ind[i]=-1;
         pr_coeffs[i]=0;
         for (int j = first_zero; j<i; ++j) {
            cblas_copy<T>(first_zero,pr_Gs+(j+1)*LL,1,pr_Gs+j*LL,1);
            cblas_copy<T>(i-first_zero,pr_Gs+(j+1)*LL+first_zero+1,1,
                  pr_Gs+j*LL+first_zero,1);
         }
         const T schur = pr_invGs[first_zero*LL+first_zero];
         cblas_copy<T>(first_zero,pr_invGs+first_zero*LL,1,pr_u,1);
         cblas_copy<T>(i-first_zero,pr_invGs+(first_zero+1)*LL+first_zero,LL,
               pr_u+first_zero,1);
         for (int j = first_zero; j<i; ++j) {
            cblas_copy<T>(first_zero,pr_invGs+(j+1)*LL,1,pr_invGs+j*LL,1);
            cblas_copy<T>(i-first_zero,pr_invGs+(j+1)*LL+first_zero+1,1,
                  pr_invGs+j*LL+first_zero,1);
         }
         cblas_syr<T>(CblasColMajor,CblasUpper,i,T(-1.0)/schur,
               pr_u,1,pr_invGs,LL);
         newAtom=false;
         i=i-2;
      } else {
         newAtom=true;
      }
      if ((iter >= length_path-1) || abs(step) < 1e-15 ||
            step == step_max2 || (normX < 1e-15) ||
            (i == (L-1)) ||
            (mode == L2ERROR && normX - constraint < 1e-15) ||
            (mode == L1COEFFS && (constraint-thrs < 1e-15))) {
         break;
      }
   }
}

/// Auxiliary function for lasso 
template <typename T>
void coreLARS2W(Vector<T>& DtR, AbstractMatrix<T>& G,
      Matrix<T>& Gs,
      Matrix<T>& Ga,
      Matrix<T>& invGs,
      Vector<T>& u,
      Vector<T>& coeffs,
      const Vector<T>& weights,
      Vector<int>& ind,
      Matrix<T>& work,
      T& normX,
      const constraint_type mode,
      const T constraint,
      const bool pos) {
   const int LL = Gs.n();
   const int K = G.n();
   const int L = MIN(LL,K);
   coeffs.setZeros();
   ind.set(-1);

   T* const pr_Gs = Gs.rawX();
   T* const pr_invGs = invGs.rawX();
   T* const pr_Ga = Ga.rawX();
   //  T* const pr_G = G.rawX();
   T* const pr_work = work.rawX();
   T* const pr_u = u.rawX();
   T* const pr_DtR = DtR.rawX();
   T* const pr_coeffs = coeffs.rawX();
   T* const pr_weights = weights.rawX();
   int* const pr_ind = ind.rawX();

   DtR.div(weights);

   // Find the most correlated element
   int currentInd = pos ? DtR.max() : DtR.fmax();
   if (mode == PENALTY && abs(DtR[currentInd]) < constraint) return;
   if (mode == L2ERROR && normX < constraint) return;
   bool newAtom=true;

   int i;
   int iter=0;
   T thrs = 0;
   for (i = 0; i<L; ++i) {
      ++iter;
      if (newAtom) {
         pr_ind[i]=currentInd;
         // Update upper part of Gs and Ga
         G.extract_rawCol(pr_ind[i],pr_Ga+i*K);
         for (int j = 0; j<=i; ++j)
            pr_Gs[i*LL+j]=pr_Ga[i*K+pr_ind[j]];

         // Update inverse of Gs
         if (i == 0) {
            pr_invGs[0]=T(1.0)/pr_Gs[0];
         } else {
            cblas_symv<T>(CblasColMajor,CblasUpper,i,T(1.0),
                  pr_invGs,LL,pr_Gs+i*LL,1,T(0.0),pr_u,1);
            const T schur =
               T(1.0)/(pr_Gs[i*LL+i]-cblas_dot<T>(i,pr_u,1,pr_Gs+i*LL,1));
            pr_invGs[i*LL+i]=schur;
            cblas_copy<T>(i,pr_u,1,pr_invGs+i*LL,1);
            cblas_scal<T>(i,-schur,pr_invGs+i*LL,1);
            cblas_syr<T>(CblasColMajor,CblasUpper,i,schur,pr_u,1,
                  pr_invGs,LL);
         }
      }

      // Compute the path direction 
      for (int j = 0; j<=i; ++j)
         pr_work[j]= pr_DtR[pr_ind[j]] > 0 ? weights[pr_ind[j]] : -weights[pr_ind[j]];
      cblas_symv<T>(CblasColMajor,CblasUpper,i+1,T(1.0),pr_invGs,LL,
            pr_work,1,T(0.0),pr_u,1);

      // Compute the step on the path
      T step_max = INFINITY;
      int first_zero = -1;
      for (int j = 0; j<=i; ++j) {
         T ratio = -pr_coeffs[j]/pr_u[j];
         if (ratio > 0 && ratio <= step_max) {
            step_max=ratio;
            first_zero=j;
         }
      }

      T current_correlation = abs<T>(pr_DtR[pr_ind[0]]);
      cblas_gemv<T>(CblasColMajor,CblasNoTrans,K,i+1,T(1.0),pr_Ga,
            K,pr_u,1,T(0.0),pr_work+2*K,1);
      vDiv<T>(K,pr_work+2*K,pr_weights,pr_work+2*K);
      cblas_copy<T>(K,pr_work+2*K,1,pr_work+K,1);
      cblas_copy<T>(K,pr_work+2*K,1,pr_work,1);

     for (int j = 0; j<=i; ++j) {
         pr_work[pr_ind[j]]=INFINITY;
         pr_work[pr_ind[j]+K]=INFINITY;
      }
      for (int j = 0; j<K; ++j) {
         pr_work[j] = ((pr_work[j] < INFINITY) && (pr_work[j] > T(-1.0))) ? (pr_DtR[j]+current_correlation)/(T(1.0)+pr_work[j]) : INFINITY;
      }
      for (int j = 0; j<K; ++j) {
         pr_work[j+K] = ((pr_work[j+K] < INFINITY) && (pr_work[j+K] < T(1.0))) ? (current_correlation-pr_DtR[j])/(T(1.0)-pr_work[j+K]) : INFINITY;
      }

      if (pos) {
         for (int j = 0; j<K; ++j) {
            pr_work[j]=INFINITY;
         }
      }
      int index = cblas_iamin<T>(2*K,pr_work,1);
      T step = pr_work[index];
      // Choose next element
      currentInd = index % K;

      // compute the coefficients of the polynome representing normX^2
      T coeff1 = 0;
      for (int j = 0; j<=i; ++j)
         coeff1 += pr_DtR[pr_ind[j]] > 0 ? pr_weights[pr_ind[j]]*pr_u[j] : 
            -pr_weights[pr_ind[j]]*pr_u[j];
      T coeff2 = 0;
      for (int j = 0; j<=i; ++j)
         coeff2 += pr_DtR[pr_ind[j]]*pr_u[j]*pr_weights[pr_ind[j]];
      T coeff3 = normX-constraint;

      T step_max2;
      if (mode == PENALTY) {
         step_max2 = current_correlation-constraint;
      } else if (mode == L2ERROR) {
         /// L2ERROR
         const T delta = coeff2*coeff2-coeff1*coeff3;
         step_max2 = delta < 0 ? INFINITY : (coeff2-sqrt(delta))/coeff1;
      } else {
         /// L1COEFFS
         step_max2 = coeff1 < 0 ? INFINITY : (constraint-thrs)/coeff1;
      }
      step = MIN(MIN(step,step_max2),step_max);

      if (step == INFINITY) break; // stop the path

      // Update coefficients
      cblas_axpy<T>(i+1,step,pr_u,1,pr_coeffs,1);

      // Update correlations
      cblas_axpy<T>(K,-step,pr_work+2*K,1,pr_DtR,1);

      // Update normX
      normX += coeff1*step*step-2*coeff2*step;

      // Update norm1
      thrs += step*coeff1;

      if (step == step_max) {
         /// Downdate, remove first_zero
         /// Downdate Ga, Gs, invGs, ind, coeffs
         for (int j = first_zero; j<i; ++j) {
            cblas_copy<T>(K,pr_Ga+(j+1)*K,1,pr_Ga+j*K,1);
            pr_ind[j]=pr_ind[j+1];
            pr_coeffs[j]=pr_coeffs[j+1];
         }
         pr_ind[i]=-1;
         pr_coeffs[i]=0;
         for (int j = first_zero; j<i; ++j) {
            cblas_copy<T>(first_zero,pr_Gs+(j+1)*LL,1,pr_Gs+j*LL,1);
            cblas_copy<T>(i-first_zero,pr_Gs+(j+1)*LL+first_zero+1,1,
                  pr_Gs+j*LL+first_zero,1);
         }
         const T schur = pr_invGs[first_zero*LL+first_zero];
         cblas_copy<T>(first_zero,pr_invGs+first_zero*LL,1,pr_u,1);
         cblas_copy<T>(i-first_zero,pr_invGs+(first_zero+1)*LL+first_zero,LL,
               pr_u+first_zero,1);
         for (int j = first_zero; j<i; ++j) {
            cblas_copy<T>(first_zero,pr_invGs+(j+1)*LL,1,pr_invGs+j*LL,1);
            cblas_copy<T>(i-first_zero,pr_invGs+(j+1)*LL+first_zero+1,1,
                  pr_invGs+j*LL+first_zero,1);
         }
         cblas_syr<T>(CblasColMajor,CblasUpper,i,T(-1.0)/schur,
               pr_u,1,pr_invGs,LL);
         newAtom=false;
         i=i-2;
      } else {
         newAtom=true;
      }
      // Choose next action
      if (iter > 4*L || abs(step) < 1e-10 ||
            step == step_max2 || (normX < 1e-10) ||
            (i == (L-1)) ||
            (mode == L2ERROR && normX - constraint < 1e-10) ||
            (mode == L1COEFFS && (constraint-thrs < 1e-10))) {
         break;
      }
   }
}



/* ************************
 * Iterative thresholding
 * ************************/

/// Implementation of IST for solving
/// \forall i, \min_{\alpha_i} ||\alpha_i||_1 
///                        s.t. ||\X_i-D\alpha_i||_2^2 <= constraint or
/// \forall i, \min_{\alpha_i} constraint*||\alpha_i||_1 + ...
///                        ... ||\X_i-D\alpha_i||_2^2 <= lambda 
template <typename T>
void ist(const Matrix<T>& X, const Matrix<T>& D, 
      SpMatrix<T>& spalpha, T lambda, constraint_type mode,
      const int itermax, 
      const T tol,
      const int numThreads) {
   Matrix<T> alpha;
   spalpha.toFull(alpha);
   spalpha.clear();
   ist(X,D,alpha,lambda,mode,itermax,tol,numThreads);
   alpha.toSparse(spalpha);
}

template <typename T>
void ist(const Matrix<T>& X, const Matrix<T>& D, 
      Matrix<T>& alpha, T lambda, constraint_type mode,
      const int itermax, 
      const T tol, const int numThreads) {

   if (mode == L1COEFFS) {
      std::cerr << "Mode not implemented" << std::endl;
      return;
   }

   int K=D.n();
   int M=X.n();
   alpha.resize(K,M);
   if (!D.isNormalized()) {
      cerr << "Current implementation of IST does not support non-normalized dictionaries" << endl;
      return;
   }

   /// compute the Gram Matrix G=D'D
   //CachedProdMatrix<T> G(D, K < 20000 && M*K/10 > K);
   //ProdMatrix<T> G(D, K < 20000 && M*K/10 > K);
   Matrix<T> G;
   D.XtX(G);
   // for (int i = 0; i<K; ++i) G[i*K+i] += 1e-6;
   G.addDiag(1e-12);
   ProdMatrix<T> DtX(D,X,false);

   int NUM_THREADS=init_omp(numThreads);

   Vector<T>* DtRT= new Vector<T>[NUM_THREADS];
   SpVector<T>* spAlphaT= new SpVector<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      DtRT[i].resize(K);
      spAlphaT[i].resize(K);
   };

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      Vector<T> coeffs;
      alpha.refCol(i,coeffs);
      Vector<T>& DtR=DtRT[numT];
      SpVector<T>& spAlpha=spAlphaT[numT];
      T norm1 = coeffs.asum();
      // Compute DtR
      DtX.copyCol(i,DtR);
      Vector<T> Xi;
      X.refCol(i,Xi);
      T normX2 = Xi.nrm2sq(); 

      if (norm1 > EPSILON) {
         coeffs.toSparse(spAlpha);
         G.mult(spAlpha,DtR,-1.0,1.0);
      }

      if (mode == PENALTY) {
         coreIST(G,DtR,coeffs,lambda,itermax,tol);
      } else {
         coreISTconstrained(G,DtR,coeffs,normX2,lambda,itermax,tol);
      }
   } 

   delete[](DtRT);
   delete[](spAlphaT);

}

template <typename T>
inline void coreIST(const AbstractMatrix<T>& G, Vector<T>& DtRv, Vector<T>& coeffsv,
      const T thrs, const int itermax, 
      const T tol) {

   const int K = G.n();
   T* const coeffs = coeffsv.rawX();
   T* const DtR = DtRv.rawX();
   //  T* const prG = G.rawX();

   const T lambda_init=thrs;
   T maxDtR = DtRv.fmaxval();
   T norm1=coeffsv.asum();
   T lambda=lambda_init;
   vAdd(K,DtR,coeffs,DtR);

   for (int iter=0; iter < itermax; ++iter) {
      for (int j = 0; j <K; ++j) {
         if (DtR[j] > lambda) {
            T diff=coeffs[j];
            coeffs[j]=DtR[j]-lambda;
            diff-=coeffs[j];
            DtR[j]-=diff;
            G.add_rawCol(j,DtR,diff);
            //cblas_axpy(K,diff,prG+j*K,1,DtR,1);
         } else if (DtR[j] < -lambda) {
            T diff=coeffs[j];
            coeffs[j]=DtR[j]+lambda;
            diff-=coeffs[j];
            DtR[j]-=diff;
            G.add_rawCol(j,DtR,diff);
            //cblas_axpy(K,diff,prG+j*K,1,DtR,1);
         } else if (coeffs[j]) {
            T diff=coeffs[j];
            coeffs[j]=T();
            DtR[j]-=diff;
            G.add_rawCol(j,DtR,diff);
            //cblas_axpy(K,diff,prG+j*K,1,DtR,1);
         }
      }
      if (iter % 5 == 1) {
         vSub(K,DtR,coeffs,DtR);         
         maxDtR = DtRv.fmaxval();
         norm1 =T();
         T DtRa = T();
         for (int j = 0; j<K; ++j) {
            if (coeffs[j]) {
               norm1 += abs(coeffs[j]);
               DtRa += DtR[j]*coeffs[j];
            }
         }
         vAdd(K,DtR,coeffs,DtR);         
         const T kappa = -DtRa+norm1*maxDtR;
         if (abs(lambda - maxDtR) < tol && kappa <= tol)
            break;
      }
   }
}


/// coreIST constrained
template <typename T>
void coreISTconstrained(const AbstractMatrix<T>& G, Vector<T>& DtRv, Vector<T>&
      coeffsv, const T normX2, const T eps, const int itermax, const T tol) {
   const int K = G.n();
   T* const coeffs = coeffsv.rawX();
   T* const DtR = DtRv.rawX();
   // T* const prG = G.rawX();
   T err = normX2;

   T norm1 = coeffsv.asum();
   if (!norm1 && err <= eps) return;
   T current_tol = 10.0*tol;
   T maxDtR = DtRv.fmaxval();
   T lambda = maxDtR;
   T lambdasq= lambda*lambda;
   if (!norm1) {
      lambdasq *= eps/err;
      lambda=sqrt(lambdasq);
   }

   Vector<int> indices(K);
   indices.set(-1);
   int* const pr_indices=indices.rawX();
   int count;

   for (int iter=0; iter < itermax; ++iter) {

      count=0;
      T old_err = err;
      for (int j = 0; j <K; ++j) {

         // Soft-thresholding
         T old_coeff = coeffs[j];
         T diff = DtR[j]+old_coeff;
         if (diff > lambda) {
            coeffs[j] = diff - lambda;
            err+=lambdasq-DtR[j]*DtR[j];
            pr_indices[count++]=j;
         } else if (diff < - lambda) {
            coeffs[j] = diff + lambda;
            err+=lambdasq-DtR[j]*DtR[j];
            pr_indices[count++]=j;
         } else {
            coeffs[j]=T();
            if (old_coeff) {
               err+=diff*diff-DtR[j]*DtR[j];
            }
         }
         // Update DtR
         diff = old_coeff-coeffs[j];
         if (diff) {
            G.add_rawCol(j,DtR,diff);
            //cblas_axpy<T>(K,old_coeff-coeffs[j],prG+j*K,1,DtR,1);
         }
      }

      maxDtR = DtRv.fmaxval();
      norm1 =T();
      T DtRa = T();
      for (int j = 0; j<count; ++j) {
         const int ind = pr_indices[j];
         norm1 += abs(coeffs[ind]);
         DtRa += DtR[ind]*coeffs[ind];
      }
      if (norm1-DtRa/maxDtR <= current_tol) {
         const bool change = ((old_err > eps) && err < eps+current_tol) ||
            (old_err < eps && err > eps-current_tol);
         if (change) {
            if (current_tol == tol) {
               break;
            } else {
               current_tol = MAX(current_tol*0.5,tol);
            }
         }
         lambdasq *= eps/err;
         lambda=sqrt(lambdasq);
      }
   }
};



/// ist for group Lasso
template <typename T>
void ist_groupLasso(const Matrix<T>* XT, const Matrix<T>& D,
      Matrix<T>* alphaT, const int Ngroups, 
      const T lambda, const constraint_type mode,
      const int itermax,
      const T tol, const int numThreads) {
   int K=D.n();
   int n = D.m();

   if (!D.isNormalized()) {
      cerr << "Current implementation of block coordinate descent does not support non-normalized dictionaries" << endl;
      return;
   }

   if (mode == L1COEFFS) {
      std::cerr << "Mode not implemented" << std::endl;
      return;
   }


   /// compute the Gram Matrix G=D'D
   Matrix<T> G;
   D.XtX(G);

   int NUM_THREADS=init_omp(numThreads);

   Matrix<T>* RtDT = new Matrix<T>[NUM_THREADS];
   Matrix<T>* alphatT = new Matrix<T>[NUM_THREADS];

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< Ngroups; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      const Matrix<T>& X = XT[i];
      int M = X.n();
      Matrix<T>& alphat = alphatT[numT];
      alphaT[i].transpose(alphat);
      Matrix<T>& RtD = RtDT[numT];
      X.mult(D,RtD,true,false);


      Vector<T> col, col2;
      T norm1 = alphat.asum();
      T normX2;

      if (!norm1) {
         Vector<T> DtR_mean(K);
         Vector<T> coeffs_mean(K);
         coeffs_mean.setZeros();
         RtD.meanRow(DtR_mean);
         coeffs_mean.setZeros();
         if (mode == PENALTY) {
            coreIST(G,DtR_mean,coeffs_mean,lambda/T(2.0),itermax,tol);
         } else {
            Vector<T> meanVec(n);
            X.meanCol(meanVec);
            normX2=meanVec.nrm2sq(); 
            coreISTconstrained(G,DtR_mean,coeffs_mean,normX2,
                  lambda,itermax,tol);
            SpVector<T> spalpha(K);
            normX2-=computeError(normX2,G,DtR_mean,coeffs_mean,spalpha);
            normX2=X.normFsq()-M*normX2;
         }
         alphat.fillRow(coeffs_mean);         
      }

      if (M > 1) {
         for (int j = 0; j<K; ++j) {
            alphat.refCol(j,col);
            const T nrm=col.nrm2sq();
            if (nrm) {
               G.refCol(j,col2);
               RtD.rank1Update(col,col2,T(-1.0));
            }
         }

         if (mode == PENALTY) {
            coreGroupIST(G,RtD,alphat,sqr<T>(M)*lambda/T(2.0),itermax,sqr<T>(M)*tol);
         } else  {
            coreGroupISTConstrained(G,RtD,alphat,normX2,M*lambda,itermax,sqr<T>(M)*tol);
         }
      }
      alphat.transpose(alphaT[i]);
   }

   delete[](RtDT);
   delete[](alphatT);
};


template <typename T>
void coreGroupIST(const Matrix<T>& G, Matrix<T>& RtDm,
      Matrix<T>& coeffsm,
      const T thrs,
      const int itermax,
      const T tol) {
   const int K = G.n();
   const int M = RtDm.m();
   T* const prG = G.rawX();
   T* const RtD = RtDm.rawX();
   T* const coeffs = coeffsm.rawX();

   const T lambda_init=thrs;
   T lambda=lambda_init;

   Vector<T> old_coeffv(M);
   T* const old_coeff = old_coeffv.rawX();
   Vector<T> normsv(K);
   T* const norms = normsv.rawX();
   coeffsm.norm_2_cols(normsv);
   Vector<T> normRtDv(K);

   Vector<int> activatev(K);
   activatev.set(3);
   int* const activate=activatev.rawX();

   for (int iter=0; iter < itermax; ++iter) {
      for (int j = 0; j <K; ++j) {
         if (activate[j] >= 0) {
            if (norms[j]) {
               cblas_copy(M,coeffs+j*M,1,old_coeff,1);
               vAdd(M,coeffs+j*M,RtD+j*M,coeffs+j*M);
               const T nrm = cblas_nrm2(M,coeffs+j*M,1);
               if (nrm > lambda) {
                  norms[j]=nrm-lambda;
                  cblas_scal(M,norms[j]/nrm,coeffs+j*M,1);
                  vSub(M,old_coeff,coeffs+j*M,old_coeff);
                  cblas_ger(CblasColMajor,M,K,T(1.0),old_coeff,1,prG+j*K,1,RtD,M);
                  activate[j]=5;
               } else {
                  memset(coeffs+j*M,0,M*sizeof(T)); 
                  norms[j]=T();
                  cblas_ger(CblasColMajor,M,K,T(1.0),old_coeff,1,prG+j*K,1,RtD,M);
                  --activate[j];
               }
            } else {
               cblas_copy(M,RtD+j*M,1,old_coeff,1);
               const T nrm = cblas_nrm2(M,old_coeff,1);
               if (nrm > lambda) {
                  norms[j]=nrm-lambda;
                  cblas_copy(M,old_coeff,1,coeffs+j*M,1);
                  cblas_scal(M,norms[j]/nrm,coeffs+j*M,1);
                  cblas_ger(CblasColMajor,M,K,T(-1.0),coeffs+j*M,1,prG+j*K,1,RtD,M);
                  activate[j]=5;
               } else {
                  activate[j] = (activate[j] == 0) ? -10 : activate[j]-1;
               }
            }
         } else {
            ++activate[j];
         }
      }

      if (iter % 5 == 4) {
         T norm1=normsv.asum();
         RtDm.norm_2sq_cols(normRtDv);
         T maxDtR = sqr(normRtDv.maxval());
         T DtRa=T();
         for (int j = 0; j<K; ++j) {
            if (norms[j]) {
               DtRa += cblas_dot(M,coeffs+j*M,1,RtD+j*M,1);
            }
         }
         if ((maxDtR - lambda) < (tol*maxDtR/norm1) && norm1-DtRa/maxDtR < tol) break;
      }
   }
};


/// Auxiliary function for ist_groupLasso
template <typename T>
void coreGroupISTConstrained(const Matrix<T>& G, Matrix<T>& RtDm,
      Matrix<T>& coeffsm, const T normR,
      const T eps,
      const int itermax,
      const T tol) {
   const int K = G.n();
   const int M = RtDm.m();
   T* const prG = G.rawX();
   T* const RtD = RtDm.rawX();
   T* const coeffs = coeffsm.rawX();

   T err = normR;

   Vector<T> old_coeffv(M);
   T* const old_coeff = old_coeffv.rawX();
   Vector<T> normsv(K);
   T* const norms = normsv.rawX();
   coeffsm.norm_2_cols(normsv);
   Vector<T> normRtDv(K);
   RtDm.norm_2sq_cols(normRtDv);

   Vector<int> activatev(K);
   activatev.set(3);
   int* const activate=activatev.rawX();

   T norm1 = normsv.sum();
   if (!norm1 && err <= eps) return;
   T current_tol = 10.0*tol;

   T maxDtR = sqr(normRtDv.maxval());
   T lambda = maxDtR;
   T lambdasq= lambda*lambda;

   if (!norm1) {
      lambdasq *= eps/err;
      lambda=sqrt(lambdasq);
   }

   for (int iter=0; iter < itermax; ++iter) {

      T old_err = err;
      for (int j = 0; j <K; ++j) {
         if (activate[j] >= 0) {
            if (norms[j]) {
               cblas_copy(M,coeffs+j*M,1,old_coeff,1);
               vAdd(M,coeffs+j*M,RtD+j*M,coeffs+j*M);
               const T nrm = cblas_nrm2(M,coeffs+j*M,1);
               if (nrm > lambda) {
                  norms[j]=nrm-lambda;
                  cblas_scal(M,norms[j]/nrm,coeffs+j*M,1);
                  vSub(M,old_coeff,coeffs+j*M,old_coeff);
                  err += cblas_dot(M,old_coeff,1,old_coeff,1)
                     +2*cblas_dot(M,old_coeff,1,RtD+j*M,1);
                  cblas_ger(CblasColMajor,M,K,T(1.0),old_coeff,1,prG+j*K,1,RtD,M);
                  activate[j]=3;
               } else {
                  memset(coeffs+j*M,0,M*sizeof(T)); 
                  norms[j]=T();
                  err += cblas_dot(M,old_coeff,1,old_coeff,1)
                     +2*cblas_dot(M,old_coeff,1,RtD+j*M,1);
                  cblas_ger(CblasColMajor,M,K,T(1.0),old_coeff,1,prG+j*K,1,RtD,M);
                  --activate[j];
               }
            } else {
               cblas_copy(M,RtD+j*M,1,old_coeff,1);
               const T nrm = cblas_nrm2(M,old_coeff,1);
               if (nrm > lambda) {
                  norms[j]=nrm-lambda;
                  cblas_copy(M,old_coeff,1,coeffs+j*M,1);
                  cblas_scal(M,norms[j]/nrm,coeffs+j*M,1);
                  err += cblas_dot(M,coeffs+j*M,1,coeffs+j*M,1)
                     -2*cblas_dot(M,coeffs+j*M,1,RtD+j*M,1);
                  cblas_ger(CblasColMajor,M,K,T(-1.0),coeffs+j*M,1,prG+j*K,1,RtD,M);
                  activate[j]=3;
               } else {
                  activate[j] = (activate[j] == 0) ? -3 : activate[j]-1;
               }
            }
         } else {
            ++activate[j];
         }
      }

      norm1 = normsv.sum();
      RtDm.norm_2sq_cols(normRtDv);
      maxDtR = sqr(normRtDv.maxval());
      T DtRa=T();
      for (int j = 0; j<K; ++j) {
         if (norms[j]) {
            DtRa += cblas_dot(M,coeffs+j*M,1,RtD+j*M,1);
         }
      }
      if (norm1-DtRa/maxDtR <= current_tol) {
         const T tol_bis=current_tol*maxDtR;
         const bool change = ((old_err > eps) && err < eps+tol_bis) ||
            (old_err < eps && err > eps-tol_bis);
         if (change) {
            if (current_tol == tol) {
               break;
            } else {
               current_tol = MAX(current_tol*0.5,tol);
            }
         }
         lambdasq *= eps/err;
         lambda=sqrt(lambdasq);
      }
   }
};

/// auxiliary function for ist_groupLasso
template <typename T>
T computeError(const T normX2,const Vector<T>& norms,
      const Matrix<T>& G,const Matrix<T>& RtD,const Matrix<T>& alphat) {
   T err2 = normX2;
   Vector<T> col,col2;
   for (int j = 0; j<G.n(); ++j) {
      if (norms[j] > EPSILON) {
         alphat.refCol(j,col);
         RtD.refCol(j,col2);
         err2 -= 2*col.dot(col2);
         T add = 0.0;
         for (int k = 0; k<j; ++k) {
            if (norms[k] > EPSILON) {
               alphat.refCol(k,col2);
               add -= G(j,k)*col.dot(col2);
            }
         }
         add += add - G(j,j)*col.nrm2sq();
         err2 += add;
      }
   }
   return err2;
}

/// auxiliary function for 
template <typename T>
T computeError(const T normX2,
      const Matrix<T>& G,const Vector<T>& DtR,const Vector<T>& coeffs,
      SpVector<T>& spAlpha) {
   coeffs.toSparse(spAlpha);
   return normX2 -G.quad(spAlpha)-2*DtR.dot(spAlpha);
};

/* ******************
 * Simultaneous OMP 
 * *****************/

template <typename T>
void somp(const Matrix<T>* X, const Matrix<T>& D, SpMatrix<T>* spalpha, 
      const int Ngroups, const int L, const T eps,const int numThreads) {
   somp(X,D,spalpha,Ngroups,L,&eps,false,numThreads);
}

template <typename T>
void somp(const Matrix<T>* XT, const Matrix<T>& D, SpMatrix<T>* spalphaT, 
      const int Ngroups, const int LL, const T* eps, const bool adapt,
      const int numThreads) {
   if (LL <= 0) return;
   const int K = D.n();
   const int L = MIN(D.m(),MIN(LL,K));

   if (!D.isNormalized()) {
      cerr << "Current implementation of OMP does not support non-normalized dictionaries" << endl;
      return;
   }

   /// compute the Gram Matrix G=D'D
   Matrix<T> G;
   D.XtX(G);

   int NUM_THREADS=init_omp(numThreads);

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< Ngroups; ++i) {
      const Matrix<T>& X = XT[i];
      const int M = X.n();
      SpMatrix<T>& spalpha = spalphaT[i];
      spalpha.clear();
      Vector<int> rv;
      Matrix<T> vM;
      T thrs = adapt ? eps[i] : M*(*eps);
      coreSOMP(X,D,G,vM,rv,L,thrs);
      spalpha.convert2(vM,rv,K);   
   }
}

template <typename T>
void coreSOMP(const Matrix<T>& X, const Matrix<T>& D, const Matrix<T>& G,
      Matrix<T>& v,
      Vector<int>& r, const int L, const T eps) {
   const int K = G.n();
   const int n = D.m();
   const int M = X.n();

   const bool big_mode = M*K*(n+L) > 2*(M*n*n+K*n*(n+L));
   r.resize(L);
   r.set(-1);
   v.resize(0,X.n());

   if (M == 1) {
      Vector<T> scores(K);
      Vector<T> norm(K);
      Vector<T> tmp(K);
      Matrix<T> Un(L,L);
      Un.setZeros();
      Matrix<T> Undn(K,L);
      Matrix<T> Unds(L,L);
      Matrix<T> Gs(K,L);
      Vector<T> Rdn(K);
      Vector<T> Xt(X.rawX(),n);
      D.multTrans(Xt,Rdn);
      Vector<T> RUn(L);
      T normX = Xt.nrm2sq();
      T lambda=0;
      coreORMP(scores,norm,tmp,Un,Undn,Unds,Gs,Rdn,G,r,RUn,normX,&eps,&L,&lambda);
      int count=0;
      for (int i = 0; i<L; ++i) {
         if (r[i] == -1) break;
         ++count;
      }
      v.resize(count,X.n());
      Vector<T> v1(v.rawX(),count);
      Vector<T> v2(RUn.rawX(),count);
      v1.copy(v2);
      return;
   }

   Matrix<T> XXtD;
   Matrix<T> XtD;
   T E;
   if (big_mode) {
      Matrix<T> XXt;
      X.XXt(XXt);
      E = XXt.trace();
      if (E < eps) return;
      XXt.mult(D,XXtD);
   } else {
      E=X.normFsq();
      if (E < eps) return;
      X.mult(D,XtD,true);
   }

   Matrix<T> A(K,L);
   A.setZeros();
   Matrix<T> B(L,K);
   B.setZeros();
   Matrix<T> S(L,L);
   S.setZeros();
   Matrix<T> Fs(K,L);
   Fs.setZeros();
   Matrix<T> Gs(K,L);
   Gs.setZeros();
   Matrix<T> As(L,L);
   As.setZeros();

   Vector<T> tmp(K);
   Vector<T> e(K);
   G.diag(e);
   Vector<T> f(K);
   if (big_mode) {
      for (int i = 0; i<K; ++i) {
         Vector<T> di;
         D.refCol(i,di);
         Vector<T> di2;
         XXtD.refCol(i,di2);
         f[i]=di.dot(di2);
      }
   } else {
      XtD.norm_2sq_cols(f);
   }
   Vector<T> c(L);
   c.setZeros();
   Vector<T> scores(K);

   /// permit unsafe fast low level accesses
   T* const prAs = As.rawX();
   T* const prA = A.rawX();
   T* const prS = S.rawX();
   T* const prGs = Gs.rawX();
   T* const prFs = Fs.rawX();
   T* const prB = B.rawX();
   T* const pr_c = c.rawX();
   T* const pr_tmp = tmp.rawX();

   int j;
   for (j = 0; j<L; ++j) {
      scores.copy(f);
      scores.div(e);
      for (int k = 0; k<j; ++k) scores[r[k]]=-1.0;
      const int currentInd = scores.max();
      const T invNorm=T(1.0)/sqrt(e[currentInd]);
      if (invNorm > 1e3) {
         j=j-1;
         break;
      }
      r[j]=currentInd;
      E -= scores[currentInd];
      for (int k = 0; k<j; ++k) prS[j*L+k]=T();
      prS[j*L+j]=T(1.0);
      for (int k = 0; k<j; ++k) prAs[k*L+j]=prA[k*K+currentInd];

      /// Cholesky update with partial reorthogonalization
      int iter = invNorm > 1.41 ? 2 : 1;
      for (int k = 0; k<iter; ++k) {
         for (int l = 0; l<j; ++l) {
            T scal = -cblas_dot<T>(j-l+1,prAs+l*L+l,1,prS+j*L+l,1);
            cblas_axpy<T>(l+1,scal,prS+l*L,1,prS+j*L,1);
         }
      }
      cblas_scal<T>(j+1,invNorm,prS+j*L,1);

      if (j == L-1 || E <= eps) {
         ++j;
         break;
      }

      /// Update e,f,scores,A,B,As,Bs,Fs,Gs,S,c
      /// Gs,S,A,As, e, Fs, B,c
      Vector<T> Gsj;
      Gs.refCol(j,Gsj);
      G.copyCol(currentInd,Gsj);
      cblas_gemv<T>(CblasColMajor,CblasNoTrans,K,j+1,T(1.0),prGs,K,prS+j*L,1,
            T(0.0),prA+j*K,1);
      prAs[j*L+j]=prA[j*K+currentInd];
      Vector<T> Aj;
      A.refCol(j,Aj);
      tmp.sqr(Aj);
      e.sub(tmp);

      Vector<T> Fsj;
      Fs.refCol(j,Fsj);
      if (big_mode) {
         Vector<T> di;
         D.refCol(currentInd,di);
         XXtD.multTrans(di,Fsj);
      } else {
         Vector<T> di;
         XtD.refCol(currentInd,di);
         XtD.multTrans(di,Fsj);
      }
      cblas_gemv<T>(CblasColMajor,CblasNoTrans,K,j+1,T(1.0),prFs,K,prS+j*L,1,
            T(0.0),prB+j,L);
      for (int k = 0; k<j;++k) pr_c[k]=T();
      for (int k = 0; k<=j;++k) 
         cblas_axpy<T>(j,prS[j*L+k],prB+r[k]*L,1,pr_c,1);
      f.add(tmp,f[currentInd]*invNorm*invNorm);
      if (j > 0) {
         cblas_gemv<T>(CblasColMajor,CblasNoTrans,K,j,T(1.0),prA,K,pr_c,1,
               T(0.0),pr_tmp,1);
      } else {
         tmp.setZeros();
      }
      cblas_axpy<T>(K,T(-1.0),prB+j,L,pr_tmp,1);
      tmp.mult(tmp,Aj);
      f.add(tmp,T(2.0));
   }
   A.clear();
   B.clear();
   Fs.clear();
   Gs.clear();
   As.clear();

   if (j == 0) return;

   Matrix<T> SSt;
   S.upperTriXXt(SSt,j);
   Matrix<T> Dg(n,j);
   for (int i = 0; i<j;++i) {
      Vector<T> Dgi;
      Dg.refCol(i,Dgi);
      D.copyCol(r[i],Dgi);
   }
   Matrix<T> SStDt;
   SSt.mult(Dg,SStDt,false,true);
   SStDt.mult(X,v);
};


#endif // DECOMP_H

