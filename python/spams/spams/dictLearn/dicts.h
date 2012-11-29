
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

/*!
 *
 * \file
 *                toolbox dictLearn
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File dicts.h
 * \brief Contains dictionary learning algorithms
 * It requires the toolbox decomp */

#ifndef DICTS_H
#define DICTS_H

#include <decomp.h>

char buffer_string[50];
enum constraint_type_D { L2,  L1L2, L1L2FL, L1L2MU};
enum mode_compute { AUTO, PARAM1, PARAM2, PARAM3};

template <typename T> struct ParamDictLearn {
   public:
      ParamDictLearn() : 
         mode(PENALTY),
         posAlpha(false),
         modeD(L2),
         posD(false),
         modeParam(AUTO),
         t0(1e-5),
         rho(5),
         gamma1(0),
         mu(0),
         lambda3(0),
         lambda4(0),
         lambda2(0),
         gamma2(0),
         approx(0.0),
         p(1.0),
         whiten(false),
         expand(false),
         isConstant(false),
         updateConstant(true),
         ThetaDiag(false),
         ThetaDiagPlus(false),
         ThetaId(false),
         DequalsW(false),
         weightClasses(false),
         balanceClasses(false),
         extend(false),
         pattern(false),
         stochastic(false),
         scaleW(false),
         batch(false),
         verbose(true),
         clean(true),
         log(false),
         updateD(true),
         updateW(true),
         updateTheta(true),
         logName(NULL), 
         iter_updateD(1) { };
      ~ParamDictLearn() { delete[](logName); };
      int iter;
      T lambda;
      constraint_type mode;
      bool posAlpha; 
      constraint_type_D modeD;
      bool posD;
      mode_compute modeParam;
      T t0;
      T rho;
      T gamma1;
      T mu;
      T lambda3;
      T lambda4;
      T lambda2;
      T gamma2;
      T approx;
      T p;
      bool whiten;
      bool expand;
      bool isConstant;
      bool updateConstant;
      bool ThetaDiag;
      bool ThetaDiagPlus;
      bool ThetaId;
      bool DequalsW;
      bool weightClasses;
      bool balanceClasses;
      bool extend;
      bool pattern;
      bool stochastic;
      bool scaleW;
      bool batch;
      bool verbose;
      bool clean;
      bool log;
      bool updateD;
      bool updateW;
      bool updateTheta;
      char* logName;
      int iter_updateD;
};

template <typename T> class Trainer {
   public:
      /// Empty constructor
      Trainer();
      /// Constructor with data
      Trainer(const int k, const int batchsize = 256,
            const int NUM_THREADS=-1);
      /// Constructor with initial dictionary
      Trainer(const Matrix<T>& D, const int batchsize = 256,
            const int NUM_THREADS=-1);
      /// Constructor with existing structure
      Trainer(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& D,
            const int itercount, const int batchsize, 
            const int NUM_THREADS);

      /// train or retrain using the matrix X
      void train(const Data<T>& X, const ParamDictLearn<T>& param);
      void trainOffline(const Data<T>& X, const ParamDictLearn<T>& param);

      /// train or retrain using the groups XT
      void train(const Data<T>& X, const vector_groups& groups,
            const int J, const constraint_type
            mode, const bool whiten = false, const T* param_C = NULL,
            const int p = 1, const bool pattern = false);

      /// Accessors
      void getA(Matrix<T>& A) const { A.copy(_A);};
      void getB(Matrix<T>& B) const { B.copy(_B);};
      void getD(Matrix<T>& D) const { D.copy(_D);};
      int getIter() const { return _itercount; };

   private:
      /// Forbid lazy copies
      explicit Trainer<T>(const Trainer<T>& trainer);
      /// Forbid lazy copies
      Trainer<T>& operator=(const Trainer<T>& trainer);

      /// clean the dictionary
      void cleanDict(const Data<T>& X, Matrix<T>& G,
            const bool posD = false,
            const constraint_type_D modeD = L2, const T gamma1 = 0, 
            const T gamma2 = 0,
            const T maxCorrel =
            0.999999);

      /// clean the dictionary
      void cleanDict(Matrix<T>& G);

      Matrix<T> _A;
      Matrix<T> _B;
      Matrix<T> _D;
      int _k;
      bool _initialDict;
      int _itercount;
      int _batchsize;
      int _NUM_THREADS;
};

/// Empty constructor
template <typename T> Trainer<T>::Trainer() : _k(0), _initialDict(false),
   _itercount(0), _batchsize(256) { 
      _NUM_THREADS=1;
#ifdef _OPENMP
      _NUM_THREADS =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
      _batchsize=floor(_batchsize*(_NUM_THREADS+1)/2);
   };

/// Constructor with data
template <typename T> Trainer<T>::Trainer(const int k, const
      int batchsize, const int NUM_THREADS) : _k(k), 
   _initialDict(false), _itercount(0),_batchsize(batchsize), 
   _NUM_THREADS(NUM_THREADS) { 
      if (_NUM_THREADS == -1) {
         _NUM_THREADS=1;
#ifdef _OPENMP
         _NUM_THREADS =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
      }
   };

/// Constructor with initial dictionary
template <typename T> Trainer<T>::Trainer(const Matrix<T>& D, 
      const int batchsize, const int NUM_THREADS) : _k(D.n()),
     _initialDict(true),_itercount(0),_batchsize(batchsize),
   _NUM_THREADS(NUM_THREADS) {
      _D.copy(D);
      _A.resize(D.n(),D.n());
      _B.resize(D.m(),D.n());
      if (_NUM_THREADS == -1) {
         _NUM_THREADS=1;
#ifdef _OPENMP
         _NUM_THREADS =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
      }
   }

/// Constructor with existing structure
template <typename T> Trainer<T>::Trainer(const Matrix<T>& A, const Matrix<T>&
      B, const Matrix<T>& D, const int itercount, const int batchsize, 
      const int NUM_THREADS) : _k(D.n()),_initialDict(true),_itercount(itercount),
   _batchsize(batchsize),
    _NUM_THREADS(NUM_THREADS) {
      _D.copy(D);
      _A.copy(A);
      _B.copy(B);
      if (_NUM_THREADS == -1) {
         _NUM_THREADS=1;
#ifdef _OPENMP
         _NUM_THREADS =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
      }
   };

template <typename T>
void Trainer<T>::cleanDict(const Data<T>& X, Matrix<T>& G, 
      const bool posD,
      const constraint_type_D modeD, const T gamma1,
      const T gamma2,
      const T maxCorrel) {
   int sparseD = modeD == L1L2 ? 2 : 6;
   const int k = _D.n();
   const int n = _D.m();
   const int M = X.n();
   T* const pr_G=G.rawX();
   Vector<T> aleat(n);
   Vector<T> col(n);
   for (int i = 0; i<k; ++i) {
      //pr_G[i*k+i] += 1e-10;
      for (int j = i; j<k; ++j) {
         if ((j > i && abs(pr_G[i*k+j])/sqrt(pr_G[i*k+i]*pr_G[j*k+j]) > maxCorrel) ||
               (j == i && abs(pr_G[i*k+j]) < 1e-4)) {
            /// remove element j and replace it by a random element of X
            const int ind = random() % M;
            Vector<T> d, g;
            _D.refCol(j,d);
            X.getData(col,ind);
            d.copy(col);
            if (modeD != L2) {
               aleat.copy(d);
               aleat.sparseProject(d,T(1.0),sparseD,gamma1,gamma2,T(2.0),posD);
            } else {
               if (posD) d.thrsPos();
               d.normalize();
            }
            G.refCol(j,g);
            _D.multTrans(d,g);
            for (int l = 0; l<_D.n(); ++l)
               pr_G[l*k+j] = pr_G[j*k+l];
         }
      }
   }
}


template <typename T>
void Trainer<T>::cleanDict(Matrix<T>& G) {
   const int k = _D.n();
   const int n = _D.m();
   T* const pr_G=G.rawX();
   for (int i = 0; i<k; ++i) {
      pr_G[i*k+i] += 1e-10;
   }
}


template <typename T>
void Trainer<T>::train(const Data<T>& X, const ParamDictLearn<T>& param) {

   T rho = param.rho;
   T t0 = param.t0;
   int sparseD = param.modeD == L1L2 ? 2 : param.modeD == L1L2MU ? 7 : 6;
   int NUM_THREADS=init_omp(_NUM_THREADS);
   if (param.verbose) {
      cout << "num param iterD: " << param.iter_updateD << endl;
      if (param.batch) {
         cout << "Batch Mode" << endl;
      } else if (param.stochastic) {
         cout << "Stochastic Gradient. rho : " << rho << ", t0 : " << t0 << endl;
      } else {
         if (param.modeParam == AUTO) {
            cout << "Online Dictionary Learning with no parameter " << endl;
         } else if (param.modeParam == PARAM1) {
            cout << "Online Dictionary Learning with parameters: " << t0 << " rho: " << rho << endl;
         } else {
            cout << "Online Dictionary Learning with exponential decay t0: " << t0 << " rho: " << rho << endl;
         }
      }
      if (param.posD) 
         cout << "Positivity constraints on D activated" << endl;
      if (param.posAlpha) 
         cout << "Positivity constraints on alpha activated" << endl;
      if (param.modeD != L2) cout << "Sparse dictionaries, mode: " << param.modeD << ", gamma1: " << param.gamma1 << ", gamma2: " << param.gamma2 << endl;
      cout << "mode Alpha " << param.mode << endl;
      if (param.clean) cout << "Cleaning activated " << endl;
      if (param.log && param.logName) {
         cout << "log activated " << endl;
         cerr << param.logName << endl;
      }
      if (param.mode == PENALTY && param.lambda==0 && param.lambda2 > 0 && !param.posAlpha)
         cout << "L2 solver is used" << endl;
      if (_itercount > 0)
         cout << "Retraining from iteration " << _itercount << endl;
      flush(cout);
   }

   const int M = X.n();
   const int K = _k;
   const int n = X.m();
   const int L = param.mode == SPARSITY ? static_cast<int>(param.lambda) : 
      param.mode == PENALTY && param.lambda == 0 && param.lambda2 > 0 && !param.posAlpha ? K : MIN(n,K);
   const int batchsize= param.batch ? M : MIN(_batchsize,M);

   if (param.verbose) {
      cout << "batch size: " << batchsize << endl;
      cout << "L: " << L << endl;
      cout << "lambda: " << param.lambda << endl;
      cout << "mode: " << param.mode << endl;
      flush(cout);
   }

   if (_D.m() != n || _D.n() != K) 
      _initialDict=false;

   srandom(0);
   Vector<T> col(n);
   if (!_initialDict) {
      _D.resize(n,K);
      for (int i = 0; i<K; ++i) {
         const int ind = random() % M;
         Vector<T> d;
         _D.refCol(i,d);
         X.getData(col,ind);
         d.copy(col);
      }
      _initialDict=true;
   }

   if (param.verbose) {
      cout << "*****Online Dictionary Learning*****" << endl;
      flush(cout);
   }

   Vector<T> tmp(n);
   if (param.modeD != L2) {
      for (int i = 0; i<K; ++i) {
         Vector<T> d;
         _D.refCol(i,d);
         tmp.copy(d);
         tmp.sparseProject(d,T(1.0),sparseD,param.gamma1,
               param.gamma2,T(2.0),param.posD);
      }
   } else {
      if (param.posD) _D.thrsPos();
      _D.normalize();
   }

   int count=0;
   int countPrev=0;
   T scalt0 =  abs<T>(t0);
   if (_itercount == 0) {
      _A.resize(K,K);
      _A.setZeros();
      _B.resize(n,K);
      _B.setZeros();
      if (!param.batch) {
         _A.setDiag(scalt0);
         _B.copy(_D);
         _B.scal(scalt0);
      }
   }

   //Matrix<T> G(K,K);

   Matrix<T> Borig(n,K);
   Matrix<T> Aorig(K,K);
   Matrix<T> Bodd(n,K);
   Matrix<T> Aodd(K,K);
   Matrix<T> Beven(n,K);
   Matrix<T> Aeven(K,K);
   SpVector<T>* spcoeffT=new SpVector<T>[_NUM_THREADS];
   Vector<T>* DtRT=new Vector<T>[_NUM_THREADS];
   Vector<T>* XT=new Vector<T>[_NUM_THREADS];
   Matrix<T>* BT=new Matrix<T>[_NUM_THREADS];
   Matrix<T>* AT=new Matrix<T>[_NUM_THREADS];
   Matrix<T>* GsT=new Matrix<T>[_NUM_THREADS];
   Matrix<T>* GaT=new Matrix<T>[_NUM_THREADS];
   Matrix<T>* invGsT=new Matrix<T>[_NUM_THREADS];
   Matrix<T>* workT=new Matrix<T>[_NUM_THREADS];
   Vector<T>* uT=new Vector<T>[_NUM_THREADS];
   for (int i = 0; i<_NUM_THREADS; ++i) {
      spcoeffT[i].resize(K);
      DtRT[i].resize(K);
      XT[i].resize(n);
      BT[i].resize(n,K);
      BT[i].setZeros();
      AT[i].resize(K,K);
      AT[i].setZeros();
      GsT[i].resize(L,L);
      GsT[i].setZeros();
      invGsT[i].resize(L,L);
      invGsT[i].setZeros();
      GaT[i].resize(K,L);
      GaT[i].setZeros();
      workT[i].resize(K,3);
      workT[i].setZeros();
      uT[i].resize(L);
      uT[i].setZeros();
   }

   Timer time, time2;
   time.start();
   srandom(0);
   Vector<int> perm;
   perm.randperm(M);

   Aodd.setZeros();
   Bodd.setZeros();
   Aeven.setZeros();
   Beven.setZeros();
   Aorig.copy(_A);
   Borig.copy(_B);

   int JJ = param.iter < 0 ? 100000000 : param.iter;
   bool even=true;
   int last_written=-40;
   int i;
   for (i = 0; i<JJ; ++i) {
      if (param.verbose) {
         cout << "Iteration: " << i << endl;
         flush(cout);
      }
      time.stop();
      if (param.iter < 0 && 
            time.getElapsed() > T(-param.iter)) break;
      if (param.log) {
         int seconds=static_cast<int>(floor(log(time.getElapsed())*5));
         if (seconds > last_written) {
            last_written++;
            sprintf(buffer_string,"%s_%d.log",param.logName,
                  last_written+40);
            writeLog(_D,T(time.getElapsed()),i,buffer_string);
            fprintf(stderr,"\r%d",i);
         }
      }
      time.start();
      
      Matrix<T> G;
      _D.XtX(G);
      if (param.clean) 
         this->cleanDict(X,G,param.posD,
               param.modeD,param.gamma1,param.gamma2);
      G.addDiag(MAX(param.lambda2,1e-10));
      int j;
      for (j = 0; j<_NUM_THREADS; ++j) {
         AT[j].setZeros();
         BT[j].setZeros();
      }

#pragma omp parallel for private(j)
      for (j = 0; j<batchsize; ++j) {
#ifdef _OPENMP
         int numT=omp_get_thread_num();
#else
         int numT=0;
#endif
         const int index=perm[(j+i*batchsize) % M];
         Vector<T>& Xj = XT[numT];
         SpVector<T>& spcoeffj = spcoeffT[numT];
         Vector<T>& DtRj = DtRT[numT];
         //X.refCol(index,Xj);
         X.getData(Xj,index);
         if (param.whiten) {
            if (param.pattern) {
               Vector<T> mean(4);
               Xj.whiten(mean,param.pattern);
            } else {
               Xj.whiten(X.V());
            }
         }
         _D.multTrans(Xj,DtRj);
         Matrix<T>& Gs = GsT[numT];
         Matrix<T>& Ga = GaT[numT];
         Matrix<T>& invGs = invGsT[numT];
         Matrix<T>& work= workT[numT];
         Vector<T>& u = uT[numT];
         Vector<int> ind;
         Vector<T> coeffs_sparse;
         spcoeffj.setL(L);
         spcoeffj.refIndices(ind);
         spcoeffj.refVal(coeffs_sparse);
         T normX=Xj.nrm2sq();
         coeffs_sparse.setZeros();
         if (param.mode < SPARSITY) {
            if (param.mode == PENALTY && param.lambda==0 && param.lambda2 > 0 && !param.posAlpha) {
               Matrix<T>& GG = G;
               u.set(0);
               GG.conjugateGradient(DtRj,u,1e-4,2*K);
               for (int k = 0; k<K; ++k) {
                  ind[k]=k;
                  coeffs_sparse[k]=u[k];
               }
            } else {
               coreLARS2(DtRj,G,Gs,Ga,invGs,u,coeffs_sparse,ind,work,normX,param.mode,param.lambda,param.posAlpha);
            }
         } else {
            if (param.mode == SPARSITY) {
               coreORMPB(DtRj,G,ind,coeffs_sparse,normX,L,T(0.0),T(0.0));
            } else if (param.mode==L2ERROR2) {
               coreORMPB(DtRj,G,ind,coeffs_sparse,normX,L,param.lambda,T(0.0));
            } else {
               coreORMPB(DtRj,G,ind,coeffs_sparse,normX,L,T(0.0),param.lambda);
            }
         }
         int count2=0;
         for (int k = 0; k<L; ++k) 
            if (ind[k] == -1) {
               break;
            } else {
               ++count2;
            }
         sort(ind.rawX(),coeffs_sparse.rawX(),0,count2-1);
         spcoeffj.setL(count2);
         AT[numT].rank1Update(spcoeffj);
         BT[numT].rank1Update(Xj,spcoeffj);
      }

      if (param.batch) {
         _A.setZeros();
         _B.setZeros();
         for (j = 0; j<_NUM_THREADS; ++j) {
            _A.add(AT[j]);
            _B.add(BT[j]);
         }
         Vector<T> di, ai,bi;
         Vector<T> newd(n);
         for (j = 0; j<param.iter_updateD; ++j) {
            for (int k = 0; k<K; ++k) {
               if (_A[k*K+k] > 1e-6) {
                  _D.refCol(k,di);
                  _A.refCol(k,ai);
                  _B.refCol(k,bi);
                  _D.mult(ai,newd,T(-1.0));
                  newd.add(bi);
                  newd.scal(T(1.0)/_A[k*K+k]);
                  newd.add(di);
                  if (param.modeD != L2) {
                     newd.sparseProject(di,T(1.0),
                           sparseD,param.gamma1,
                           param.gamma2,T(2.0),param.posD);
                  } else {
                     if (param.posD) newd.thrsPos();
                     newd.normalize2();
                     di.copy(newd);
                  }
               } else if (param.clean) {
                  _D.refCol(k,di);
                  di.setZeros();
               }
            }
         }
      } else if (param.stochastic) {
         _A.setZeros();
         _B.setZeros();
         for (j = 0; j<_NUM_THREADS; ++j) {
            _A.add(AT[j]);
            _B.add(BT[j]);
         }
         _D.mult(_A,_B,false,false,T(-1.0),T(1.0));
         T step_grad=rho/T(t0+batchsize*(i+1));
         _D.add(_B,step_grad);
         Vector<T> dj;
         Vector<T> dnew(n);
         if (param.modeD != L2) {
            for (j = 0; j<K; ++j) {
               _D.refCol(j,dj);
               dnew.copy(dj);
               dnew.sparseProject(dj,T(1.0),sparseD,param.gamma1,
                     param.gamma2,T(2.0),param.posD);
            }
         } else {
            for (j = 0; j<K; ++j) {
               _D.refCol(j,dj);
               if (param.posD) dj.thrsPos();
               dj.normalize2();
            }
         }
      } else {

         /// Dictionary Update
         /// Check the epoch parity
         int epoch = (((i+1) % M)*batchsize) / M;
         if ((even && ((epoch % 2) == 1)) || (!even && ((epoch % 2) == 0))) {
            Aodd.copy(Aeven);
            Bodd.copy(Beven);
            Aeven.setZeros();
            Beven.setZeros();
            count=countPrev;
            countPrev=0;
            even=!even;
         }

         int ii=_itercount+i;
         int num_elem=MIN(2*M, ii < batchsize ? ii*batchsize :
               batchsize*batchsize+ii-batchsize);
         T scal2=T(T(1.0)/batchsize);
         T scal;
         int totaliter=_itercount+count;
         if (param.modeParam == PARAM2) {
            scal=param.rho;
         } else if (param.modeParam == PARAM1) {
            scal=MAX(0.95,pow(T(totaliter)/T(totaliter+1),-rho));
         } else {
            scal = T(_itercount+num_elem+1-
                  batchsize)/T(_itercount+num_elem+1);
         }
         Aeven.scal(scal);
         Beven.scal(scal);
         Aodd.scal(scal);
         Bodd.scal(scal);
         if ((_itercount > 0 && i*batchsize < M) 
               || (_itercount == 0 && t0 != 0 && 
                  i*batchsize < 10000)) {
            Aorig.scal(scal);
            Borig.scal(scal);
            _A.copy(Aorig);
            _B.copy(Borig);
         } else {
            _A.setZeros();
            _B.setZeros();
         }
         for (j = 0; j<_NUM_THREADS; ++j) {
            Aeven.add(AT[j],scal2);
            Beven.add(BT[j],scal2);
         }
         _A.add(Aodd);
         _A.add(Aeven);
         _B.add(Bodd);
         _B.add(Beven);
         ++count;
         ++countPrev;

         Vector<T> di, ai,bi;
         Vector<T> newd(n);
         for (j = 0; j<param.iter_updateD; ++j) {
            for (int k = 0; k<K; ++k) {
               if (_A[k*K+k] > 1e-6) {
                  _D.refCol(k,di);
                  _A.refCol(k,ai);
                  _B.refCol(k,bi);
                  _D.mult(ai,newd,T(-1.0));
                  newd.add(bi);
                  newd.scal(T(1.0)/_A[k*K+k]);
                  newd.add(di);
                  if (param.modeD != L2) {
                     newd.sparseProject(di,T(1.0),sparseD,
                           param.gamma1,param.gamma2,T(2.0),param.posD);
                  } else {
                     if (param.posD) newd.thrsPos();
                     newd.normalize2();
                     di.copy(newd);
                  }
               } else if (param.clean && 
                     ((_itercount+i)*batchsize) > 10000) {
                  _D.refCol(k,di);
                  di.setZeros();
               }
            }
         }
      }
   }

   _itercount += i;
   if (param.verbose)
      time.printElapsed();
   delete[](spcoeffT);
   delete[](DtRT);
   delete[](AT);
   delete[](BT);
   delete[](GsT);
   delete[](invGsT);
   delete[](GaT);
   delete[](uT);
   delete[](XT);
   delete[](workT);
};


template <typename T>
void writeLog(const Matrix<T>& D, const T time, int iter, 
      char* name) {
   std::ofstream f;
   f.precision(12);
   f.flags(std::ios_base::scientific);
   f.open(name, ofstream::trunc);
   f << time << " " << iter << std::endl;
   for (int i = 0; i<D.n(); ++i) {
      for (int j = 0; j<D.m(); ++j) {
         f << D[i*D.m()+j] << " ";
      }
      f << std::endl;
   }
   f << std::endl;
   f.close();
};


template <typename T>
void Trainer<T>::trainOffline(const Data<T>& X, 
      const ParamDictLearn<T>& param) {

   int sparseD = param.modeD == L1L2 ? 2 : 6;
   int J = param.iter;
   int batch_size= _batchsize;
   int batchsize= _batchsize;
   int NUM_THREADS=init_omp(_NUM_THREADS);

   const int n = X.m();
   const int K = _k;
   const int M = X.n();
   cout << "*****Offline Dictionary Learning*****" << endl;
   fprintf(stderr,"num param iterD: %d\n",param.iter_updateD);
   cout << "batch size: " << _batchsize << endl;
   cout << "lambda: " << param.lambda << endl;
   cout << "X: " << n << " x " << M << endl;
   cout << "D: " << n << " x " << K << endl;
   flush(cout);

   srandom(0);
   Vector<T> col(n);
   if (!_initialDict) {
      _D.resize(n,K);
      for (int i = 0; i<K; ++i) {
         const int ind = random() % M;
         Vector<T> d;
         _D.refCol(i,d);
         X.getData(col,ind);
         d.copy(col);
      }
      _initialDict=true;
   }

   Vector<T> tmp(n);
   if (param.modeD != L2) {
      for (int i = 0; i<K; ++i) {
         Vector<T> d;
         _D.refCol(i,d);
         tmp.copy(d);
         tmp.sparseProject(d,T(1.0),sparseD,param.gamma1,
               param.gamma2,T(2.0),param.posD);
      }
   } else {
      if (param.posD) _D.thrsPos();
      _D.normalize();
   }

   Matrix<T> G(K,K);
   Matrix<T> coeffs(K,M);
   coeffs.setZeros();

   Matrix<T> B(n,K);
   Matrix<T> A(K,K);

   SpVector<T>* spcoeffT=new SpVector<T>[NUM_THREADS];
   Vector<T>* DtRT=new Vector<T>[NUM_THREADS];
   Vector<T>* coeffsoldT=new Vector<T>[NUM_THREADS];
   Matrix<T>* BT=new Matrix<T>[NUM_THREADS];
   Matrix<T>* AT=new Matrix<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      spcoeffT[i].resize(K);
      DtRT[i].resize(K);
      coeffsoldT[i].resize(K);
      BT[i].resize(n,K);
      BT[i].setZeros();
      AT[i].resize(K,K);
      AT[i].setZeros();
   }

   Timer time;
   time.start();
   srandom(0);
   Vector<int> perm;
   perm.randperm(M);
   int JJ = J < 0 ? 100000000 : J;
   Vector<T> weights(M);
   weights.setZeros();

   for (int i = 0; i<JJ; ++i) {
      if (J < 0 && time.getElapsed() > T(-J)) break;
      _D.XtX(G);
      if (param.clean) 
         this->cleanDict(X,G,param.posD,
               param.modeD,param.gamma1,param.gamma2);
      int j;
#pragma omp parallel for private(j)
      for (j = 0; j<batch_size; ++j) {
#ifdef _OPENMP
         int numT=omp_get_thread_num();
#else
         int numT=0;
#endif
         const int ind=perm[(j+i*batch_size) % M];
         Vector<T> Xj, coeffj;
         SpVector<T>& spcoeffj = spcoeffT[numT];
         Vector<T>& DtRj = DtRT[numT];
         Vector<T>& oldcoeffj = coeffsoldT[numT];
         X.getData(Xj,ind);
         if (param.whiten) {
            if (param.pattern) {
               Vector<T> mean(4);
               Xj.whiten(mean,param.pattern);
            } else {
               Xj.whiten(X.V());
            }
         }
         coeffs.refCol(ind,coeffj);
         oldcoeffj.copy(coeffj);
         _D.multTrans(Xj,DtRj);
         coeffj.toSparse(spcoeffj);
         G.mult(spcoeffj,DtRj,T(-1.0),T(1.0));
         if (param.mode == PENALTY) {
            coreIST(G,DtRj,coeffj,param.lambda,200,T(1e-3));
         } else {
            T normX = Xj.nrm2sq();
            coreISTconstrained(G,DtRj,coeffj,normX,param.lambda,200,T(1e-3));
         }
         oldcoeffj.toSparse(spcoeffj);
         AT[numT].rank1Update(spcoeffj,-weights[ind]);
         coeffj.toSparse(spcoeffj);
         AT[numT].rank1Update(spcoeffj);
         weights[ind]++;
         oldcoeffj.scal(weights[ind]);
         oldcoeffj.sub(coeffj);
         oldcoeffj.toSparse(spcoeffj);
         BT[numT].rank1Update(Xj,spcoeffj,T(-1.0));
      }

      A.setZeros();
      B.setZeros();
      T scal;
      int totaliter=i;
      int ii = i;
      int num_elem=MIN(2*M, ii < batchsize ? ii*batchsize :
            batchsize*batchsize+ii-batchsize);
      if (param.modeParam == PARAM2) {
         scal=param.rho;
      } else if (param.modeParam == PARAM1) {
         scal=MAX(0.95,pow(T(totaliter)/T(totaliter+1),-param.rho));
      } else {
         scal = T(num_elem+1-
               batchsize)/T(num_elem+1);
      }
      for (j = 0; j<NUM_THREADS; ++j) {
         A.add(AT[j]);
         B.add(BT[j]);
         AT[j].scal(scal);
         BT[j].scal(scal);
      }
      weights.scal(scal);
      Vector<T> di, ai,bi;
      Vector<T> newd(n);
      for (j = 0; j<param.iter_updateD; ++j) {
         for (int k = 0; k<K; ++k) {
            if (A[k*K+k] > 1e-6) {
               _D.refCol(k,di);
               A.refCol(k,ai);
               B.refCol(k,bi);
               _D.mult(ai,newd,T(-1.0));
               newd.add(bi);
               newd.scal(T(1.0)/A[k*K+k]);
               newd.add(di);
               if (param.modeD != L2) {
                  newd.sparseProject(di,T(1.0),
                        sparseD,param.gamma1,
                        param.gamma2,T(2.0),param.posD);
               } else {
                  if (param.posD) newd.thrsPos();
                  newd.normalize2();
                  di.copy(newd);
               }
            } else if (param.clean) {
               _D.refCol(k,di);
               di.setZeros();
            }
         }
      }
   }
   _D.XtX(G);
   if (param.clean) 
      this->cleanDict(X,G,param.posD,param.modeD,
            param.gamma1,param.gamma2);
   time.printElapsed();
   delete[](spcoeffT);
   delete[](DtRT);
   delete[](AT);
   delete[](BT);
   delete[](coeffsoldT);
}



#endif

