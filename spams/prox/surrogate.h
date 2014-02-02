/* Software SPAMS v2.4 - Copyright 2009-2013 Julien Mairal 
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

#ifndef SURROGATE_H
#define SURROGATE_H

#include <fista.h>

Timer timer;
Timer timer2;
Timer timer3;

using namespace FISTA;

template <typename T> struct ParamSurrogate { 
   ParamSurrogate() { 
      num_threads=1; 
      iters=100;
      epochs=100;
      minibatches=1; 
      normalized=false; 
      weighting_mode=0;
      averaging_mode=0;
      determineEta=true;
      eta=1;
      t0=0;
      verbose=false;
      random=false;
      optimized_solver=true;
      strategy=0;
   };
   ~ParamSurrogate() { };

   void print() const {
      if (random) cerr << "Randomized Sampling" << endl;
      if (minibatches > 1) cerr << "Mini-batches of size " << minibatches << endl;
      if (averaging_mode) cerr << "With averaging" << endl;
      cerr << "Weighting Scheme " << weighting_mode << endl;
   };

   int num_threads;
   int iters;
   int epochs;
   int minibatches;
   bool normalized;
   int weighting_mode;
   int averaging_mode;
   bool determineEta;
   bool optimized_solver;
   T eta;
   T t0;
   bool verbose;
   bool random;
   int strategy;
};

template <typename T, typename U>
class SmoothFunction {
   public:
      SmoothFunction(const U& Xt, const Vector<T>& y,
            const bool is_normalized=false, 
            const int nbatches=1, 
            const bool random = false, const T genericL= T(1.0)) : 
         _Xt(&Xt), _y(&y), _is_normalized(is_normalized), 
         _n(Xt.n()), 
         _p(Xt.m()),
         _nbatches(nbatches),
         _sizebatch(nbatches),
         _random(random),
         _counter(0),
         _save_n(Xt.n()), 
         _genericL(genericL),
         _constantL(is_normalized) { 
            _current_batch.resize(nbatches); 
            _num_batches = MAX(_n /_nbatches,1);
            if (!is_normalized) {
               _L.resize(_n);
               typename U::col spw;
               for (int i = 0; i<_n; ++i) {
                  _Xt->refCol(i,spw);
                  _L[i]=_genericL*spw.nrm2sq();
               }
               _genericL  = _L.sum()/this->_n;
            }
         };
      virtual ~SmoothFunction() { };

      /// evaluate the function value on the full dataset
      inline T eval(const Vector<T>& input) const {
         typename U::col spw;
         T tmp=0;
         for (int i = 0; i<_n; ++i) {
            _Xt->refCol(i,spw);
            const T y= (*_y)[i];
            const T s = input.dot(spw);
            tmp += this->eval_simple(y,s);  
         }
         return tmp / _n;
      };
      inline T eval_block(const Vector<T>& input) const {
         typename U::col spw;
         T tmp=0;
         for (int i = 0; i<_sizebatch; ++i) {
            const int ind = _current_batch[i];
            _Xt->refCol(ind,spw);
            const T y= (*_y)[ind];
            const T s = input.dot(spw);
            tmp += this->eval_simple(y,s);  
         }
         return tmp;
      };

      // rho = (1-w)rho + w (rho_sample)
      // output = (1-w)output + w(input - (1/rho) nabla(input))
      inline void add_sample_gradient(const Vector<T>& input, 
            Vector<T>& z, T& rho, const T w) {
         // update new_rho =>  (1-w)rho + w (rho_sample)
         if (_is_normalized) {
            // rho is already equal to genericL
            typename U::col spw;
            z.scal(T(1.0)-w);
            z.add(input,w);
            const T scal = w/(rho*_sizebatch);
            for (int i = 0; i<_sizebatch; ++i) {
               const int ind = _current_batch[i]; 
               _Xt->refCol(ind,spw);
               const T y= (*_y)[ind];
               const T s = input.dot(spw);
               z.add(spw,-scal*this->gradient_simple(y,s));
            }
         } else {
            T rho_sample=0;
            for (int i = 0; i<_sizebatch; ++i) {
               rho_sample += _L[_current_batch[i]];
            }
            rho_sample /= _sizebatch;
            //const T new_rho=(1-w)*rho + w*rho_sample;
            const T new_rho=rho;
            typename U::col spw;
            z.scal(rho*(T(1.0)-w)/new_rho);
            z.add(input,rho_sample*w/new_rho);
            const T scal = w/(_sizebatch*new_rho);
            for (int i = 0; i<_sizebatch; ++i) {
               const int ind = _current_batch[i];
               _Xt->refCol(ind,spw);
               const T y= (*_y)[ind];
               const T s = input.dot(spw);
               z.add(spw,-scal*this->gradient_simple(y,s));
            }
            rho=new_rho;
         }
      };
      inline void add_sample_gradient2(const Vector<T>& input, 
            Vector<T>& z, const T rho) {
         typename U::col spw;
         z.copy(input);
         z.scal(rho);
         for (int i = 0; i<_sizebatch; ++i) {
            const int ind = _current_batch[i]; 
            _Xt->refCol(ind,spw);
            const T y= (*_y)[ind];
            const T s = input.dot(spw);
            z.add(spw,-this->gradient_simple(y,s));
         }
      };
      virtual void add_sample_gradient3(const Vector<T>& input, 
            Vector<T>& output1, Vector<T>& output2, T& val) {
         typename U::col spw;
         val=0;
         for (int i = 0; i<_sizebatch; ++i) {
            const int ind = _current_batch[i]; 
            _Xt->refCol(ind,spw);
            const T y= (*_y)[ind];
            const T s = input.dot(spw);
            val+=this->eval_simple(y,s);
            const T s2 = -this->gradient_simple(y,s);
            output1.add(spw,s2-output2[ind]);
            output2[ind]=s2;
         }
      };

      void inline add_scal_grad(Vector<T>& input, const T scalin, const T scal, Vector<T>& av, const T scal2 = 0) {
         _tmp.resize(_sizebatch);
         typename U::col spw;
         if (_sizebatch==1) {
            const int ind = _current_batch[0];
            _Xt->refCol(ind,spw);
            const T s = scalin*input.dot(spw);
            const T y= (*_y)[ind];
            const T s2 = this->gradient_simple(y,s);
            input.add(spw,scal*s2);
            if (scal2) av.add(spw,scal2*s2);
         } else {
            for (int i = 0; i<_sizebatch; ++i) {
               const int ind = _current_batch[i];
               _Xt->refCol(ind,spw);
               _tmp[i] = scalin*input.dot(spw);
            }
            const T scalb = scal/_sizebatch;
            const T scal2b = scal2/_sizebatch;
            for (int i = 0; i<_sizebatch; ++i) {
               const int ind = _current_batch[i];
               const T y= (*_y)[ind];
               _Xt->refCol(ind,spw);
               const T s2  = this->gradient_simple(y,_tmp[i]); 
               input.add(spw,scalb*s2);
               if (scal2) av.add(spw,scal2b*s2);
            }
         }
      };
      inline T scal_grad(const T s) {  
         const int ind = _current_batch[0];
         const T y= (*_y)[ind];
         return this->gradient_simple(y,s);
      }
      virtual T dotprod_gradient3(const Vector<T>& input, 
            Vector<T>& stats) {
         typename U::col spw;
         T val=0;
         for (int i = 0; i<_sizebatch; ++i) {
            const int ind = this->_current_batch[i]; 
            _Xt->refCol(ind,spw);
            val-= stats[ind]*spw.dot(input); 
         }
         return val;
      };

      virtual T eval_simple(const T y, const T s) const = 0;
      virtual T gradient_simple(const T y, const T s) const = 0;
      
      /// compute a global constant L
      inline T genericL() const { return _genericL; };

      virtual void refData(typename U::col& output) {
         const int ind = _current_batch[0];
         _Xt->refCol(ind,output);
      }
      /// subsample the dataset
      virtual void subsample(const int n) {
         _save_n=_n;
         _counter=0;
         _n = MIN(n,_n);
         _num_batches = MAX(_n /_nbatches,1);
      };
      /// restore the full dataset
      virtual void un_subsample() {
         _n = _save_n;
         _counter=0;
         _num_batches = MAX(_n /_nbatches,1);
      };
      inline int n() const { return _n; };
      inline int p() const { return _p; };
      inline bool constantL() const { return _constantL; };
      void inline setRandom(const bool random) { _random=random;};
      void inline setMiniBatches(const int nbatches) { 
         _nbatches=nbatches;
         _sizebatch=_nbatches;
         _current_batch.resize(nbatches);
         _num_batches = MAX(_n /_nbatches,1);
      };
      int inline nbatches() const { return _nbatches; };
      int inline num_batches() const { return _num_batches; };
      int inline num_batch() const { return _num_batch; };

      /// choose a new sample
      void inline choose_random_batch() {
         if (_random) {
            for (int i = 0; i<_nbatches; ++i)   // size of the mini batches
               _current_batch[i]=random() % _n; 
         } else {
            for (int i = 0; i<_nbatches; ++i) 
               _current_batch[i]= _counter++ % _n; 
         }
      };
      int inline choose_random_fixedbatch() {
         const int size_lastbatch =_nbatches+(_n-_nbatches*_num_batches);
         _num_batch= (_random ? random() : _counter++) % _num_batches;
         _current_batch.resize(size_lastbatch);
         _sizebatch = _num_batch == _num_batches-1 ? size_lastbatch : _nbatches;
         for (int i = 0; i<_sizebatch; ++i)   // size of the mini batches
            _current_batch[i]=_num_batch*_nbatches+i; 
         return _num_batch;
      };
      T inline getL(Vector<T>& stats) {
         if (_constantL) {
            for (int i = 0; i<_num_batches-1; ++i) 
               stats[i]=_genericL*_nbatches;
            const int size_lastbatch =_nbatches+(_n-_nbatches*_num_batches);
            stats[_num_batches-1]=_genericL*size_lastbatch;
         } else {
            for (int i = 0; i<_num_batches; ++i) {
               stats[i]=0;
               for (int j = 0; j<_nbatches; ++j) {
                  stats[i]+=_L[i*_nbatches+j];
               }
            }
            const int size_lastbatch =_nbatches+(_n-_nbatches*_num_batches);
            for (int j = _nbatches; j<size_lastbatch; ++j) 
               stats[_num_batches-1]+=_L[(_num_batches-1)*_nbatches+j];
         }
      };
      /// compute the constant L for the current sample
      inline T sampleL() const {
         T rho_sample=0;
         for (int i = 0; i<_sizebatch; ++i) {
            rho_sample += _L[_current_batch[i]];
         }
         return rho_sample / _sizebatch;
      };

   private:
      explicit SmoothFunction<T,U>(const SmoothFunction<T,U>& dict);
      SmoothFunction<T,U>& operator=(const SmoothFunction<T,U>& dict);

   protected:
      Vector<int> _current_batch;
      int _n;
      int _nbatches;
      int _sizebatch;
      int _num_batches;
      int _num_batch;
      bool _random;
      int _counter;
      int _save_n;
      int _p;
      bool _constantL;

      const U* _Xt;
      const Vector<T>* _y;

      T _genericL;
      Vector<T> _L;
      T _is_normalized;
      Vector<T> _tmp;
};

template <typename T, typename U>
class LogisticFunction :  public SmoothFunction<T, U > {

   public:
      LogisticFunction(const U& Xt, const Vector<T>& y,
            const bool is_normalized=false, const int nbatches=1, const bool random = false) : 
         SmoothFunction<T,U>(Xt,y,is_normalized,nbatches,random,T(0.25)) { };
      virtual ~LogisticFunction() { };

      virtual T inline eval_simple(const T y, const T s) const {
         return logexp2<T>(-y*s);
      };
      virtual T inline gradient_simple(const T y, const T s) const {
         return -y/(T(1.0)+exp_alt<T>(y*s));
      };

   private:
      explicit LogisticFunction<T,U>(const LogisticFunction<T,U>& dict);
      LogisticFunction<T,U>& operator=(const LogisticFunction<T,U>& dict);

};

template <typename T, typename U>
class SquareFunction :  public SmoothFunction<T, U > {

   public:
      SquareFunction(const U& Xt, const Vector<T>& y,
            const bool is_normalized=false, const int nbatches=1, const bool random = false) : 
         SmoothFunction<T,U>(Xt,y,is_normalized,nbatches,random,T(1.0)) { };
      virtual ~SquareFunction() { };

      virtual T inline eval_simple(const T y, const T s) const {
         return 0.5*(y-s)*(y-s);
      };
      virtual T inline gradient_simple(const T y, const T s) const {
         return s-y;
      };

   private:
      explicit SquareFunction<T,U>(const SquareFunction<T,U>& dict);
      SquareFunction<T,U>& operator=(const SquareFunction<T,U>& dict);

};


// T is double or float
// D is the input type (vector or matrices)
// U is the data type (Matrix or SpMatrix)
template <typename T, typename U>
class OnlineSurrogate {
   public:
      OnlineSurrogate()  { };
      virtual ~OnlineSurrogate() {  };
      virtual void update_surrogate(const Vector<T>& input, const T weight) = 0;
      virtual T eval_function(const Vector<T>& input) = 0;
      virtual void minimize_surrogate(Vector<T>& output) = 0;
      virtual void subsample(const int n) = 0;
      virtual void un_subsample() = 0;
      virtual void initialize(const Vector<T>& input) = 0;
      virtual int n() const = 0; 
      virtual int num_batches() const  = 0;
      virtual int setRandom(const bool random) = 0;

   private:
      explicit OnlineSurrogate<T,U>(const OnlineSurrogate<T,U>& dict);
      OnlineSurrogate<T,U>& operator=(const OnlineSurrogate<T,U>& dict);
};

template <typename T, typename U>
class IncrementalSurrogate : public OnlineSurrogate<T,U> {
   public:
      IncrementalSurrogate() : _first_pass(true) { };
      virtual ~IncrementalSurrogate()  {  };
      virtual void update_incremental_surrogate(const Vector<T>& input) = 0;
      virtual void initialize_incremental(const Vector<T>& input, const int strategy = 0) = 0;
      virtual void minimize_incremental_surrogate(Vector<T>& output) = 0;
      virtual T get_param() const = 0;
      virtual T set_param(const T param) = 0;
      virtual T set_param_strong_convexity() { };
      inline void setFirstPass(const bool pass) { _first_pass = pass; };
      virtual T get_diff() const = 0;
      virtual T get_scal_diff() const = 0; //{ return 0; };
      virtual T reset_diff() =0 ;

   protected:
      int _strategy;
      bool _first_pass;

   private:
      explicit IncrementalSurrogate<T,U>(const IncrementalSurrogate<T,U>& dict);
      IncrementalSurrogate<T,U>& operator=(const IncrementalSurrogate<T,U>& dict);

};


template <typename T, typename U> 
class QuadraticSurrogate : public IncrementalSurrogate<T,U> {
   public:
      QuadraticSurrogate(SmoothFunction<T,U>* function) :
         _function(function) { _rho=T(1.0); _z.resize(function->p()); _scalL=T(1.0); }
      virtual ~QuadraticSurrogate() {  };

      virtual void inline update_surrogate(const Vector<T>& input, const T weight) {
         _function->choose_random_batch();
         // z = (1-w)output + w(input - (1/rho) nabla(input))
         _function->add_sample_gradient(input,_z,_rho,weight);
      };
      virtual T eval_function(const Vector<T>& input) {
         return _function->eval(input);
      };
      virtual void minimize_surrogate(Vector<T>& output) {
         output.copy(_z);
      };
      virtual void initialize(const Vector<T>& input) {
         this->_rho = _function->genericL();
         _z.copy(input);
      };
      virtual void subsample(const int n) { _function->subsample(n); };
      virtual void un_subsample() { _function->un_subsample(); };
      virtual int n() const { return _function->n(); };
      virtual int num_batches() const  { return _function->num_batches(); };
      virtual int setRandom(const bool random) { _function->setRandom(random); };

      /// incremental part
      virtual void update_incremental_surrogate(const Vector<T>& input) { 
         const int num_batch = _function->choose_random_fixedbatch();
         Vector<T> z_old;
         _stats2.refCol(num_batch,z_old);
         const T rho_old=_stats[num_batch];
         if (this->_strategy <= 2 || this->_strategy == 4) {
            if (!this->_first_pass)
               _z.sub(z_old);
            _function->add_sample_gradient2(input,z_old,rho_old*_scalL);
            _z.add(z_old);
         } else {
            T old_value=0;
            T old_valueb=0;
            if (!this->_first_pass) {
               _z.add(z_old,-rho_old);
               _z3.copy(input);
               _z3.sub(z_old);
               //old_value = _stats4[num_batch] + _function->dotprod_gradient3(_z3,_stats3)+0.5*rho_old*_scalL*_z3.nrm2sq();
               old_value = _stats4[num_batch] + _function->dotprod_gradient3(_z3,_stats3); // f_old+ nabla f(old)'( new-old)
               old_valueb=0.5*rho_old*_z3.nrm2sq();
            }
            z_old.copy(input);
            _z.add(z_old,rho_old);
            _function->add_sample_gradient3(input,_z2,_stats3,_stats4[num_batch]);
            if (!this->_first_pass) {
               _diff += (old_value-_stats4[num_batch]); // should be non-positive (convexity inequality)
               _diffb+=old_valueb;
//               _diff += (old_value-_stats4[num_batch]   > 0 ? T(1.0) : -T(1.0));
            }
         }
      };
      virtual void initialize_incremental(const Vector<T>& input, const int strategy) {
         const int p = input.n();
         this->_strategy = strategy;
         _z.resize(p);
         _z.setZeros();
         this->_rho = _function->n()*_function->genericL();
         const int num_batches = _function->num_batches();
         const int n = _function->n();
         this->_stats.resize(num_batches);
         _function->getL(this->_stats);
         this->_stats2.resize(p,num_batches,false);
         if (strategy == 3) {
            this->_stats3.resize(n);
            this->_stats3.setZeros();
            this->_stats4.resize(num_batches);
            this->_stats4.setZeros();
            _z2.resize(p);
            _z2.setZeros();
         }
      };
      virtual void minimize_incremental_surrogate(Vector<T>& output) {
         if (this->_strategy <= 2 || this->_strategy==4) {
            output.copy(_z);
            output.scal(T(1.0)/(_scalL*_rho));
         } else {
            output.copy(_z);
            output.add(_z2,T(1.0)/_scalL);
            output.scal(T(1.0)/(_rho));
         }
      };

      inline T get_param() const { return _scalL; };
      inline T set_param(const T param)  { _scalL = param; };
      virtual T get_scal_diff() const { 
         return -(_diff/_diffb)/_scalL;
      };
      virtual T get_diff() const { 
         return _diff+_scalL*_diffb; 
      };
      virtual T reset_diff()  { _diff=0; _diffb=0; };

   private:
      explicit QuadraticSurrogate<T,U>(const QuadraticSurrogate<T,U>& dict);
      QuadraticSurrogate<T,U>& operator=(const QuadraticSurrogate<T,U>& dict);

   protected:
      SmoothFunction<T,U>* _function; // contains all the data
      T _rho;
      Vector<T> _z;    
      Vector<T> _z2;
      Vector<T> _z3;

      T _scalL;
      Vector<T> _stats;   // rho_i
      Matrix<T> _stats2;  //  Theta_i     (rho_i Theta_i - nabla f_i  if strategy <= 2)
      Vector<T> _stats3;  // -nabla f_i   not used if strategy <= 2
      Vector<T> _stats4;          // last surrogate constant  not used if strategy <= 2
      T _diff;
      T _diffb;

};

template <typename T, typename U> 
class ProximalSurrogate : public QuadraticSurrogate<T,U> {

   public:
      ProximalSurrogate(SmoothFunction<T,U>* function,
            Regularizer<T>* prox, const T lambda) :
         QuadraticSurrogate<T,U>(function),
         _prox(prox), _lambda(lambda) { };

      virtual T eval_function(const Vector<T>& input) {
         return this->_function->eval(input)+_lambda*_prox->eval(input);
      };
      virtual void minimize_surrogate(Vector<T>& output) {
         _prox->prox(this->_z,output,_lambda/this->_rho);
      };
      virtual void minimize_incremental_surrogate(Vector<T>& output) {
         const int n = this->_function->n();
         if (this->_strategy <= 2 || this->_strategy==4) {
            if (_prox->id() == RIDGE) {
               output.add_scal(this->_z,T(1.0)/(this->_scalL*this->_rho+n*_lambda),0);
            } else if (_prox->id() == L1) {
               this->_z.softThrsholdScal(output,n*_lambda,T(1.0)/(this->_scalL*this->_rho));
            } else {
               Vector<T>& tmp = this->_z3;
               tmp.copy(this->_z);
               tmp.scal(T(1.0)/(this->_scalL*this->_rho));
               _prox->prox(tmp,output,n*_lambda/(this->_scalL*this->_rho));
            }
         } else {
            if (_prox->id() == RIDGE) {
               output.copy(this->_z);
               const T s = T(1.0)/(this->_rho+n*_lambda/this->_scalL);
               output.add_scal(this->_z2,s/this->_scalL,s);
            } else if (_prox->id() == L1) {
               output.copy(this->_z);
               output.add(this->_z2,T(1.0)/this->_scalL);
               output.softThrsholdScal(output,n*_lambda/(this->_scalL),T(1.0)/(this->_rho));
            } else {
               Vector<T>& tmp = this->_z3;
               tmp.copy(this->_z);
               tmp.add_scal(this->_z2,T(1.0)/(this->_rho*this->_scalL),T(1.0)/(this->_rho));
               _prox->prox(tmp,output,n*_lambda/(this->_scalL*this->_rho));
            }
         }
      };
      void inline changeLambda(const T lambda) { _lambda=lambda;};
      inline T set_param_strong_convexity()  { this->_scalL = _prox->id() == RIDGE ? this->_lambda/this->_rho: 0; };

   private:
      explicit ProximalSurrogate<T,U>(const ProximalSurrogate<T,U>& dict);
      ProximalSurrogate<T,U>& operator=(const ProximalSurrogate<T,U>& dict);

      Regularizer<T>* _prox;
      T _lambda;
};

template <typename T, typename U> 
class StochasticSolver {
   public:
      StochasticSolver() { };
      StochasticSolver(OnlineSurrogate<T,U>& surrogate, const
            ParamSurrogate<T>& param) : _eta(param.eta), _t0(param.t0),
      _minibatches(param.minibatches), 
      _weighting_mode(param.weighting_mode),
      _surrogate(&surrogate)  { 
         _logs.resize(3); 
      };

      virtual void solve(const Vector<T>& w0, Vector<T>& w, Vector<T>& wav, const int iters = 0, const
            bool auto_params = true, const int averaging_mode = 0, const bool
            verbose = false, const bool evaluate = true);

      virtual void getLogs(Vector<T>& logs) { logs.copy(_logs); };
      virtual int n() const { return _surrogate->n(); };

   private:
      explicit StochasticSolver<T,U>(const StochasticSolver<T,U>& dict);
      StochasticSolver<T,U>& operator=(const StochasticSolver<T,U>& dict);

   protected:
      void auto_parameters(const Vector<T>& w0, Vector<T>& w, Vector<T>& wav, const int averaging_mode = 0);
      void t0_to_eta() {
         switch (_weighting_mode) {
            case 0 : _eta=_t0+1; break;
            case 1 : _eta=sqr_alt<T>(_t0+1); break;
            case 2 : _eta=power<T>(_t0+1,0.75); break;
            default: break;
         }
      };
      T t_to_weight(const int t) const {
         switch (_weighting_mode) {
            case 0: return _eta/(static_cast<T>(t)+_t0); 
            case 1: return _eta/sqr_alt<T>(static_cast<T>(t)+_t0); 
            case 2: return _eta/power<T>(static_cast<T>(t)+_t0,0.75); 
            default: return  t==1 ? T(1.0) : _eta;
         }
      };
      virtual void subsample(const int newn) { _surrogate->subsample(newn); };
      virtual void un_subsample() { _surrogate->un_subsample(); };

      T _eta;
      T _t0;
      Vector<T> _logs;
      int _minibatches;
      int _weighting_mode;
      OnlineSurrogate<T,U>* _surrogate;
};

template <typename T, typename U>
void StochasticSolver<T,U>::auto_parameters(const Vector<T>& w0, Vector<T>& w, Vector<T>& wav, const int averaging_mode) {
   const int newn= this->n()/20;
   const int iters = ceil(this->n()/(20*_minibatches));
   /// inspired from bottou's determineta0 function
   T factor = 0.5;
   T lo_t0 = _t0;
   t0_to_eta();
   //const int ind_res=averaging_mode ? 1 : 0;
   const int ind_res=0;
   this->subsample(newn);
   this->solve(w0,w,w,iters,false,0,false);

   T loCost = _logs[ind_res];
   cerr << _logs[0] << " ";
   // try to reduce
   for (int t = 0; t<15; ++t) 
   {
      T hi_t0 = lo_t0* factor;
      if (hi_t0 < 1 || hi_t0 > 10000000) break;
      _t0 = hi_t0;
      t0_to_eta();
      this->solve(w0,w,w,iters,false,0,false);
      T hiCost = _logs[ind_res];
      cerr << _logs[0] << " ";
      if (hiCost > loCost && t==0) {
         factor=2.0;
      } else {
         if (hiCost >= loCost) break;
         lo_t0=hi_t0;
         loCost=hiCost;
      }
   }
   cerr << endl;
   _t0 = lo_t0;
   t0_to_eta();
   //cerr << "t0: " << _t0 << " eta: " << _eta << endl;
   this->un_subsample();
};

template <typename T, typename U>
void StochasticSolver<T,U>::solve(const Vector<T>& w0, Vector<T>& w, Vector<T>& wav, const int iters, 
      const bool auto_params, const int averaging_mode, const bool verbose, const bool evaluate) {

   if (verbose && iters > 0) {
      cout << "Standard Proximal Solver" << endl;
      if (auto_params)
         cout << "Automatic Parameters Adjustment" << endl;
   }
   Timer time;
   time.start();
   _logs[2]=0;
   w.copy(w0);
   if (averaging_mode) wav.copy(w0);
   if (auto_params && iters > 0) this->auto_parameters(w0,w,wav,averaging_mode);
   _surrogate->initialize(w0);
   T tmpweight=0;
   for (int i = 1; i<= iters; ++i) {
      const T weight = t_to_weight(i);
      _surrogate->update_surrogate(w,weight);
      _surrogate->minimize_surrogate(w);
      switch (averaging_mode) {
         case 1: wav.scal(T(1.0)-weight); wav.add(w,weight); break;
         case 2: tmpweight+=weight; wav.scal(T(1.0)-weight/tmpweight); wav.add(w,weight/tmpweight); break;
         default: break;
      };
   }

   time.stop();
   if (evaluate) {
      _logs[0]=_surrogate->eval_function(w);
      if (averaging_mode) _logs[1]=_surrogate->eval_function(wav);
   }
   _logs[2]=time.getElapsed();

   if (verbose && evaluate) {
      time.printElapsed();
      cout << "Result without averaging after " << iters << " iterations, cost = " << this->_logs[0] << endl;
      if (averaging_mode) cout << "Result with averaging after " << iters << " iterations, cost = " << this->_logs[1] << endl;
   }
};

template <typename T, typename U> 
class StochasticSmoothRidgeSolver : public StochasticSolver<T,U> {
   public:
      StochasticSmoothRidgeSolver(SmoothFunction<T,U>& function, const T lambda,
            const ParamSurrogate<T>& param) : _function(&function), _lambda(lambda) {
         this->_eta=param.eta;
         this->_t0=param.t0;
         this->_minibatches=param.minibatches;
         this->_weighting_mode=param.weighting_mode;
         this->_logs.resize(3); 
      };

      virtual void solve(const Vector<T>& w0, Vector<T>& w, Vector<T>& wav, const int iters = 0, const
            bool auto_params = true, const int averaging_mode = 0, const bool
            verbose = false, const bool evaluate = true);

   protected:
      virtual void subsample(const int newn) { _function->subsample(newn); };
      virtual void un_subsample() { _function->un_subsample(); };
      virtual int n() const { return _function->n(); };

   private:
      explicit StochasticSmoothRidgeSolver<T,U>(const StochasticSmoothRidgeSolver<T,U>& dict);
      StochasticSmoothRidgeSolver<T,U>& operator=(const StochasticSmoothRidgeSolver<T,U>& dict);

      SmoothFunction<T,U>* _function;
      T _lambda;
};



template <typename T,typename U>
void StochasticSmoothRidgeSolver<T,U>::solve(const Vector<T>& w0, Vector<T>& w, Vector<T>& wav, const int iters,
      const bool auto_params, const int averaging_mode, const bool verbose, const bool evaluate) {
   Timer time;
   time.start();
   if (verbose && iters > 0) {
      cout << "Dedicated L2 Solver" << endl;
      if (auto_params)
         cout << "Automatic Parameters Adjustment" << endl;
   }

   this->_logs[2]=0;
   w.copy(w0);
   if (averaging_mode) wav.copy(w0);
   if (auto_params && iters > 0) this->auto_parameters(w0,w,wav,averaging_mode);

   T rho=_function->genericL();
   T alpha = T(1.0);
   T beta = T(1.0);
   T gamma = 0;
   bool first_averaging=true;

   for (int t = 1; t<= iters; ++t) {

      T weight=this->t_to_weight(t);
      _function->choose_random_batch();
      if (!_function->constantL()) rho = (T(1.0)-weight)*rho + weight*_function->sampleL();
      const T kappa=rho/(rho+_lambda);
      const T one_minus_kappa=_lambda/(rho+_lambda);
      T newalpha = alpha*(T(1.0)-weight*one_minus_kappa);
      const T scal=-weight/((rho+_lambda)*newalpha);
      if (averaging_mode && weight < T(0.5)) {
         if (first_averaging) {
            first_averaging=false;
            wav.copy(w);
            wav.scal(alpha);
         } else {
            beta*=(1-weight);
            gamma=(T(1.0)-weight)*gamma*alpha/newalpha;
            const T scal2=-weight*gamma/((rho+_lambda)*beta);
            gamma -= weight;
            _function->add_scal_grad(w,alpha,scal,wav,scal2); //  x <- x + scal * nabla f( alpha x) 
         }
      } else {
         _function->add_scal_grad(w,alpha,scal,wav); //  x <- x + scal * nabla f( alpha x)
      }
      alpha=newalpha;
      if (alpha < 1e-10) {
         w.scal(alpha);
         alpha=T(1.0);
      }
      if (beta < 1e-10) {
         wav.scal(beta);
         beta=T(1.0);
      }
   }
   w.scal(alpha);
   if (averaging_mode) {
      wav.scal(beta);
      wav.add(w,-gamma);
   } 
   time.stop();
   if (evaluate) {
      this->_logs[0]=_function->eval(w)+0.5*_lambda*w.nrm2sq();
      if (averaging_mode) this->_logs[1]=_function->eval(wav)+0.5*_lambda*wav.nrm2sq();
   }
   this->_logs[2]=time.getElapsed();

   if (verbose && evaluate) {
      time.printElapsed();
      cout << "Result without averaging after " << iters << " iterations, cost = " << this->_logs[0] << endl;
      if (averaging_mode) cout << "Result with averaging after " << iters << " iterations, cost = " << this->_logs[1] << endl;
   }
}

template <typename T> 
class StochasticSmoothL1Solver : public StochasticSolver<T,SpMatrix<T> > {
   public:
      StochasticSmoothL1Solver(SmoothFunction<T,SpMatrix<T> >& function, const T lambda,
            const ParamSurrogate<T>& param) : _function(&function), _lambda(lambda) {
         this->_eta=param.eta;
         this->_t0=param.t0;
         this->_minibatches=param.minibatches;
         this->_weighting_mode=param.weighting_mode;
         this->_logs.resize(3); 
      };

      virtual void solve(const Vector<T>& w0, Vector<T>& w, Vector<T>& wav, const int iters = 0, const
            bool auto_params = true, const int averaging_mode = 0, const bool
            verbose = false, const bool evaluate = true);

   protected:
      virtual void subsample(const int newn) { _function->subsample(newn); };
      virtual void un_subsample() { _function->un_subsample(); };
      virtual int n() const { return _function->n(); };

   private:
      explicit StochasticSmoothL1Solver<T>(const StochasticSmoothL1Solver<T>& dict);
      StochasticSmoothL1Solver<T>& operator=(const StochasticSmoothL1Solver<T>& dict);

      SmoothFunction<T, SpMatrix<T> >* _function;
      T _lambda;
};



template <typename T>
void StochasticSmoothL1Solver<T>::solve(const Vector<T>& w0, Vector<T>& w, Vector<T>& wav, const int iters,
      const bool auto_params, const int averaging_mode, const bool verbose, const bool evaluate) {
   Timer time;
   time.start();
   w.copy(w0);
   this->_logs[2]=0;
   if (verbose && iters > 0) 
      cout << "Dedicated L1-Sparse Solver" << endl;

   if (auto_params && iters > 0) this->auto_parameters(w0,w,wav,averaging_mode);
   const int n = _function->n();
   const int p = _function->p();
   const T rho=_function->genericL();
   Vector<T> sumw(n);
   sumw[0]=0;
   Vector<T> prod(n);
   prod[0]=T(1.0);

   SpVector<T> col;
   Triplet<T,T,int>* pr_t = new Triplet<T,T,int>[p];
   memset(pr_t,0,p*sizeof(Triplet<T,T,int>));
   
   int counter=0;
   const T lambda_d_rho=_lambda/rho;
   int renorm2=0;
   int forgetting_offset=-1;
   int middle_offset=0;
   for (int t = 1; t<= iters; ++t) {
      const T weight=this->t_to_weight(t);
      _function->choose_random_batch();
      const T onemw=T(1.0)-weight;
      const int next_counter=(counter + 1) % n;
      sumw[next_counter]= sumw[counter]+weight*lambda_d_rho;
      if (sumw[next_counter] > T(1e50)) {
         sumw.add(-sumw[next_counter]);
      }
      const int prev_counter = (n + counter - 1) % n;
      prod[next_counter] = (t==1) ? T(1.0) : prod[counter]*onemw;
      if (prod[next_counter] < T(1e-8) || forgetting_offset == next_counter) {
         T scal=T(1.0)/prod[next_counter];
         for (int i = middle_offset; i != next_counter; i = (i+1) %n)
            prod[i] *= scal;
         prod[next_counter]=T(1.0);
         forgetting_offset=middle_offset;
         middle_offset=next_counter;
         renorm2++;
      }

      const T thrs1 = onemw*lambda_d_rho;
      const T thrs2 = weight/rho;

      _function->refData(col); // might be counter instead
      T* v = col.rawX();
      INTM * r = col.rawR();
      const int L = col.L();
      T s = 0;

      for (int i = 0; i< L; ++i) {
         const int ind=r[i];
         T& prx = pr_t[ind].x;
         T& prz = pr_t[ind].z;
         const int& prs = pr_t[ind].s;
         const T val =abs<T>(prx);
         const bool negval = prx < 0;
         const int sc = prs;
         if (val) {
            if (sc==counter) {
               prz = negval ? prx - thrs1 : prx + thrs1;
               s+=prx*v[i];
            } else {
               const T val2 = val + sumw[sc];
               const T new_val = val2 - sumw[counter];
               T& xval=prx;
               if (new_val > 0) {
                  xval = negval ? -new_val : new_val;
                  prz =new_val + thrs1;
                  s+=xval*v[i];
               } else {
                  xval=0;
                  // test if result is zero.
                  int up = counter + n;
                  int down = sc <= counter ? sc + n : sc;
                  bool proceed = true;
                  const int forget_offset = forgetting_offset <= counter ? forgetting_offset +n : forgetting_offset;
                  if (sc <= forget_offset) {
                     if (val2 <= sumw[forgetting_offset]) {
                        prz=0;
                        proceed = false;
                     } else {
                        down = forget_offset;
                     }
                  }
                  if (proceed) {
                     while ( up - down > 1) {
                        int current = (up+down)/2;
                        assert(current != up);
                        assert(current != down);
                        if (val2 <= sumw[current % n]) {
                           up=current; // always satisfied for up
                        } else {
                           down=current; // never satisfied for down
                        }
                     }
                     const T z_jm1= val2 - sumw[(n + up-1) % n] + lambda_d_rho;
                     prz = z_jm1 * (prod[next_counter]/prod[up % n]);
                  }
               }
               if (negval) prz=-prz;
            }
         } else {
            const int down = sc <= counter ? sc + n : sc;
            const int forget_offset = forgetting_offset <= counter ? forgetting_offset +n : forgetting_offset;
            prz = down <= forget_offset ? 0 : prz*(prod[next_counter]/prod[sc]);
         }
      }
      s = _function->scal_grad(s);
      s *= thrs2;

      /// gradient is s*col
      /// all pr_x corresponding to non-zeros grad have been updated
      /// pr_z is updated except for the gradient
      counter = next_counter;
      for (int i = 0; i< L; ++i) {
         const int ind=r[i];
         T& prx = pr_t[ind].x;
         T& prz = pr_t[ind].z;
         int& prs = pr_t[ind].s;
         prz -= s*v[i];
         if (prz > lambda_d_rho) {
            prx=prz-lambda_d_rho;
         } else if (prz < -lambda_d_rho) {
            prx=prz+lambda_d_rho;
         } else {
            prx=0;
         }
         prs = counter;
      }

      if (t==iters) {
         for (int i = 0; i<p; ++i) {
            T& prx = pr_t[i].x;
            const int& prs = pr_t[i].s;
            if (prx && counter != prs) {
               const T diff=sumw[counter] - sumw[prs];
               if (prx > diff) {
                  prx -= diff;
               } else if (prx < -diff) {
                  prx += diff;
               } else {
                  prx =0;
               }
            }
         }
      }
   }

   for (int i = 0; i<p; ++i) {
      w[i]=pr_t[i].x;
   }
   delete[](pr_t);
   time.stop();
   if (evaluate) {
      this->_logs[0]=_function->eval(w)+_lambda*w.asum();
      this->_logs[1]=0;
   } 
   this->_logs[2]=time.getElapsed();

   if (verbose && evaluate) {
      time.printElapsed();
      cout << "Result without averaging after " << iters << " iterations, cost = " << this->_logs[0] << endl;
   }
}

template <typename T, typename U>
void stochasticProximal(const Vector<T>& y, const U& X, const Vector<T>& w0,
      Vector<T>& w, Vector<T>& wav, const ParamFISTA<T>& paramprox, const ParamSurrogate<T>& param, 
      const T lambda, Vector<T>& logs) {
   SmoothFunction<T, U >* function;
   switch (paramprox.loss) {
      case LOG: function = new LogisticFunction<T, U>(X,y,param.normalized,param.minibatches,param.random); break;
      case SQUARE: function = new SquareFunction<T, U>(X,y,param.normalized,param.minibatches,param.random); break;
      default: function=NULL; cerr << "Unknown loss function" << endl; return;
   }
   if (paramprox.regul==RIDGE && param.optimized_solver) {
      StochasticSmoothRidgeSolver<T, U> solver(*function,lambda,param);
      solver.solve(w0,w,wav,param.iters,param.determineEta,param.averaging_mode,param.verbose);
      solver.getLogs(logs);
   } else {
      Regularizer<T,Vector<T> >* regul = setRegularizerVectors<T>(paramprox);
      ProximalSurrogate<T, U> surrogate(function,regul,lambda);
      StochasticSolver<T, U> solver(surrogate,param);
      solver.solve(w0,w,wav,param.iters,param.determineEta,param.averaging_mode,param.verbose);
      solver.getLogs(logs);
      delete(regul);
   }
   delete(function);
};

template <typename T>
void stochasticProximalSparse(const Vector<T>& y, const SpMatrix<T>& X, const Vector<T>& w0,
      Vector<T>& w, Vector<T>& wav, const ParamFISTA<T>& paramprox, const ParamSurrogate<T>& param, 
      const T lambda, Vector<T>& logs) {
   SmoothFunction<T, SpMatrix<T> >* function;
   switch (paramprox.loss) {
      case LOG: function = new LogisticFunction<T, SpMatrix<T> >(X,y,param.normalized,param.minibatches,param.random); break;
      case SQUARE: function = new SquareFunction<T, SpMatrix<T> >(X,y,param.normalized,param.minibatches,param.random); break;
      default: function=NULL; cerr << "Unknown loss function" << endl; return;
   }
   if (paramprox.regul==RIDGE && param.optimized_solver) {
      StochasticSmoothRidgeSolver<T, SpMatrix<T> > solver(*function,lambda,param);
      solver.solve(w0,w,wav,param.iters,param.determineEta,param.averaging_mode,param.verbose);
      solver.getLogs(logs);
   } else if (paramprox.regul==L1 && param.averaging_mode == 0 && param.optimized_solver) {
      StochasticSmoothL1Solver<T> solver(*function,lambda,param);
      solver.solve(w0,w,wav,param.iters,param.determineEta,param.averaging_mode,param.verbose);
      solver.getLogs(logs);
   } else {
      Regularizer<T,Vector<T> >* regul = setRegularizerVectors<T>(paramprox);
      ProximalSurrogate<T, SpMatrix<T> > surrogate(function,regul,lambda);
      StochasticSolver<T, SpMatrix<T> > solver(surrogate,param);
      solver.solve(w0,w,wav,param.iters,param.determineEta,param.averaging_mode,param.verbose);
      solver.getLogs(logs);
      delete(regul);
   }
   delete(function);
};

template <typename T, typename U>
void stochasticProximal(const Vector<T>& y, const U& X, const Matrix<T>& w0M,
      Matrix<T>& wM, Matrix<T>& wavM, const ParamFISTA<T>& paramprox, const ParamSurrogate<T>& param, 
      const Vector<T>& lambdaV, Matrix<T>& logsM) {
   const int num_lambdas=lambdaV.n();
   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i<num_lambdas; ++i) {
      Vector<T> w0;
      Vector<T> w;
      Vector<T> wav;
      Vector<T> logs;
      w0M.refCol(i,w0);
      wM.refCol(i,w);
      wavM.refCol(i,wav);
      logsM.refCol(i,logs);
      stochasticProximal(y,X,w0,w,wav,paramprox,param,lambdaV[i],logs);
   }
};

template <typename T>
void stochasticProximalSparse(const Vector<T>& y, const SpMatrix<T>& X, const Matrix<T>& w0M,
      Matrix<T>& wM, Matrix<T>& wavM, const ParamFISTA<T>& paramprox, const ParamSurrogate<T>& param, 
      const Vector<T>& lambdaV, Matrix<T>& logsM) {
   const int num_lambdas=lambdaV.n();
   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i<num_lambdas; ++i) {
      Vector<T> w0;
      Vector<T> w;
      Vector<T> wav;
      Vector<T> logs;
      w0M.refCol(i,w0);
      wM.refCol(i,w);
      wavM.refCol(i,wav);
      logsM.refCol(i,logs);
      stochasticProximalSparse(y,X,w0,w,wav,paramprox,param,lambdaV[i],logs);
   }
};

template <typename T, typename U> 
class IncrementalSolver {
   public:
      IncrementalSolver(IncrementalSurrogate<T,U>& surrogate, const
            ParamSurrogate<T>& param) : _minibatches(param.minibatches), 
      _surrogate(&surrogate)  { 
         _logs.resize(3); 
      };
      ~IncrementalSolver() { };

      virtual void solve(const Vector<T>& w0, Vector<T>& w, const int epochs = 0, const bool
            verbose = false, const bool evaluate = true, const int strategy = 3,
            const bool warm_restart = false);

      virtual void getLogs(Vector<T>& logs) { logs.copy(_logs); };

   protected:
      void auto_parameters(const Vector<T>& w0, Vector<T>& w, const int strategy = 3);

   private:
      explicit IncrementalSolver<T,U>(const IncrementalSolver<T,U>& dict);
      IncrementalSolver<T,U>& operator=(const IncrementalSolver<T,U>& dict);

      virtual void subsample(const int newn) { _surrogate->subsample(newn); };
      virtual void un_subsample() { _surrogate->un_subsample(); };

      Vector<T> _logs;
      IncrementalSurrogate<T,U>* _surrogate;
      int _minibatches;
};

template <typename T, typename U>
void IncrementalSolver<T,U>::solve(const Vector<T>& w0, Vector<T>& w, const int epochs, 
      const bool verbose, const bool
      evaluate, const int strategy, const bool warm_restart) {
   if (verbose && epochs > 0 && !warm_restart) {
      cout << "Standard Incremental Solver" << endl;
   }
   /// strategy 0: do not change L
   /// strategy 1: try to adjust L 
   Timer time;
   time.start();
   _logs.set(0);
   if (strategy == 4) _surrogate->set_param_strong_convexity();
   if (strategy >= 1 && strategy < 4 && !warm_restart) auto_parameters(w0,w,strategy);
   w.copy(w0);
   if (!warm_restart)
      _surrogate->initialize_incremental(w0,strategy);
   const int n = _surrogate->n();
   const int num_batches = _surrogate->num_batches();
   if (strategy == 3) _surrogate->reset_diff();
   if (epochs > 0) {
      /// first epoch
      _surrogate->setRandom(warm_restart);
      _surrogate->setFirstPass(!warm_restart);
      for (int j = 0; j< num_batches; ++j) {
         _surrogate->update_incremental_surrogate(w);
         _surrogate->minimize_incremental_surrogate(w);
      }
      if (strategy == 3 && warm_restart) {
         if ((_surrogate->get_diff()) < 0)  {
            T fact = (_surrogate->get_scal_diff());
            _surrogate->set_param(fact*_surrogate->get_param());
         } 
         _surrogate->reset_diff();
      }
      if (strategy == 3) _surrogate->reset_diff();

      /// classical epochs
      _surrogate->setRandom(true);
      _surrogate->setFirstPass(false);
      for (int i = 1; i<epochs; ++i) {
         for (int j = 0; j< num_batches; ++j) {
            _surrogate->update_incremental_surrogate(w);  
            _surrogate->minimize_incremental_surrogate(w);
         }
         if (strategy == 3) {
            if ((_surrogate->get_diff()) < 0) {
               T fact = (_surrogate->get_scal_diff());
               _surrogate->set_param(fact*_surrogate->get_param());
            }
            _surrogate->reset_diff();
         }
      }
   }
   time.stop();
   _logs[2]=time.getElapsed();
   if (evaluate)
      _logs[0]=_surrogate->eval_function(w);
   if (verbose && evaluate) {
      time.printElapsed();
      //timer2.printElapsed();
      //timer3.printElapsed();
      cout << "Result after " << epochs << " epochs, cost = " << this->_logs[0] << endl;
   }
};

template <typename T, typename U>
void IncrementalSolver<T,U>::auto_parameters(const Vector<T>& w0, Vector<T>& w, const int strategy) {
   const int newn= _surrogate->n()/20;
   /// inspired from bottou's determineta0 function
   T factor = 0.5;
   T lo_param = _surrogate->get_param();
   _surrogate->subsample(newn);
   this->solve(w0,w,1,false,true,0);

   T loCost = _logs[0];
   //cerr << _logs[0] << " ";
   // try to reduce
   for (int t = 0; t<15; ++t) 
   {
      T hi_param = lo_param* factor;
      if (hi_param < 1e-8 || hi_param > 1) break;
      _surrogate->set_param(hi_param);
      this->solve(w0,w,1,false,true,0);
      T hiCost = _logs[0];
  //          cerr << _logs[0] << " ";
      if (hiCost > loCost && t==0) {
         factor=2.0;
      } else {
         if (hiCost >= loCost) break;
         lo_param=hi_param;
         loCost=hiCost;
      }
   }
//    cerr << endl;
   _surrogate->set_param(strategy >= 2 ? lo_param/20 : lo_param);
   // cerr << "param: " << lo_param << endl;
   _surrogate->un_subsample();
};

template <typename T, typename U>
void incrementalProximal(const Vector<T>& y, const U& X, const Vector<T>& w0,
      Vector<T>& w, const ParamFISTA<T>& paramprox, const ParamSurrogate<T>& param, 
      const T lambda, Vector<T>& logs) {
   SmoothFunction<T, U >* function;
   switch (paramprox.loss) {
      case LOG: function = new LogisticFunction<T, U>(X,y,param.normalized,param.minibatches,param.random); break;
      case SQUARE: function = new SquareFunction<T, U>(X,y,param.normalized,param.minibatches,param.random); break;
      default: function=NULL; cerr << "Unknown loss function" << endl; return;
   }
   Regularizer<T,Vector<T> >* regul = setRegularizerVectors<T>(paramprox);
   ProximalSurrogate<T, U> surrogate(function,regul,lambda);
   IncrementalSolver<T, U> solver(surrogate,param);
   solver.solve(w0,w,param.epochs,param.verbose,true,param.strategy);
   solver.getLogs(logs);
   delete(regul);
   delete(function);
};

template <typename T, typename U>
void incrementalProximal(const Vector<T>& y, const U& X, const Matrix<T>& w0M,
      Matrix<T>& wM, const ParamFISTA<T>& paramprox, const ParamSurrogate<T>& param, 
      const Vector<T>& lambdaV, Matrix<T>& logsM) {

   cout << "Incremental proximal algorithm" << endl;
   cout << "heuristic mode " << param.strategy << endl;
   if (param.strategy==2)
      cout << " WARNING, strategy 2 is unsafe " << endl;
   const int num_lambdas=lambdaV.n();
   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i<num_lambdas; ++i) {
      Vector<T> w0;
      Vector<T> w;
      Vector<T> logs;
      w0M.refCol(i,w0);
      wM.refCol(i,w);
      logsM.refCol(i,logs);
      incrementalProximal(y,X,w0,w,paramprox,param,lambdaV[i],logs);
   }
};

template <typename T, typename U>
void incrementalProximalSeq(const Vector<T>& y, const U& X, const Matrix<T>& w0M,
      Matrix<T>& wM, const ParamFISTA<T>& paramprox, const ParamSurrogate<T>& param, 
      const Vector<T>& lambdaV, Matrix<T>& logsM) {
   const int num_lambdas=lambdaV.n();
   SmoothFunction<T, U >* function;
   switch (paramprox.loss) {
      case LOG: function = new LogisticFunction<T, U>(X,y,param.normalized,param.minibatches,param.random); break;
      case SQUARE: function = new SquareFunction<T, U>(X,y,param.normalized,param.minibatches,param.random); break;
      default: function=NULL; cerr << "Unknown loss function" << endl; return;
   }
   Regularizer<T,Vector<T> >* regul = setRegularizerVectors<T>(paramprox);
   ProximalSurrogate<T, U> surrogate(function,regul,lambdaV[0]);
   IncrementalSolver<T, U> solver(surrogate,param);

   cout << "path-following incremental algorithm" << endl;
   cout << "heuristic mode " << param.strategy << endl;
   if (param.strategy==2)
      cout << " WARNING, strategy 2 is unsafe " << endl;

   for (int i = 0; i<num_lambdas; ++i) {
      Vector<T> w0;
      Vector<T> w;
      Vector<T> logs;
      if (i==0) {
         w0M.refCol(i,w0);
      } else {
         wM.refCol(i-1,w0);
      }
      wM.refCol(i,w);
      logsM.refCol(i,logs);
      surrogate.changeLambda(lambdaV[i]);
      //solver.solve(w0,w,param.epochs,param.verbose,true,param.strategy,false);
      solver.solve(w0,w,param.epochs,param.verbose,true,param.strategy,i > 0);
      solver.getLogs(logs);
   }
   delete(regul);
   delete(function);
};




#endif
