
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

#ifndef FISTA_H
#define FISTA_H

#include <linalg.h>
#include <project.h>

namespace FISTA {

   enum loss_t { SQUARE, SQUARE_MISSING, LOG, LOGWEIGHT, MULTILOG, CUR, HINGE, INCORRECT_LOSS};
   enum regul_t { L0, L1, RIDGE, L2, LINF, L1CONSTRAINT, ELASTICNET, FUSEDLASSO, GROUPLASSO_L2, GROUPLASSO_LINF, GROUPLASSO_L2_L1, GROUPLASSO_LINF_L1, L1L2, L1LINF, L1L2_L1, L1LINF_L1, TREE_L0, TREE_L2, TREE_LINF, GRAPH, GRAPH_RIDGE, GRAPH_L2, TREEMULT, GRAPHMULT, L1LINFCR, NONE, TRACE_NORM, TRACE_NORM_VEC, RANK, RANK_VEC, INCORRECT_REG, GRAPH_PATH_L0, GRAPH_PATH_CONV};

   regul_t regul_from_string(char* regul) {
      if (strcmp(regul,"l0")==0) return L0;
      if (strcmp(regul,"l1")==0) return L1;
      if (strcmp(regul,"l2")==0) return RIDGE;
      if (strcmp(regul,"linf")==0) return LINF;
      if (strcmp(regul,"l2-not-squared")==0) return L2;
      if (strcmp(regul,"l1-constraint")==0) return L1CONSTRAINT;
      if (strcmp(regul,"elastic-net")==0) return ELASTICNET;
      if (strcmp(regul,"fused-lasso")==0) return FUSEDLASSO;
      if (strcmp(regul,"group-lasso-l2")==0) return GROUPLASSO_L2;
      if (strcmp(regul,"group-lasso-linf")==0) return GROUPLASSO_LINF;
      if (strcmp(regul,"sparse-group-lasso-l2")==0) return GROUPLASSO_L2_L1;
      if (strcmp(regul,"sparse-group-lasso-linf")==0) return GROUPLASSO_LINF_L1;
      if (strcmp(regul,"l1l2")==0) return L1L2;
      if (strcmp(regul,"l1linf")==0) return L1LINF;
      if (strcmp(regul,"l1l2+l1")==0) return L1L2_L1;
      if (strcmp(regul,"l1linf+l1")==0) return L1LINF_L1;
      if (strcmp(regul,"tree-l0")==0) return TREE_L0;
      if (strcmp(regul,"tree-l2")==0) return TREE_L2;
      if (strcmp(regul,"tree-linf")==0) return TREE_LINF;
      if (strcmp(regul,"graph")==0) return GRAPH;
      if (strcmp(regul,"graph-ridge")==0) return GRAPH_RIDGE;
      if (strcmp(regul,"graph-l2")==0) return GRAPH_L2;
      if (strcmp(regul,"multi-task-tree")==0) return TREEMULT;
      if (strcmp(regul,"multi-task-graph")==0) return GRAPHMULT;
      if (strcmp(regul,"l1linf-row-column")==0) return L1LINFCR;
      if (strcmp(regul,"trace-norm")==0) return TRACE_NORM;
      if (strcmp(regul,"trace-norm-vec")==0) return TRACE_NORM_VEC;
      if (strcmp(regul,"rank")==0) return RANK;
      if (strcmp(regul,"rank-vec")==0) return RANK_VEC;
      if (strcmp(regul,"graph-path-l0")==0) return GRAPH_PATH_L0;
      if (strcmp(regul,"graph-path-conv")==0) return GRAPH_PATH_CONV;
      if (strcmp(regul,"none")==0) return NONE;
      return INCORRECT_REG;
   }

   loss_t loss_from_string(char* loss) {
      if (strcmp(loss,"square")==0) return SQUARE;
      if (strcmp(loss,"square-missing")==0) return SQUARE_MISSING;
      if (strcmp(loss,"logistic")==0) return LOG;
      if (strcmp(loss,"weighted-logistic")==0) return LOGWEIGHT;
      if (strcmp(loss,"hinge")==0) return HINGE;
      if (strcmp(loss,"multi-logistic")==0) return MULTILOG;
      if (strcmp(loss,"cur")==0) return CUR;
      return INCORRECT_LOSS;
   }

   void print_loss(const loss_t& loss) {
      switch (loss) {
         case SQUARE: cout << "Square loss" << endl; break;
         case SQUARE_MISSING: cout << "Square loss with missing data" << endl; break;
         case LOG: cout << "Logistic loss" << endl; break;
         case LOGWEIGHT: cout << "Weighted Logistic loss" << endl; break;
         case HINGE: cout << "Hinge loss" << endl; break;
         case MULTILOG: cout << "Multiclass logistic Loss" << endl; break;
         case CUR: cout << "CUR decomposition" << endl; break;
         default: cerr << "Not implemented" << endl;
      }
   };

   bool loss_for_matrices(const loss_t& loss) {
      return loss==MULTILOG || loss==CUR;
   }

   void print_regul(const regul_t& regul) {
      switch (regul) {
         case L0: cout << "L0 regularization" << endl; break;
         case L1: cout << "L1 regularization" << endl; break;
         case RIDGE: cout << "L2-squared regularization" << endl; break;
         case L2: cout << "L2-not-squared regularization" << endl; break;
         case L1CONSTRAINT: cout << "L1 constraint regularization" << endl; break;
         case LINF: cout << "Linf regularization" << endl; break;
         case ELASTICNET: cout << "Elastic-net regularization" << endl; break;
         case FUSEDLASSO: cout << "Fused Lasso or total variation regularization" << endl; break;
         case GROUPLASSO_L2: cout << "Group Lasso L2" << endl; break;
         case GROUPLASSO_LINF: cout << "Group Lasso LINF" << endl; break;
         case GROUPLASSO_L2_L1: cout << "Group Lasso L2 + L1" << endl; break;
         case GROUPLASSO_LINF_L1: cout << "Group Lasso LINF + L1" << endl; break;
         case L1L2: cout << "L1L2 regularization" << endl; break;
         case L1LINF: cout << "L1LINF regularization" << endl; break;
         case TRACE_NORM: cout << "Trace Norm regularization" << endl; break;
         case TRACE_NORM_VEC: cout << "Trace Norm regularization for vectors" << endl; break;
         case RANK: cout << "Rank regularization" << endl; break;
         case RANK_VEC: cout << "Rank regularization for vectors" << endl; break;
         case L1L2_L1: cout << "L1L2 regularization + L1" << endl; break;
         case L1LINF_L1: cout << "L1LINF regularization + L1" << endl; break;
         case TREE_L0: cout << "Tree-L0 regularization" << endl; break;
         case TREE_L2: cout << "Tree-L2 regularization" << endl; break;
         case TREE_LINF: cout << "Tree-Linf regularization" << endl; break;
         case GRAPH: cout << "Graph regularization" << endl; break;
         case GRAPH_RIDGE: cout << "Graph+ridge regularization" << endl; break;
         case GRAPH_L2: cout << "Graph regularization with l2" << endl; break;
         case TREEMULT: cout << "multitask tree regularization" << endl; break;
         case GRAPHMULT: cout << "multitask graph regularization" << endl; break;
         case L1LINFCR: cout << "L1LINF regularization on rows and columns" << endl; break;
         case GRAPH_PATH_L0: cout << "Graph path non-convex regularization" << endl; break;
         case GRAPH_PATH_CONV: cout << "Graph path convex regularization" << endl; break;
         case NONE: cout << "No regularization" << endl; break;
         default: cerr << "Not implemented" << endl;
      }
   };

   bool regul_for_matrices(const regul_t& regul) {
      return regul==L1L2 || regul==L1LINF || regul==L1L2_L1 || regul==L1LINF_L1 
         || regul==TREEMULT || regul==GRAPHMULT || regul==L1LINFCR ||
         regul==TRACE_NORM || regul==RANK;
   }

   template <typename T> struct ParamFISTA { 
      ParamFISTA() { num_threads=1; max_it=100; L0=0.1; gamma=1.5; tol=1e-10; 
         it0=10; max_iter_backtracking=1000; loss=SQUARE; compute_gram=false; admm=false; lin_admm=false;
         intercept=false; regul=RIDGE; resetflow=false; delta=0; lambda2=0; lambda3=0; verbose=false; 
         pos=false; clever=true; a=1.0; b=0.0; c=1.0;
         log=false; logName=NULL; ista=false; subgrad=false;
         length_names=30; 
         name_regul=new char[length_names];
         name_loss=new char[length_names];
         is_inner_weights=false;
         inner_weights=NULL;
         eval=false;
         size_group=1;
         sqrt_step=true;
         transpose=false;
         fixed_step=false;
         copied=false;
         eval_dual_norm=false;
         groups=NULL;
         ngroups=0;
      }
      ~ParamFISTA() { 
         if (!copied) {
            delete[](name_regul); 
            delete[](name_loss); 
         }
      };
      int num_threads;
      int max_it;
      T L0;
      T gamma;
      int length_names;
      T lambda;
      T delta;
      T lambda2;
      T lambda3;
      T a;
      T b;
      T c;
      T tol;
      int it0;
      int max_iter_backtracking;
      loss_t loss;
      bool compute_gram;
      bool lin_admm;
      bool admm;
      bool intercept;
      bool resetflow;
      regul_t regul;
      char* name_regul;
      char* name_loss;
      bool verbose;
      bool pos;
      bool clever;
      bool log;
      bool ista;
      bool copied;
      bool subgrad;
      char* logName;
      bool is_inner_weights;
      T* inner_weights;
      bool eval;
      int size_group;
      bool sqrt_step;
      bool transpose;
      bool fixed_step;
      bool eval_dual_norm;
      int* groups;
      int ngroups;
   };

   template <typename T> struct ParamReg { 
      ParamReg() { size_group=1; lambda2d1 = 0; lambda=0; lambda3d1 = 0; pos=false; intercept=false; num_cols=1; graph_st=NULL; tree_st=NULL;
      graph_path_st=NULL; resetflow=false; clever=false; linf=true; transpose=false; ngroups=0;
      groups=NULL; };
      T lambda2d1;
      T lambda3d1;
      T lambda;
      int size_group;
      bool pos;
      bool intercept;
      int num_cols;
      GraphPathStruct<T>* graph_path_st;
      GraphStruct<T>* graph_st;
      TreeStruct<T>* tree_st;
      bool resetflow;
      bool clever;
      bool linf;
      bool transpose;
      int ngroups;
      int* groups;
   };

   template <typename T>
      bool param_for_admm(const ParamFISTA<T>& param) {
         return (param.admm) && (param.loss==SQUARE || param.loss == HINGE) 
            && (param.regul==GRAPH_L2 || param.regul==GRAPH || param.regul == NONE);
      };


   template <typename T,  typename F = Matrix<T>, typename D = Vector<T> , 
            typename E = Vector<T> >
               class SplittingFunction {
                  public:
                     SplittingFunction() { };
                     virtual ~SplittingFunction() { };

                     virtual void init(const E& y) { };
                     virtual T eval(const D& input) const = 0;
                     virtual void reset() { };
                     virtual T eval_split(const F& input) const = 0;
                     virtual T eval_weighted(const D& input,const F& input_struct, const T* weights) const { return this->eval(input);};
                     virtual int num_components() const = 0;
                     virtual void prox_split(F& splitted_w, const T lambda) const = 0;
                     virtual void init_split_variables(F& splitted_w) const = 0;
                     virtual void init_prim_var(E& prim_var) const { };
                     virtual void prox_prim_var(E& out,const E& dual_var, const E& prim_var, const T gamma) const { };
                     virtual void compute_new_prim(E& prim, const E& prim_var, const E& dual_var, const T gamma, const T delta) const { };
                     virtual void add_mult_design_matrix(const E& prim, E& out, const T fact) const { };

                  private:
                     explicit SplittingFunction<T,F,D,E>(const SplittingFunction<T,F,D,E>& loss);
                     SplittingFunction<T,F,D,E>& operator=(const SplittingFunction<T,F,D,E>& loss);
               };

   template <typename T, typename D = Vector<T> , typename E = Vector<T> >
      class Loss {
         public:
            Loss() { };
            virtual ~Loss() { };

            virtual void init(const E& input) = 0;
            virtual T eval(const D& input) const = 0;
            virtual void grad(const D& input, D& output) const = 0;
            virtual inline bool test_backtracking(const D& y, const D& grad, const D& prox, const T L) const {
               D tmp;
               tmp.copy(prox);
               tmp.sub(y);
               return (this->eval(prox) <= this->eval(y) + grad.dot(tmp) + 0.5*L*tmp.nrm2sq());
            };
            virtual T fenchel(const D& input) const = 0;
            virtual bool is_fenchel() const { return true; };
            virtual void var_fenchel(const D& x, D& grad1, D& grad2,
                  const bool intercept = false) const = 0;

         private:
            explicit Loss<T,D,E>(const Loss<T,D,E>& dict);
            Loss<T,D,E>& operator=(const Loss<T,D,E>& dict);
      };

   template <typename T> 
      class SqLossMissing : public Loss<T> {
         public:
            SqLossMissing(const AbstractMatrixB<T>& D) : _D(&D) { };
            virtual ~SqLossMissing() { };

            inline void init(const Vector<T>& x) { 
               _x.copy(x); 
               _missingvalues.clear();
               for (int i = 0; i<_x.n(); ++i) {
                  if (isnan(_x[i])) {
                     _x[i]=0;
                     _missingvalues.push_back(i);
                  }
               }
            };

            inline T eval(const Vector<T>& alpha) const {
               Vector<T> residual;
               residual.copy(_x);
               SpVector<T> spalpha(alpha.n());
               alpha.toSparse(spalpha);
               _D->mult(spalpha,residual,T(-1.0),T(1.0));
               for (ListIterator<int> it = _missingvalues.begin();
                     it != _missingvalues.end(); ++it)
                  residual[*it]=0;
               return 0.5*residual.nrm2sq();
            }

            inline void grad(const Vector<T>& alpha, Vector<T>& grad) const {
               Vector<T> residual;
               residual.copy(_x);
               SpVector<T> spalpha(alpha.n());
               alpha.toSparse(spalpha);
               _D->mult(spalpha,residual,T(-1.0),T(1.0));
               for (ListIterator<int> it = _missingvalues.begin();
                     it != _missingvalues.end(); ++it)
                  residual[*it]=0;
               _D->multTrans(residual,grad,T(-1.0),T(0.0));
            };
            virtual T fenchel(const Vector<T>& input) const {
               return 0.5*input.nrm2sq()+input.dot(_x);
            };
            virtual void var_fenchel(const Vector<T>& x, 
                  Vector<T>& grad1, Vector<T>& grad2, 
                  const bool intercept) const {
               grad1.copy(_x);
               SpVector<T> spalpha(x.n());
               x.toSparse(spalpha);
               _D->mult(spalpha,grad1,T(1.0),T(-1.0));
               for (ListIterator<int> it = _missingvalues.begin();
                     it != _missingvalues.end(); ++it)
                  grad1[*it]=0;
               if (intercept)
                  grad1.whiten(1); // remove the mean of grad1
               _D->multTrans(grad1,grad2,T(1.0),T(0.0));
            };

         private:
            explicit SqLossMissing<T>(const SqLossMissing<T>& dict);
            SqLossMissing<T>& operator=(const SqLossMissing<T>& dict);
            const AbstractMatrixB<T>* _D;
            Vector<T> _x;
            List<int> _missingvalues;
      };

   template <typename T> 
      class SqLoss : public Loss<T>, public SplittingFunction<T> {
         public:
            SqLoss(const AbstractMatrixB<T>& D) : _D(&D) { _compute_gram = false; };
            SqLoss(const AbstractMatrixB<T>& D, const Matrix<T>& G) : _D(&D), _G(&G) { _compute_gram = true; };
            virtual ~SqLoss() { };

            inline void init(const Vector<T>& x) { 
               _x.copy(x); 
               if (_compute_gram) {
                  _D->multTrans(x,_DtX);
               } 
            };

            inline T eval(const Vector<T>& alpha) const {
               Vector<T> residual;
               residual.copy(_x);
               SpVector<T> spalpha(alpha.n());
               alpha.toSparse(spalpha);
               if (spalpha.L() < alpha.n()/2) {
                  _D->mult(spalpha,residual,T(-1.0),T(1.0));
               } else {
                  _D->mult(alpha,residual,T(-1.0),T(1.0));
               }
               return 0.5*residual.nrm2sq();
            }
            inline void grad(const Vector<T>& alpha, Vector<T>& grad) const {
               if (_compute_gram) {
                  grad.copy(_DtX);
                  SpVector<T> spalpha(alpha.n());
                  alpha.toSparse(spalpha);
                  _G->mult(spalpha,grad,T(1.0),-T(1.0));
               } else {
                  Vector<T> residual;
                  residual.copy(_x);
                  SpVector<T> spalpha(alpha.n());
                  alpha.toSparse(spalpha);
                  _D->mult(spalpha,residual,T(-1.0),T(1.0));
                  _D->multTrans(residual,grad,T(-1.0),T(0.0));
               }
            };
            virtual inline bool test_backtracking(const Vector<T>& y, const Vector<T>& grad, const Vector<T>& prox, const T L) const {
               Vector<T> tmp;
               tmp.copy(y);
               tmp.sub(prox);
               SpVector<T> sptmp(tmp.n());
               tmp.toSparse(sptmp);
               if (_compute_gram) {
                  return (_G->quad(sptmp) <= L*sptmp.nrm2sq());
               } else {
                  Vector<T> tmp2(_D->m());
                  _D->mult(sptmp,tmp2);
                  return (tmp2.nrm2sq() <= L*sptmp.nrm2sq());
               }
            };
            virtual T fenchel(const Vector<T>& input) const {
               return 0.5*input.nrm2sq()+input.dot(_x);
            };
            virtual void var_fenchel(const Vector<T>& x, 
                  Vector<T>& grad1, Vector<T>& grad2, 
                  const bool intercept) const {
               grad1.copy(_x);
               SpVector<T> spalpha(x.n());
               x.toSparse(spalpha);
               _D->mult(spalpha,grad1,T(1.0),T(-1.0));
               if (intercept)
                  grad1.whiten(1); // remove the mean of grad1
               _D->multTrans(grad1,grad2,T(1.0),T(0.0));
            };
            inline int num_components() const { return _D->m();};

            inline void prox_split(Matrix<T>& splitted_w, const T lambda) const {
               const int n = this->num_components();
               Vector<T> row(_D->n());
               Vector<T> wi;
               for (int i = 0; i<n; ++i) {
                  _D->copyRow(i,row);
                  splitted_w.refCol(i,wi);
                  const T xtw=row.dot(wi);
                  const T xtx=row.dot(row);
                  wi.add(row,-lambda*(xtw-_x[i])/(T(1.0)+lambda*xtx));
               }
            };
            inline T eval_split(const Matrix<T>& input) const {
               const int n = this->num_components();
               Vector<T> row(_D->n());
               Vector<T> wi;
               T sum = 0;
               for (int i = 0; i<n; ++i) {
                  _D->copyRow(i,row);
                  input.refCol(i,wi);
                  const T xtw=row.dot(wi);
                  sum += 0.5*(_x[i]-xtw)*(_x[i]-xtw);
               }
               return sum;
            };
            inline void init_split_variables(Matrix<T>& splitted_w) const {
               splitted_w.resize(_D->n(),_D->m());
               splitted_w.setZeros();
            };
            inline void init_prim_var(Vector<T>& prim_var) const {
               prim_var.resize(_D->m());
               prim_var.setZeros();
            }
            virtual void prox_prim_var(Vector<T>& out,const Vector<T>& dual_var, 
                  const Vector<T>& prim_var, const T c) const {
               const T gamma=T(1.0)/c;
               out.copy(dual_var);
               out.scal(-gamma);
               _D->mult(prim_var,out,T(1.0),T(1.0));
               out.add(_x,gamma);
               out.scal(T(1.0)/(T(1.0)+gamma));
            };
            inline void compute_new_prim(Vector<T>& prim, const Vector<T>& prim_var, 
                  const Vector<T>& dual_var, const T gamma, const T delta) const { 
               Vector<T> tmp;
               _D->mult(prim,tmp);
               tmp.scal(-gamma);
               tmp.add(prim_var);
               tmp.add(dual_var,gamma);
               _D->multTrans(tmp,prim,T(1.0),delta);
            };
            inline void add_mult_design_matrix(const Vector<T>& prim, 
                  Vector<T>& out, const T fact) const { 
               _D->mult(prim,out,fact,T(1.0));
            };

         private:
            explicit SqLoss<T>(const SqLoss<T>& dict);
            SqLoss<T>& operator=(const SqLoss<T>& dict);
            const AbstractMatrixB<T>* _D;
            Vector<T> _x;
            bool _compute_gram;
            const Matrix<T>* _G;
            Vector<T> _DtX;
      };

   template <typename T> 
      class HingeLoss : public SplittingFunction<T > {
         public:
            HingeLoss(const AbstractMatrixB<T>& X) : _X(&X) {  };
            virtual ~HingeLoss() { };

            inline void init(const Vector<T>& y) { 
               _y.copy(y);
            };
            inline T eval(const Vector<T>& w) const {
               Vector<T> tmp(_X->m());
               SpVector<T> spw(w.n());
               w.toSparse(spw);
               _X->mult(spw,tmp);
               tmp.mult(_y,tmp);
               tmp.neg();
               tmp.add(T(1.0));
               tmp.thrsPos();
               return tmp.sum()/tmp.n();
            };
            virtual T eval_split(const Matrix<T>& input) const {
               Vector<T> row(_X->n());
               Vector<T> wi;
               T sum = 0;
               for (int i = 0; i<_X->n(); ++i) {
                  _X->copyRow(i,row);
                  input.refCol(i,wi);
                  sum += MAX(0,T(1.0)-_y[i]*row.dot(wi));
               }
               return sum/_X->m();
            };
            virtual int num_components() const { return _X->m(); };
            inline void init_split_variables(Matrix<T>& splitted_w) const {
               splitted_w.resize(_X->n(),_X->m());
               splitted_w.setZeros();
            };
            inline void init_prim_var(Vector<T>& prim_var) const {
               prim_var.resize(_X->m());
               prim_var.setZeros();
            }
            inline void prox_prim_var(Vector<T>& out,const Vector<T>& dual_var, 
                  const Vector<T>& prim_var, const T lambda, const T c) const {
               const T gamma=T(1.0)/c;
               out.copy(dual_var);
               out.scal(-gamma);
               _X->mult(prim_var,out,T(1.0),T(1.0));
               const T thrs=T(1.0)-gamma;
               for (int i = 0; i<out.n(); ++i) {
                  const T y = _y[i]*out[i];
                  if (y < thrs) {
                     out[i]+=_y[i]*gamma;
                  } else if (y < T(1.0)) {
                     out[i]=_y[i];
                  }
               }
            }
            inline void compute_new_prim(Vector<T>& prim, const Vector<T>& prim_var, 
                  const Vector<T>& dual_var, const T gamma, const T delta) const { 
               Vector<T> tmp;
               _X->mult(prim,tmp);
               tmp.scal(-gamma);
               tmp.add(prim_var);
               tmp.add(dual_var,gamma);
               _X->multTrans(tmp,prim,T(1.0),delta);
            };
            inline void add_mult_design_matrix(const Vector<T>& prim, Vector<T>& out,
                  const T fact) const { 
               _X->mult(prim,out,fact,T(1.0));
            };
            inline void prox_split(Matrix<T>& splitted_w, const T lambda) const {
               const int n = this->num_components();
               Vector<T> row(_X->n());
               Vector<T> wi;
               for (int i = 0; i<n; ++i) {
                  _X->copyRow(i,row);
                  splitted_w.refCol(i,wi);
                  const T xtw=row.dot(wi);
                  const T xtx=row.dot(row);
                  const T diff=1-_y[i]*xtw;
                  if (diff > lambda*xtx) {
                     wi.add(row,lambda*_y[i]);
                  } else if (diff > 0) {
                     wi.add(row,_y[i]*diff/xtx);
                  }
               }
            };

         private:
            explicit HingeLoss<T>(const HingeLoss<T>& dict);
            HingeLoss<T>& operator=(const HingeLoss<T>& dict);

            const AbstractMatrixB<T>* _X;
            Vector<T> _y;
      };

   template <typename T, bool weighted = false> 
      class LogLoss : public Loss<T> {
         public:
            LogLoss(const AbstractMatrixB<T>& X) : _X(&X) {  };
            virtual ~LogLoss() { };

            inline void init(const Vector<T>& y) { 
               _y.copy(y);
               if (weighted) {
                  int countpos=0;
                  for (int i = 0; i<y.n(); ++i)
                     if (y[i]>0) countpos++; 
                  _weightpos=T(1.0)/countpos;
                  _weightneg=T(1.0)/MAX(1e-3,(y.n()-countpos));
               }
            };

            inline T eval(const Vector<T>& w) const {
               Vector<T> tmp(_X->m());
               SpVector<T> spw(w.n());
               w.toSparse(spw);
               _X->mult(spw,tmp);
               tmp.mult(_y,tmp);
               tmp.neg();
               tmp.logexp();
               if (weighted) {
                  T sum=0;
                  for (int i = 0; i<tmp.n(); ++i)
                     sum+= _y[i]>0 ? _weightpos*tmp[i] : _weightneg*tmp[i];
                  return sum;
               } else {
                  return tmp.sum()/tmp.n();
               }
            };
            inline void grad(const Vector<T>& w, Vector<T>& grad) const {
               Vector<T> tmp(_X->m());
               SpVector<T> spw(w.n());
               w.toSparse(spw);
               _X->mult(spw,tmp);
               tmp.mult(_y,tmp);
               tmp.exp();
               tmp.add(T(1.0));
               tmp.inv();
               tmp.mult(_y,tmp);
               tmp.neg();
               if (weighted) {
                  for (int i = 0; i<tmp.n(); ++i)
                     tmp[i] *= _y[i] > 0 ? _weightpos : _weightneg;
                  _X->multTrans(tmp,grad);
               } else {
                  _X->multTrans(tmp,grad);
                  grad.scal(T(1.0)/_X->m());
               }
            };
            virtual bool is_fenchel() const { return !weighted; };
            virtual T fenchel(const Vector<T>& input) const {
               T sum = 0;
               if (weighted) {
               // TODO : check that
                  for (int i = 0; i<input.n(); ++i) {
                     T prod = _y[i]>0 ? input[i]/_weightpos : -input[i]/_weightneg;
                     sum += _y[i] >0 ? _weightpos*(xlogx(1.0+prod)+xlogx(-prod)) : _weightneg*(xlogx(1.0+prod)+xlogx(-prod));
                  }
                  return sum;
               } else {
                  for (int i = 0; i<input.n(); ++i) {
                     T prod = _y[i]*input[i]*_X->m();
                     sum += xlogx(1.0+prod)+xlogx(-prod);
                  }
                  return sum/_X->m();
               }
            };
            virtual void var_fenchel(const Vector<T>& w, Vector<T>& grad1, Vector<T>& grad2, const bool intercept) const {
               grad1.resize(_X->m());
               SpVector<T> spw(w.n());
               w.toSparse(spw);
               _X->mult(spw,grad1);
               grad1.mult(_y,grad1);
               grad1.exp();
               grad1.add(T(1.0));
               grad1.inv();
               grad1.mult(_y,grad1);
               grad1.neg();   // -gradient (no normalization)
               if (intercept)
                  grad1.project_sft_binary(_y);
               grad1.scal(T(1.0)/_X->m());
               _X->multTrans(grad1,grad2);
            };
         private:
            explicit LogLoss<T,weighted>(const LogLoss<T,weighted>& dict);
            LogLoss<T,weighted>& operator=(const LogLoss<T,weighted>& dict);

            const AbstractMatrixB<T>* _X;
            Vector<T> _y;
            T _weightpos;
            T _weightneg;
      };

   template <typename T> 
      class MultiLogLoss : public Loss<T, Matrix<T> > {
         public:
            MultiLogLoss(const AbstractMatrixB<T>& X) : _X(&X) {  };

            virtual ~MultiLogLoss() { };

            inline void init(const Vector<T>& y) { 
               _y.resize(y.n());
               for (int i = 0; i<y.n(); ++i)
                  _y[i] = static_cast<int>(y[i]);
            };
            inline T eval(const Matrix<T>& W) const {
               Matrix<T> tmp;
               _X->multSwitch(W,tmp,true,true);
               //W.mult(*_X,tmp,true,true);
               Vector<T> col;
               T sum=0;
               for (int i = 0; i<tmp.n(); ++i) {
                  tmp.refCol(i,col);
                  sum+=col.softmax(_y[i]);
               }
               return sum/tmp.n();
            };
            inline void grad(const Matrix<T>& W, Matrix<T>& grad) const {
               Matrix<T> tmp;
               _X->multSwitch(W,tmp,true,true);
               //W.mult(*_X,tmp,true,true);
               Vector<T> col;
               grad.resize(W.m(),W.n());
               for (int i = 0; i<tmp.n(); ++i) {
                  tmp.refCol(i,col);
                  col.add(-col[_y[i]]);
                  bool overweight=false;
                  for (int j = 0; j<col.n(); ++j)
                     if (col[j] > 1e2)
                        overweight=true;
                  if (overweight) {
                     const int ind =col.fmax();
                     col.setZeros();
                     col[ind]=1;
                  } else {
                     col.exp();
                     col.scal(T(1.0)/col.sum());
                     col.scal(T(1.0)/col.sum());
                  }
                  col[_y[i]] = col[_y[i]]-T(1.0);
               }
               _X->mult(tmp,grad,true,true);
               grad.scal(T(1.0)/_X->m());
            };
            virtual T fenchel(const Matrix<T>& input) const {
               T sum = 0;
               Vector<T> col;
               for (int i = 0; i<input.n(); ++i) {
                  const int clas = _y[i];
                  input.refCol(i,col);
                  for (int j = 0; j<input.m(); ++j) {
                     if (j == clas) {
                        sum += xlogx(_X->m()*input[i*input.m()+j]+1.0);
                     } else {
                        sum += xlogx(_X->m()*input[i*input.m()+j]);
                     }
                  }
               }
               return sum/_X->m();
            };
            virtual void var_fenchel(const Matrix<T>& W, Matrix<T>& grad1, Matrix<T>& grad2, const bool intercept) const  {
               _X->multSwitch(W,grad1,true,true);
               //W.mult(*_X,grad1,true,true);
               Vector<T> col;
               for (int i = 0; i<grad1.n(); ++i) {
                  grad1.refCol(i,col);
                  col.add(-col[_y[i]]);
                  bool overweight=false;
                  for (int j = 0; j<col.n(); ++j)
                     if (col[j] > 1e2)
                        overweight=true;
                  if (overweight) {
                     const int ind =col.fmax();
                     col.setZeros();
                     col[ind]=1;
                  } else {
                     col.exp();
                     col.scal(T(1.0)/col.sum());
                     col.scal(T(1.0)/col.sum());
                  }
                  col[_y[i]] = col[_y[i]]-T(1.0);
               }
               if (intercept) {
                  Vector<T> row;
                  for (int i = 0; i<grad1.m(); ++i) {
                     grad1.extractRow(i,row);
                     row.project_sft(_y,i);
                     grad1.setRow(i,row);
                  }
               }
               grad1.scal(T(1.0)/_X->m());
               grad2.resize(W.m(),W.n());
               _X->mult(grad1,grad2,true,true);
            };
         private:
            explicit MultiLogLoss<T>(const MultiLogLoss<T>& dict);
            MultiLogLoss<T>& operator=(const MultiLogLoss<T>& dict);

            const AbstractMatrixB<T>* _X;
            Vector<int> _y;
      };

   template <typename T> 
      class LossCur: public Loss<T, Matrix<T>, Matrix<T> > {
         public:
            LossCur(const AbstractMatrixB<T>& X) : _X(&X) {  };

            virtual ~LossCur() { };

            inline void init(const Matrix<T>& y) { };

            inline T eval(const Matrix<T>& A) const {
               Matrix<T> tmp(_X->m(),A.n());
               _X->mult(A,tmp);
               Matrix<T> tmp2;
               //tmp2.copy(*_X);
               _X->copyTo(tmp2);
               //tmp.mult(*_X,tmp2,false,false,T(-1.0),T(1.0));
               _X->multSwitch(tmp,tmp2,false,false,T(-1.0),T(1.0));
               return 0.5*tmp2.normFsq();
            };
            inline void grad(const Matrix<T>& A, Matrix<T>& grad) const {
               Matrix<T> tmp(_X->m(),A.n());
               _X->mult(A,tmp);
               Matrix<T> tmp2;
               //tmp2.copy(*_X);
               _X->copyTo(tmp2);
               //tmp.mult(*_X,tmp2,false,false,T(-1.0),T(1.0));
               _X->multSwitch(tmp,tmp2,false,false,T(-1.0),T(1.0));
               //tmp2.mult(*_X,tmp,false,true,T(-1.0),T(0.0));
               _X->multSwitch(tmp2,tmp,true,false,T(-1.0),T(0.0));
               grad.resize(A.m(),A.n());
               _X->mult(tmp,grad,true,false);
            };
            virtual T fenchel(const Matrix<T>& input) const {
               return 0.5*input.normFsq()+_X->dot(input);
            }
            virtual void var_fenchel(const Matrix<T>& A, Matrix<T>& grad1, Matrix<T>& grad2, const bool intercept) const {
               Matrix<T> tmp(_X->m(),A.n());
               _X->mult(A,tmp);
               //grad1.copy(*_X);
               _X->copyTo(grad1);
               //tmp.mult(*_X,grad1,false,false,T(1.0),T(-1.0));
               _X->multSwitch(tmp,grad1,false,false,T(1.0),T(-1.0));
               //grad1.mult(*_X,tmp,false,true,T(1.0),T(0.0));
               _X->multSwitch(grad1,tmp,true,false,T(1.0),T(0.0));
               grad2.resize(A.m(),A.n());
               _X->mult(tmp,grad2,true,false);
            };
         private:
            explicit LossCur<T>(const LossCur<T>& dict);
            LossCur<T>& operator=(const LossCur<T>& dict);

            const AbstractMatrixB<T>* _X;
      };

   template <typename T> 
      class SqLossMat : public Loss<T, Matrix<T> , Matrix<T> > {
         public:
            SqLossMat(const AbstractMatrixB<T>& D) : _D(&D) { _compute_gram = false; };
            SqLossMat(const AbstractMatrixB<T>& D, const Matrix<T>& G) : _D(&D), _G(&G) { 
               _compute_gram = true; };
            virtual ~SqLossMat() { };

            virtual inline void init(const Matrix<T>& x) { 
               _x.copy(x); 
               if (_compute_gram) {
                  _D->mult(x,_DtX,true,false);
               } 
            };

            inline T eval(const Matrix<T>& alpha) const {
               Matrix<T> residual;
               residual.copy(_x);
               SpMatrix<T> spalpha;
               alpha.toSparse(spalpha);
               _D->mult(spalpha,residual,false,false,T(-1.0),T(1.0));
               return 0.5*residual.normFsq();
            }
            inline void grad(const Matrix<T>& alpha, Matrix<T>& grad) const {
               SpMatrix<T> spalpha;
               alpha.toSparse(spalpha);
               if (_compute_gram) {
                  grad.copy(_DtX);
                  _G->mult(spalpha,grad,false,false,T(1.0),-T(1.0));
               } else {
                  Matrix<T> residual;
                  residual.copy(_x);
                  _D->mult(spalpha,residual,false,false,T(-1.0),T(1.0));
                  _D->mult(residual,grad,true,false,T(-1.0),T(0.0));
               }
            };
            virtual inline bool test_backtracking(const Matrix<T>& y, const Matrix<T>& grad, const Matrix<T>& prox, const T L) const {
               Matrix<T> tmp;
               tmp.copy(y);
               tmp.sub(prox);
               SpMatrix<T> sptmp;
               tmp.toSparse(sptmp);
               if (_compute_gram) {
                  SpVector<T> col;
                  T sum=0;
                  for (int i = 0; i<sptmp.n(); ++i) {
                     sptmp.refCol(i,col);
                     sum += _G->quad(col);
                  }
                  return (sum <= L*sptmp.normFsq());
               } else {
                  Matrix<T> tmp2;
                  _D->mult(sptmp,tmp2);
                  return (tmp2.normFsq() <= L*sptmp.normFsq());
               }
            };
            virtual T fenchel(const Matrix<T>& input) const {
               return 0.5*input.normFsq()+input.dot(_x);
            };
            virtual void var_fenchel(const Matrix<T>& x, Matrix<T>& grad1, Matrix<T>& grad2, const bool intercept) const {
               grad1.copy(_x);
               SpMatrix<T> spalpha;
               x.toSparse(spalpha);
               _D->mult(spalpha,grad1,false,false,T(1.0),T(-1.0));
               if (intercept)
                  grad1.center();
               _D->mult(grad1,grad2,true,false,T(1.0),T(0.0));
            };

         private:
            explicit SqLossMat<T>(const SqLossMat<T>& dict);
            SqLossMat<T>& operator=(const SqLossMat<T>& dict);
            const AbstractMatrixB<T>* _D;
            Matrix<T> _x;
            bool _compute_gram;
            const Matrix<T>* _G;
            Matrix<T> _DtX;
      };

   template <typename T, typename L>
      class LossMatSup : public Loss<T,Matrix<T>, Matrix<T> > {
         public:
            LossMatSup() { };

            virtual ~LossMatSup() { 
               for (int i = 0; i<_N; ++i) {
                  delete(_losses[i]);
                  _losses[i]=NULL;
               }
               delete[](_losses);
            };

            virtual void init(const Matrix<T>& input) {
               Vector<T> col;
               _m=input.m();
               for (int i = 0; i<_N; ++i) {
                  input.refCol(i,col);
                  _losses[i]->init(col);
               }
            };

            inline T eval(const Matrix<T>& w) const {
               Vector<T> col;
               T sum = 0;
               for (int i = 0; i<_N; ++i) {
                  w.refCol(i,col);
                  sum+=_losses[i]->eval(col);
               }
               return sum;
            }
            inline void grad(const Matrix<T>& w, Matrix<T>& grad) const {
               Vector<T> col, col2;
               grad.resize(w.m(),w.n());
               for (int i = 0; i<_N; ++i) {
                  w.refCol(i,col);
                  grad.refCol(i,col2);
                  _losses[i]->grad(col,col2);
               }
            };
            virtual T fenchel(const Matrix<T>& input) const {
               Vector<T> col;
               T sum = 0;
               for (int i = 0; i<_N; ++i) {
                  input.refCol(i,col);
                  sum += _losses[i]->fenchel(col);
               }
               return sum;
            }
            virtual void var_fenchel(const Matrix<T>& x, Matrix<T>& grad1, Matrix<T>& grad2, const bool intercept) const {
               grad1.resize(_m,x.n());
               grad2.resize(x.m(),x.n());
               Vector<T> col, col2, col3;
               for (int i = 0; i<_N; ++i) {
                  x.refCol(i,col);
                  grad1.refCol(i,col2);
                  grad2.refCol(i,col3);
                  _losses[i]->var_fenchel(col,col2,col3,intercept);
               }
            };
            virtual bool is_fenchel() const {
               bool ok=true;
               for (int i = 0; i<_N; ++i) 
                  ok = ok && _losses[i]->is_fenchel();
               return ok;
            };
            virtual void dummy() = 0;

         private:
            explicit LossMatSup<T,L>(const LossMatSup<T,L>& dict);
            LossMatSup<T,L>& operator=(const LossMatSup<T,L>& dict);
            int _m;

         protected:
            int _N;
            L** _losses;
      };

   template <typename T, typename L>
      class LossMat : public LossMatSup<T,L> { };

   template <typename T, bool weighted>
      class LossMat<T, LogLoss<T,weighted> > : public LossMatSup<T, LogLoss<T,weighted> > {
         public:
            LossMat(const int N, const AbstractMatrixB<T>& X) {
               this->_N=N;
               this->_losses=new LogLoss<T,weighted>*[this->_N];
               Vector<T> col;
               for (int i = 0; i<this->_N; ++i) 
                  this->_losses[i]=new LogLoss<T,weighted>(X);
            }
            virtual void dummy() { };
            virtual ~LossMat() { };
      };

   template <typename T>
      class LossMat<T, SqLossMissing<T> > : public LossMatSup<T, SqLossMissing<T> > {
         public:
            LossMat(const int N, const AbstractMatrixB<T>& X) {
               this->_N=N;
               this->_losses=new SqLossMissing<T>*[this->_N];
               Vector<T> col;
               for (int i = 0; i<this->_N; ++i) 
                  this->_losses[i]=new SqLossMissing<T>(X);
            }
            virtual void dummy() { };
            virtual ~LossMat() { };
      };

   template <typename T, typename D = Vector<T> >
      class Regularizer {
         public:
            Regularizer() { };
            Regularizer(const ParamReg<T>& param) { 
               _intercept=param.intercept;
               _pos=param.pos;
            }
            virtual ~Regularizer() { };

            virtual void reset() { };
            virtual void prox(const D& input, D& output, const T lambda) = 0;
            virtual T eval(const D& input) const = 0;
            /// returns phi^star( input ) and ouput=input if the fenchel is unconstrained
            /// returns 0 and scale input such that phi^star(output)=0 otherwise
            virtual void fenchel(const D& input, T& val, T& scal) const = 0;
            virtual bool is_fenchel() const { return true; };
            virtual bool is_intercept() const { return _intercept; };
            virtual bool is_subgrad() const { return false; };
            virtual void sub_grad(const D& input, D& output) const {  };
            virtual T eval_paths(const D& x, SpMatrix<T>& paths_mat) const { return this->eval(x); };
            virtual T eval_dual_norm(const D& x) const { return 0; };
            // TODO complete for all norms
            virtual T eval_dual_norm_paths(const D& x, SpMatrix<T>& path) const { return this->eval_dual_norm(x); };

         protected:
            bool _pos;
            bool _intercept;

         private:
            explicit Regularizer<T,D>(const Regularizer<T,D>& reg);
            Regularizer<T,D>& operator=(const Regularizer<T,D>& reg);

      };

   template <typename T> 
      class Lasso : public Regularizer<T> {
         public:
            Lasso(const ParamReg<T>& param) : Regularizer<T>(param) { };
            virtual ~Lasso() { };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.copy(x);
               if (this->_pos) y.thrsPos();
               y.softThrshold(lambda);
               if (this->_intercept) y[y.n()-1] = x[y.n()-1];
            };
            T inline eval(const Vector<T>& x) const { 
               return (this->_intercept ? x.asum() - abs(x[x.n()-1]) : x.asum());
            };
            void inline fenchel(const Vector<T>& input, T& val, T& scal) const {
               Vector<T> output;
               output.copy(input);
               if (this->_pos) output.thrsPos();
               T mm = output.fmaxval();
               scal= mm > 1.0 ? T(1.0)/mm : 1.0;
               val=0;
               if (this->_intercept & abs<T>(output[output.n()-1]) > EPSILON) val=INFINITY; 
            };
            virtual bool is_subgrad() const { return true; };
            virtual void sub_grad(const Vector<T>& input, Vector<T>& output) const {  
               output.resize(input.n());
               if (!this->_pos) {
                  for (int i = 0; i<input.n(); ++i) {
                     output[i] = input[i] > 0 ? T(1.0) : input[i] < 0 ? -T(1.0) : 0;
                  }
               } else {
                  for (int i = 0; i<input.n(); ++i) {
                     output[i] = input[i] > 0 ? T(1.0) : 0;
                  }
               }
               if (this->_intercept) output[output.n()-1]=0;
            }
      };

   template <typename T> 
      class LassoConstraint : public Regularizer<T> {
         public:
            LassoConstraint(const ParamReg<T>& param) : Regularizer<T>(param) { _thrs=param.lambda; };
            virtual ~LassoConstraint() { };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               Vector<T> tmp;
               tmp.copy(x);
               if (this->_intercept) {
                  tmp[tmp.n()-1]=0;
                  tmp.sparseProject(y,_thrs,1,0,0,0,this->_pos);
                  y[y.n()-1] = x[y.n()-1];
               } else {
                  tmp.sparseProject(y,_thrs,1,0,0,0,this->_pos);
               }
            };
            T inline eval(const Vector<T>& x) const {
               return 0;
            };
            void inline fenchel(const Vector<T>& input, T& val, T& scal) const { 
               scal=1.0;
               Vector<T> output;
               output.copy(input);
               if (this->_intercept) output[output.n()-1]=0;
               val = _thrs*(this->_pos ? MAX(output.maxval(),0) : output.fmaxval());
            };
            virtual bool is_subgrad() const { return false; };
         private:
            T _thrs;
      };

   template <typename T> 
      class Lzero : public Regularizer<T> {
         public:
            Lzero(const ParamReg<T>& param) : Regularizer<T>(param) { };
            virtual ~Lzero() { };

            virtual bool is_fenchel() const { return false; };
            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.copy(x);
               if (this->_pos) y.thrsPos();
               y.hardThrshold(sqrt(2*lambda));
               if (this->_intercept) y[y.n()-1] = x[y.n()-1];
            };
            T inline eval(const Vector<T>& x) const { 
               return (this->_intercept ? x.lzero() - 1 : x.lzero());
            };
            void inline fenchel(const Vector<T>& input, T& val, T& scal) const { };
      };

   template <typename T> 
      class None: public Regularizer<T>, public SplittingFunction<T, SpMatrix<T> > {
         public:
            None() { };
            None(const ParamReg<T>& param) { };
            virtual ~None() { };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.copy(x);
            };
            T inline eval(const Vector<T>& x) const {  return 0; };
            void inline fenchel(const Vector<T>& input, T& val, T& scal) const {  };
            virtual bool is_fenchel() const { return false; };
            virtual bool is_subgrad() const { return true; };
            virtual void sub_grad(const Vector<T>& input, Vector<T>& output) const { 
               output.setZeros();
            } 
            virtual void reset() { };
            virtual T eval_split(const SpMatrix<T>& input) const { return 0; };
            virtual int num_components() const { return 0; };
            virtual void prox_split(SpMatrix<T>& splitted_w, const T lambda) const { };
            virtual void init_split_variables(SpMatrix<T>& splitted_w) const { };
            virtual void init(const Vector<T>& y) { };
      };

   template <typename T> 
      class Ridge: public Regularizer<T> {
         public:
            Ridge(const ParamReg<T>& param) : Regularizer<T>(param) { };
            virtual ~Ridge() { };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.copy(x);
               if (this->_pos) y.thrsPos();
               y.scal(T(1.0/(1.0+lambda)));
               if (this->_intercept) y[y.n()-1] = x[y.n()-1];
            };
            T inline eval(const Vector<T>& x) const { 
               return (this->_intercept ? 0.5*x.nrm2sq() - 0.5*x[x.n()-1]*x[x.n()-1] : 0.5*x.nrm2sq());
            };
            void inline fenchel(const Vector<T>& input, T& val, T& scal) const {
               Vector<T> tmp;
               tmp.copy(input);
               if (this->_pos) tmp.thrsPos();
               val=this->eval(tmp);
               scal=T(1.0);
               if (this->_intercept & abs<T>(tmp[tmp.n()-1]) > EPSILON) val=INFINITY; 
            };
            virtual bool is_subgrad() const { return true; };
            virtual void sub_grad(const Vector<T>& input, Vector<T>& output) const {  
               output.resize(input.n());
               if (!this->_pos) {
                  for (int i = 0; i<input.n(); ++i) {
                     output[i] = input[i] > 0 ? 0.5*input[i] : 0;
                  }
               } else {
                  output.copy(input);
                  output.scal(0.5);
               }
               if (this->_intercept) output[output.n()-1]=0;
            }
      };

   template <typename T> 
      class normL2: public Regularizer<T> {
         public:
            normL2(const ParamReg<T>& param) : Regularizer<T>(param) { };
            virtual ~normL2() { };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.copy(x);
               if (this->_pos) y.thrsPos();
               Vector<T> xref(x.rawX(),this->_intercept ? x.n()-1 : x.n());
               const T nrm=xref.nrm2();
               if (nrm < lambda) {
                  y.setZeros();
               } else {
                  y.scal(T(1.0) - lambda/nrm);
               }
               if (this->_intercept) y[y.n()-1] = x[y.n()-1];
            };
            T inline eval(const Vector<T>& x) const { 
               Vector<T> xref(x.rawX(),this->_intercept ? x.n()-1 : x.n());
               return xref.nrm2(); 
            };
            /// TODO add subgradient
            void inline fenchel(const Vector<T>& input, T& val, T& scal) const {
               Vector<T> output;
               output.copy(input);
               if (this->_pos) output.thrsPos();
               T mm = output.nrm2();
               scal= mm > 1.0 ? T(1.0)/mm : 1.0;
               val=0;
               if (this->_intercept & abs<T>(output[output.n()-1]) > EPSILON) val=INFINITY; 
            };
      };

   template <typename T> 
      class normLINF: public Regularizer<T> {
         public:
            normLINF(const ParamReg<T>& param) : Regularizer<T>(param) { };
            virtual ~normLINF() { };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.copy(x);
               if (this->_pos) y.thrsPos();
               Vector<T> xref(y.rawX(),this->_intercept ? x.n()-1 : x.n());
               Vector<T> row(xref.n());
               xref.l1project(row,lambda);
               for (int j = 0; j<xref.n(); ++j)
                  y[j]=y[j]-row[j];
               if (this->_intercept) y[y.n()-1] = x[y.n()-1];
            };
            T inline eval(const Vector<T>& x) const { 
               Vector<T> xref(x.rawX(),this->_intercept ? x.n()-1 : x.n());
               return xref.fmaxval(); 
            };
            /// TODO add subgradient
            void inline fenchel(const Vector<T>& input, T& val, T& scal) const {
               Vector<T> output;
               output.copy(input);
               if (this->_pos) output.thrsPos();
               T mm = output.asum();
               scal= mm > 1.0 ? T(1.0)/mm : 1.0;
               val=0;
               if (this->_intercept & abs<T>(output[output.n()-1]) > EPSILON) val=INFINITY; 
            };
      };

   template <typename T, typename D, typename RegA, typename RegB, bool order = true, bool scale_lambda = false>
      class ComposeProx: public Regularizer<T,D> {
         public:
            ComposeProx(const ParamReg<T>& param) : Regularizer<T,D>(param) {
               _lambda2d1=param.lambda2d1;
               _regA=new RegA(param);
               _regB=new RegB(param);
            }
            virtual ~ComposeProx() { delete(_regA); delete(_regB); };

            void inline prox(const D& x, D& y, const T lambda) {
               D tmp;
               if (scale_lambda) {
                  if (order) {
                     _regA->prox(x,tmp,lambda);
                     _regB->prox(tmp,y,lambda*_lambda2d1/(T(1.0)+lambda));
                  } else {
                     _regB->prox(x,tmp,lambda*_lambda2d1);
                     _regA->prox(tmp,y,lambda/(T(1.0)+lambda*_lambda2d1));
                  }
               } else {
                  if (order) {
                     _regA->prox(x,tmp,lambda);
                     _regB->prox(tmp,y,lambda*_lambda2d1);
                  } else {
                     _regB->prox(x,tmp,lambda*_lambda2d1);
                     _regA->prox(tmp,y,lambda);
                  }
               }
            };
            T inline eval(const D& x) const { 
               return _regA->eval(x) + _lambda2d1*_regB->eval(x);
            };
            virtual bool is_fenchel() const { return false; };
            void inline fenchel(const D& input, T& val, T& scal) const { };
            virtual bool is_subgrad() const { return _regA->is_subgrad() && _regB->is_subgrad(); };
            virtual void sub_grad(const D& input, D& output) const {  
               _regA->sub_grad(input,output);
               D tmp;
               _regB->sub_grad(input,tmp);
               output.add(tmp,_lambda2d1);
            };
         private:
            RegA* _regA;
            RegB* _regB;
            T _lambda2d1;
      };

   template <typename T>
      struct ElasticNet {
         typedef ComposeProx< T, Vector<T>, Lasso<T>, Ridge<T>, true > type;
      };

   template <typename T> 
      class FusedLasso: public Regularizer<T> {
         public:
            FusedLasso(const ParamReg<T>& param) : Regularizer<T>(param) {
               _lambda2d1=param.lambda2d1;
               _lambda3d1=param.lambda3d1;
            };
            virtual ~FusedLasso() {  };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.resize(x.n());
               Vector<T> copyx;
               copyx.copy(x);
               copyx.fusedProjectHomotopy(y,_lambda2d1*lambda,lambda,_lambda3d1*lambda,true);
            };
            T inline eval(const Vector<T>& x) const { 
               T sum = T();
               const int maxn = this->_intercept ? x.n()-1 : x.n();
               for (int i = 0; i<maxn-1; ++i)
                  sum += abs(x[i+1]-x[i]) + _lambda2d1*abs(x[i]) + 0.5*_lambda3d1*x[i]*x[i];
               sum += _lambda2d1*abs(x[maxn-1])+0.5*_lambda3d1*x[maxn-1]*x[maxn-1];
               return sum;
            };
            virtual bool is_fenchel() const { return false; };
            void inline fenchel(const Vector<T>& input, T& val, T& scal) const { };

         private:
            T _lambda2d1;
            T _lambda3d1;
      };

   template <typename T> 
      class GraphLasso : public Regularizer<T>, public SplittingFunction<T, SpMatrix<T> > {
         public:
            GraphLasso(const ParamReg<T>& param) : Regularizer<T>(param) { 
               const bool resetflow = param.resetflow;
               const bool linf = param.linf;
               const bool clever = param.clever;
               const GraphStruct<T>& graph_st=*(param.graph_st);
               _clever=clever;
               _resetflow=resetflow;
               _graph.create_graph(graph_st.Nv,graph_st.Ng,graph_st.weights,
                     graph_st.gv_ir,graph_st.gv_jc,graph_st.gg_ir,graph_st.gg_jc);
               _graph.save_capacities();
               _work.resize(graph_st.Nv+graph_st.Ng+2);
               _weights.resize(graph_st.Ng);
               for (int i = 0; i<graph_st.Ng; ++i) _weights[i] = graph_st.weights[i];
               _old_lambda=-1.0;
               _linf=linf;
            };
            virtual ~GraphLasso() {   };

            void inline reset() { _old_lambda = -1.0; };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               if (!_linf) {
                  cerr << "Not implemented" << endl;
                  exit(1);
               }
               y.copy(x);
               _graph.restore_capacities();
               _graph.set_weights(_weights.rawX(),lambda);
               if (_old_lambda < 0 || _resetflow) {
                  _graph.reset_flow();
               } else {
                  if (lambda != _old_lambda)
                     _graph.scale_flow(lambda/_old_lambda);
               }
               if (this->_pos) {
                  Vector<T> xc;
                  xc.copy(x);
                  xc.thrsPos();
                  _graph.proximal_operator(xc.rawX(),y.rawX(),_clever);
               } else {
                  _graph.proximal_operator(x.rawX(),y.rawX(),_clever);
               }
#ifdef VERB2
               T duality_gap2 = y.nrm2sq()-y.dot(x)+lambda*this->eval(y);
               cerr << "duality_gap2 " << duality_gap2 << endl;
#endif
               _old_lambda=lambda;
            };

            T inline eval(const Vector<T>& x) const { 
               Graph<T>* gr = const_cast<Graph<T>* >(&_graph);
               gr->restore_capacities();
               return gr->norm(x.rawX(),_work.rawX(),_weights.rawX(),_linf);
            };
            virtual bool is_fenchel() const {
               return _linf;
            };
            void inline fenchel(const Vector<T>& input, T& val, T& scal) const {
               Graph<T>* gr = const_cast<Graph<T>* >(&_graph);
               if (!_resetflow) {
                  gr->save_flow();
               }
               gr->reset_flow();
               gr->restore_capacities();
               Vector<T> output;
               output.copy(input);
               if (this->_pos) output.thrsPos();
               T mm = gr->dual_norm_inf(output,_weights);
               if (!_resetflow)
                  gr->restore_flow();
               scal= mm > 1.0 ? T(1.0)/mm : 1.0;
               val=0;
               if (this->_intercept & abs<T>(input[input.n()-1]) > EPSILON) val=INFINITY; 
            };

            virtual void init(const Vector<T>& y) { };
            inline int num_components() const { return _weights.n(); };
            inline void prox_split(SpMatrix<T>& splitted_w, const T lambda) const {
               Vector<T> tmp;
               SpVector<T> col;
               if (_linf) {
                  for (int i = 0; i<splitted_w.n(); ++i) {
                     splitted_w.refCol(i,col);
                     tmp.setData(col.rawX(),col.nzmax());                     
                     Vector<T> res;
                     res.copy(tmp);
                     vAbs<T>(res.n(),res.rawX(),res.rawX());
                     T thrs=project_tree_l1(res.rawX(),res.n(),lambda);
                     tmp.thrsabsmin(thrs);
                  }
               } else {
                  for (int i = 0; i<splitted_w.n(); ++i) {
                     splitted_w.refCol(i,col);
                     tmp.setData(col.rawX(),col.nzmax());                     
                     const T nrm = tmp.nrm2();
                     if (nrm > lambda*_weights[i]) {
                        tmp.scal(T(1.0)-lambda*_weights[i]/nrm);
                     } else {
                        tmp.setZeros();
                     }
                  }
               }
            };
            inline void init_split_variables(SpMatrix<T>& splitted_w) const {
               Graph<T>* gr = const_cast<Graph<T>* >(&_graph);
               gr->init_split_variables(splitted_w);
            };
            inline T eval_split(const SpMatrix<T>& input) const {
               SpVector<T> col;
               T sum = 0;
               for (int i = 0; i<input.n(); ++i) {
                  input.refCol(i,col);
                  sum += _linf ? _weights[i]*col.fmaxval()  : _weights[i]*col.nrm2();
               }
               return sum;
            }
            inline T eval_weighted(const Vector<T>& input,
                  const SpMatrix<T>& input_struct, const T* inner_weight) const {
               SpVector<T> col;
               T sum = 0;
               Vector<T> tmp(input_struct.m());
               for (int i = 0; i<input_struct.n(); ++i) {
                  input_struct.refCol(i,col);
                  tmp.setn(col.L());
                  for (int j = 0; j<col.L(); ++j)
                     tmp[j]=inner_weight[j]*input[col.r(j)];
                  sum += _linf ? _weights[i]*tmp.fmaxval()  : _weights[i]*tmp.nrm2();
               }
               return sum;
            }


         private:
            bool _clever;
            Graph<T> _graph;
            bool _resetflow;
            Vector<T> _work;
            Vector<T> _weights;
            T _old_lambda;
            bool _linf;
      };

   template <typename T>
      struct GraphLassoRidge {
         typedef ComposeProx<T, Vector<T>, GraphLasso<T>, Ridge<T>, true> type;
      };

   template <typename T> 
      class TreeLasso : public Regularizer<T> {
         public:
            TreeLasso(const ParamReg<T>& param) : Regularizer<T>(param) {
               const TreeStruct<T>& tree_st=*(param.tree_st);
               const bool linf = param.linf;
               _tree.create_tree(tree_st.Nv,tree_st.own_variables,
                     tree_st.N_own_variables,tree_st.weights,
                     tree_st.groups_ir,tree_st.groups_jc,
                     tree_st.Ng,0);
               _linf=linf;
            };
            virtual ~TreeLasso() { };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.copy(x);
               if (this->_pos) y.thrsPos();
               Vector<T> yp;
               if (this->_intercept) {
                  yp.setData(y.rawX(),y.n()-1);
               } else {
                  yp.setData(y.rawX(),y.n());
               }
               _tree.proj(yp,_linf,lambda);
            };
            T inline eval(const Vector<T>& x) const { 
               return const_cast<Tree_Seq<T>* >(&_tree)->val_norm(x.rawX(),0,_linf);
            };
            void inline fenchel(const Vector<T>& y, T& val, T& scal) const {
               if (_linf) {
                  Vector<T> yp;
                  if (this->_intercept) {
                     yp.setData(y.rawX(),y.n()-1);
                  } else {
                     yp.setData(y.rawX(),y.n());
                  }
                  Vector<T> yp2;
                  yp2.copy(yp);
                  if (this->_pos) yp2.thrsPos();
                  T mm = const_cast<Tree_Seq<T>* >(&_tree)->dual_norm_inf(yp2);
                  scal= mm > 1.0 ? T(1.0)/mm : 1.0;
                  val=0;
                  if (this->_intercept & abs<T>(y[y.n()-1]) > EPSILON) val=INFINITY; 
               } 
            };
            virtual bool is_fenchel() const {
               return _linf;
            };
            virtual bool is_subgrad() const { return true; };
            virtual void sub_grad(const Vector<T>& input, Vector<T>& output) const {
               output.resize(input.n());
               const_cast<Tree_Seq<T>*>(&_tree)->sub_grad(input,output,_linf);
               if (this->_intercept) output[output.n()-1]=0;
            }
         private:
            Tree_Seq<T> _tree;
            bool _linf;
      };

   template <typename T> 
      class TreeLzero : public Regularizer<T> {
         public:
            TreeLzero(const ParamReg<T>& param) : Regularizer<T>(param) {
               const TreeStruct<T>& tree_st=*(param.tree_st);
               _tree.create_tree(tree_st.Nv,tree_st.own_variables,
                     tree_st.N_own_variables,tree_st.weights,
                     tree_st.groups_ir,tree_st.groups_jc,
                     tree_st.Ng,0);
            };
            virtual ~TreeLzero() { };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.copy(x);
               if (this->_pos) y.thrsPos();
               Vector<T> yp;
               if (this->_intercept) {
                  yp.setData(y.rawX(),y.n()-1);
               } else {
                  yp.setData(y.rawX(),y.n());
               }
               _tree.proj_zero(yp,lambda);
            };
            T inline eval(const Vector<T>& x) const { 
               return const_cast<Tree_Seq<T>* >(&_tree)->val_zero(x.rawX(),0);
            };
            virtual bool is_fenchel() const { return false; };
            void inline fenchel(const Vector<T>& y, T& val, T& scal) const {  };

         private:
            Tree_Seq<T> _tree;
      };

   template <typename T, typename ProxMat>
      class ProxMatToVec : public Regularizer<T> {
         public:
            ProxMatToVec(const ParamReg<T>& param) : Regularizer<T>(param) { 
               _size_group=param.size_group;
               ParamReg<T> param2=param;
               param2.intercept=false;
               _proxy = new ProxMat(param2);
            };
            virtual ~ProxMatToVec() { delete(_proxy); };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.resize(x.n());
               int size_vec=this->_intercept ? x.n()-1 : x.n();
               Matrix<T> mX(x.rawX(),_size_group,size_vec/_size_group);
               Matrix<T> mY(y.rawX(),_size_group,size_vec/_size_group);
               _proxy->prox(mX,mY,lambda);
               if (this->_intercept) y[y.n()-1]=x[x.n()-1];
            }
            T inline eval(const Vector<T>& x) const {
               int size_vec=this->_intercept ? x.n()-1 : x.n();
               Matrix<T> mX(x.rawX(),_size_group,size_vec/_size_group);
               return _proxy->eval(mX);
            }
            virtual bool is_fenchel() const { return (_proxy->is_fenchel()); };
            void inline fenchel(const Vector<T>& x, T& val, T& scal) const {
               int size_vec=this->_intercept ? x.n()-1 : x.n();
               Matrix<T> mX(x.rawX(),_size_group,size_vec/_size_group);
               _proxy->fenchel(mX,val,scal);
            };

         private:
            int _size_group;
            ProxMat* _proxy;
      };

   template <typename T, typename Reg>
      class GroupProx : public Regularizer<T> {
         public:
            GroupProx(const ParamReg<T> & param) : Regularizer<T>(param) {
               ParamReg<T> param2=param;
               param2.intercept=false;
               _size_group=param.size_group;
               if (param.groups) {
                  int num_groups=0;
                  for (int i = 0; i<param.ngroups; ++i) num_groups=MAX(num_groups,param.groups[i]);
                  _groups.resize(num_groups);
                  for (int i = 0; i<num_groups; ++i) _groups[i]=new list_int();
                  for (int i = 0; i<param.ngroups; ++i) _groups[param.groups[i]-1]->push_back(i); 
               } 
               _prox = new Reg(param2);
            }
            virtual ~GroupProx() { 
               delete(_prox); 
               for (int i = 0; i<_groups.size(); ++i) delete(_groups[i]);
            };

            void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
               y.copy(x);
               const int maxn= this->_intercept ? x.n()-1 : x.n();
               if (!_groups.empty()) {
                  for (int i = 0; i<_groups.size(); ++i) {
                     list_int* group=_groups[i];
                     Vector<T> tmp(group->size());
                     Vector<T> tmp2(group->size());
                     int count=0;
                     for (const_iterator_int it = group->begin(); it != group->end(); ++it) {
                        tmp[count++]=x[*it];
                     }
                     _prox->prox(tmp,tmp2,lambda);
                     count=0;
                     for (const_iterator_int it = group->begin(); it != group->end(); ++it) {
                        y[*it]=tmp2[count++];
                     }
                  }
               } else {
                  Vector<T> tmp;
                  Vector<T> tmp2;
                  const int p = _size_group;
                  for (int i = 0; i+p-1<maxn; i+=p) {
                     tmp.setPointer(x.rawX()+i,p);
                     tmp2.setPointer(y.rawX()+i,p);
                     _prox->prox(tmp,tmp2,lambda);
                  }
               }
            }
            T inline eval(const Vector<T>& x) const {
               const int maxn= this->_intercept ? x.n()-1 : x.n();
               T sum=0;
               if (!_groups.empty()) {
                  for (int i = 0; i<_groups.size(); ++i) {
                     list_int* group=_groups[i];
                     Vector<T> tmp(group->size());
                     int count=0;
                     for (const_iterator_int it = group->begin(); it != group->end(); ++it) {
                        tmp[count++]=x[*it];
                     }
                     sum+=_prox->eval(tmp);
                  }
               } else {
                  Vector<T> tmp;
                  const int p = _size_group;
                  for (int i = 0; i+p-1<maxn; i+=p) {
                     tmp.setPointer(x.rawX()+i,p);
                     sum+=_prox->eval(tmp);
                  }
               }
               return sum;
            }
            virtual bool is_fenchel() const { return _prox->is_fenchel(); };
            void inline fenchel(const Vector<T>& x, T& val, T& scal) const { 
               const int maxn= this->_intercept ? x.n()-1 : x.n();
               T val2;
               T scal2;
               scal=T(1.0);
               val=0;
               if (!_groups.empty()) {
                  for (int i = 0; i<_groups.size(); ++i) {
                     list_int* group=_groups[i];
                     Vector<T> tmp(group->size());
                     int count=0;
                     for (const_iterator_int it = group->begin(); it != group->end(); ++it) {
                        tmp[count++]=x[*it];
                     }
                     _prox->fenchel(tmp,val2,scal2);
                     val+=val2;
                     scal=MIN(scal,scal2);
                  }
               } else {
                  const int p = _size_group;
                  Vector<T> tmp;
                  for (int i = 0; i+p-1<maxn; i+=p) {
                     tmp.setPointer(x.rawX()+i,p);
                     _prox->fenchel(tmp,val2,scal2);
                     val+=val2;
                     scal=MIN(scal,scal2);
                  }
               }
            };
         protected:
            int _size_group;
            std::vector<list_int*> _groups;
            Reg* _prox;
      };

   template <typename T>
      struct GroupLassoL2 {
         typedef GroupProx<T, normL2<T> > type;
      };

   template <typename T>
      struct GroupLassoLINF {
         typedef GroupProx<T, normLINF<T> > type;
      };

   template <typename T>
      struct GroupLassoL2_L1 {
         typedef ComposeProx<T, Vector<T>, typename GroupLassoL2<T>::type, Lasso<T>, false> type;
      };

   template <typename T>
      struct GroupLassoLINF_L1 {
         typedef ComposeProx<T, Vector<T>, typename GroupLassoLINF<T>::type, Lasso<T>, false> type;
      };

   template <typename T>
      class MixedL1L2 : public Regularizer<T,Matrix<T> > {
         public:
            MixedL1L2(const ParamReg<T>& param) : Regularizer<T,Matrix<T> >(param) { };
            virtual ~MixedL1L2() { };

            void inline prox(const Matrix<T>& x, Matrix<T>& y, const T lambda) {
               Vector<T> norm;
               y.copy(x);
               if (this->_pos) y.thrsPos();
               y.norm_2_rows(norm);
               y.setZeros();
               const int m = x.m();
               const int n = x.n();
               for (int i = 0; i<m; ++i) {
                  if (norm[i] > lambda) {
                     T scal = (norm[i]-lambda)/norm[i];
                     for (int j = 0; j<n; ++j) 
                        y[j*m+i] = x[j*m+i]*scal;
                  }
               }
               if (this->_pos) y.thrsPos();
               if (this->_intercept)
                  for (int j = 0; j<n; ++j) 
                     y[j*m+m-1]=x[j*m+m-1];
            }
            T inline eval(const Matrix<T>& x) const {
               Vector<T> norm;
               x.norm_2_rows(norm);
               return this->_intercept ? norm.asum() - norm[norm.n() -1] : norm.asum();
            }
            virtual bool is_subgrad() const { return true; };
            virtual void sub_grad(const Matrix<T>& input, Matrix<T>& output) const { 
               Vector<T> norm;
               input.norm_2_rows(norm);
               for (int i = 0; i<norm.n(); ++i) {
                  if (norm[i] < 1e-20) norm[i]=T(1.0);
               }
               norm.inv();
               if (this->_intercept) norm[norm.n()-1]=0;
               output.copy(input);
               output.multDiagLeft(norm);
            };
            void inline fenchel(const Matrix<T>& input, T& val, T& scal) const {
               Vector<T> norm;
               if (this->_pos) {
                  Matrix<T> output;
                  output.copy(input);
                  output.thrsPos();
                  output.norm_2_rows(norm);
               } else {
                  input.norm_2_rows(norm);
               }
               T mm = norm.fmaxval();
               scal= mm > 1.0 ? T(1.0)/mm : 1.0; 
               val=0;
               if (this->_intercept & abs<T>(norm[norm.n()-1]) > EPSILON) val=INFINITY; 
            };
      };



   template <typename T>
      class MixedL1LINF : public Regularizer<T,Matrix<T> > {
         public:
            MixedL1LINF(const ParamReg<T>& param) : Regularizer<T,Matrix<T> >(param) { };
            virtual ~MixedL1LINF() { };

            void inline prox(const Matrix<T>& x, Matrix<T>& y, const T lambda) {
               y.copy(x);
               if (this->_pos) y.thrsPos();
               Vector<T> row(x.n());
               Vector<T> row2(x.n());
               const int maxn= this->_intercept ? x.m()-1 : x.m();
               for (int i = 0; i< maxn; ++i) {
                  for (int j = 0; j<x.n(); ++j)
                     row[j]=y(i,j);
                  row.l1project(row2,lambda);
                  for (int j = 0; j<x.n(); ++j)
                     y(i,j) = row[j]-row2[j];
               }
            }
            T inline eval(const Matrix<T>& x) const { 
               Vector<T> norm;
               x.norm_inf_rows(norm);
               return this->_intercept ? norm.asum() - norm[norm.n() -1] : norm.asum();
            }
            void inline fenchel(const Matrix<T>& input, T& val, T& scal) const {
               Vector<T> norm;
               if (this->_pos) {
                  Matrix<T> output;
                  output.copy(input);
                  output.thrsPos();
                  output.norm_l1_rows(norm);
               } else {
                  input.norm_l1_rows(norm);
               }
               if (this->_intercept) norm[norm.n()-1]=0;
               T mm = norm.fmaxval();
               scal= mm > 1.0 ? T(1.0)/mm : 1.0; 
               val=0;
               if (this->_intercept & abs<T>(norm[norm.n()-1]) > EPSILON) val=INFINITY; 
            };
            virtual bool is_subgrad() const { return true; };
            virtual void sub_grad(const Matrix<T>& input, Matrix<T>& output) const { 
               output.resize(input.m(),input.n());
               output.setZeros();
               const T maxm= this->_intercept ? input.m()-1 : input.m();
               Vector<T> row(input.n());
               for (int i = 0; i<maxm; ++i) {
                  input.copyRow(i,row);
                  T max=row.fmaxval();
                  if (max > 1e-15) {
                     int num_max=0;
                     for (int j = 0; j<row.n(); ++j) {
                        if (abs<T>(max-abs<T>(row[j])) < 1e-15) 
                           num_max++;
                     }
                     T add = T(1.0)/num_max;
                     for (int j = 0; j<row.n(); ++j) {
                        if (abs<T>(max-abs<T>(row[j])) < 1e-15) 
                           row[j] = row[j] > 0 ? add : -add;
                     }
                     output.setRow(i,row);
                  }
               }
            };
      };

   template <typename T>
      class TraceNorm : public Regularizer<T,Matrix<T> > {
         public:
            TraceNorm(const ParamReg<T>& param) : Regularizer<T,Matrix<T> >(param) { 
               if (param.intercept) {
                  cerr << "Trace norm implementation is not compatible with intercept, intercept deactivated" << endl;
               }
               if (param.pos) {
                  cerr << "Trace norm implementation is not compatible with non-negativity constraints" << endl;
               }

            };
            virtual ~TraceNorm() { };

            void inline prox(const Matrix<T>& x, Matrix<T>& y, const T lambda) {
               //Matrix<T> tmp;               
               //tmp.copy(x);
               Matrix<T> U;
               Matrix<T> V;
               Vector<T> S;
               x.svd(U,S,V);
               S.softThrshold(lambda);
               U.multDiagRight(S);
               U.mult(V,y);
              /* Vector<T> u0(x.m());
               u0.setZeros();
               Vector<T> u, v;
               for (int i = 0; i<MIN(x.m(),x.n()); ++i) {
                  tmp.svdRankOne(u0,u,v);
                  T val=v.nrm2();
                  if (val < lambda) break;
                  y.rank1Update(u,v,(val-lambda)/val);
                  tmp.rank1Update(u,v,-T(1.0));
               }*/
            }
            T inline eval(const Matrix<T>& x) const {
               Vector<T> tmp;
               x.singularValues(tmp);
               return tmp.sum();
            /*   Matrix<T> XtX;
               if (x.m() > x.n()) {
                  x.XtX(XtX);
               } else {
                  x.XXt(XtX);
               }
               T sum=0;
               Vector<T> u0(XtX.m());
               u0.setAleat();
               for (int i = 0; i<XtX.m(); ++i) {
                  T val=XtX.eigLargestMagnSym(u0,u0); // uses power method
                  XtX.rank1Update(u0,u0,-val);
                  sum+=sqrt(val);
                  if (val <= 1e-10) break;
               }
               return sum;
               */
            }
            void inline fenchel(const Matrix<T>& input, T& val, T& scal) const {
               //Vector<T> u0(input.m());
               //u0.setZeros();
               //Vector<T> u, v;
               //input.svdRankOne(u0,u,v);
               //T mm = v.nrm2();
               Vector<T> tmp;
               input.singularValues(tmp);
               T mm = tmp.fmaxval();
               scal= mm > 1.0 ? T(1.0)/mm : 1.0;
               val=0;
            };
      };


   template <typename T>
      class Rank : public Regularizer<T,Matrix<T> > {
         public:
            Rank(const ParamReg<T>& param) : Regularizer<T,Matrix<T> >(param) { 
               if (param.intercept) {
                  cerr << "Rank implementation is not compatible with intercept, intercept deactivated" << endl;
               }
               if (param.pos) {
                  cerr << "Rank implementation is not compatible with non-negativity constraints" << endl;
               }

            };
            virtual ~Rank() { };

            void inline prox(const Matrix<T>& x, Matrix<T>& y, const T lambda) {
               Matrix<T> tmp;               
               tmp.copy(x);
               y.resize(x.m(),x.n());
               y.setZeros();
               Vector<T> u0(x.m());
               u0.setZeros();
               Vector<T> u, v;
               for (int i = 0; i<MIN(x.m(),x.n()); ++i) {
                  tmp.svdRankOne(u0,u,v);
                  T val=v.nrm2();
                  if (val*val < lambda) break;
                  y.rank1Update(u,v);
                  tmp.rank1Update(u,v,-T(1.0));
               }
            }
            T inline eval(const Matrix<T>& x) const {
               Matrix<T> XtX;
               if (x.m() > x.n()) {
                  x.XtX(XtX);
               } else {
                  x.XXt(XtX);
               }
               T sum=0;
               Vector<T> u0(XtX.m());
               u0.setAleat();
               for (int i = 0; i<XtX.m(); ++i) {
                  T val=XtX.eigLargestMagnSym(u0,u0); // uses power method
                  XtX.rank1Update(u0,u0,-val);
                  sum++;
                  if (val <= 1e-10) break;
               }
               return sum;
            }
            virtual bool is_fenchel() const { return false; };
            void inline fenchel(const Matrix<T>& input, T& val, T& scal) const { };
      };

    template <typename T>
       inline void convert_paths_to_mat(const List<Path<long long>*>& paths,SpMatrix<T>& paths_mat, const int n) {
          int nzmax=0;
          for (ListIterator<Path<long long>*> it=paths.begin(); it != paths.end(); ++it)
             nzmax+=it->nodes.size();
          paths_mat.resize(n,paths.size(),nzmax);
          int* pB =paths_mat.pB();
          int* pE =paths_mat.pE();
          int* r =paths_mat.r();
          T* v =paths_mat.v();
          int count_col=0;
          int count=0;
          pB[0]=0;
          for (ListIterator<Path<long long>*> it_path=paths.begin(); 
                it_path != paths.end(); ++it_path) {
             for (const_iterator_int it = it_path->nodes.begin(); 
                   it != it_path->nodes.end(); ++it) {
                r[count]= *it;
                v[count++]= it_path->flow;
             }
             pB[++count_col]=count;
          }
          for (int i = 0; i<paths_mat.n(); ++i) sort(r,v,pB[i],pE[i]-1);
       };
 
    template <typename T> 
       class GraphPathL0 : public Regularizer<T> {
          public:
             GraphPathL0(const ParamReg<T>& param) : Regularizer<T>(param) {
                const GraphPathStruct<T>& graph=*(param.graph_path_st);
                _graph.init_graph(graph);
             }
             virtual ~GraphPathL0() { };
 
             void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
                // DEBUG
                y.copy(x);
                if (this->_pos) y.thrsPos();
                _graph.proximal_l0(y.rawX(),lambda);
             };
             T inline eval(const Vector<T>& x) const { 
                return const_cast<GraphPath<T>* >(&_graph)->eval_l0(x.rawX());
             };
             T inline eval_paths(const Vector<T>& x, SpMatrix<T>& paths_mat) const { 
                List<Path<long long>*> paths;
                T val=const_cast<GraphPath<T>* >(&_graph)->eval_l0(x.rawX(),&paths);
                convert_paths_to_mat<T>(paths,paths_mat,_graph.n());
                for (ListIterator<Path<>*> it_path=paths.begin(); 
                      it_path != paths.end(); ++it_path) delete(*it_path);
                return val;
             };
 
             virtual bool is_fenchel() const { return false; };
             void inline fenchel(const Vector<T>& input, T& val, T& scal) const {  };
 
          private:
             GraphPath<T> _graph;
       };
 
    template <typename T> 
       class GraphPathConv : public Regularizer<T> {
          public:
             GraphPathConv(const ParamReg<T>& param) : Regularizer<T>(param) {
                const GraphPathStruct<T>& graph=*(param.graph_path_st);
                _graph.init_graph(graph);
             }
             virtual ~GraphPathConv() {  };
 
             void inline prox(const Vector<T>& x, Vector<T>& y, const T lambda) {
                y.copy(x);
                if (this->_pos) y.thrsPos();
                _graph.proximal_conv(y.rawX(),lambda);
             };
             T inline eval(const Vector<T>& x) const { 
                return const_cast<GraphPath<T>* >(&_graph)->eval_conv(x.rawX());
             };
             T inline eval_dual_norm(const Vector<T>& x) const { 
                return const_cast<GraphPath<T>* >(&_graph)->eval_dual_norm(x.rawX(),NULL);
             };
             T inline eval_paths(const Vector<T>& x, SpMatrix<T>& paths_mat) const { 
                List<Path<long long>*> paths;
                T val=const_cast<GraphPath<T>* >(&_graph)->eval_conv(x.rawX(),&paths);
                convert_paths_to_mat<T>(paths,paths_mat,_graph.n());
                for (ListIterator<Path<long long>*> it_path=paths.begin(); 
                      it_path != paths.end(); ++it_path) delete(*it_path);
                return val;
             };
             T inline eval_dual_norm_paths(const Vector<T>& x, SpMatrix<T>& paths_mat) const { 
                Path<long long> path;
                T val=const_cast<GraphPath<T>* >(&_graph)->eval_dual_norm(x.rawX(),&path.nodes);
                List<Path<long long>*> paths;
                paths.push_back(&path);
                path.flow_int=1;
                path.flow=double(1.0);
                convert_paths_to_mat<T>(paths,paths_mat,_graph.n());
                return val;
             };
             virtual bool is_fenchel() const { return true; };

             void inline fenchel(const Vector<T>& input, T& val, T& scal) const {
                T mm;
                if (this->_pos) {
                   Vector<T> output;
                   output.copy(input);
                   output.thrsPos();
                   mm = const_cast<GraphPath<T>* >(&_graph)->eval_dual_norm(output.rawX(),NULL);
                } else {
                   mm = const_cast<GraphPath<T>* >(&_graph)->eval_dual_norm(input.rawX(),NULL);
                }
                scal= mm > 1.0 ? T(1.0)/mm : 1.0;
                val=0;
                if (this->_intercept & abs<T>(input[input.n()-1]) > EPSILON) val=INFINITY; 
             };
          private:
             GraphPath<T> _graph;
       };


   template <typename T,typename Reg>
      class RegMat : public Regularizer<T,Matrix<T> > {
         public:
            RegMat(const ParamReg<T>& param) : Regularizer<T,Matrix<T> >(param) { 
               _transpose=param.transpose;
               const int N = param.num_cols;
               _regs=new Reg*[N];
               _N=N;
               for (int i = 0; i<N; ++i) 
                  _regs[i]=new Reg(param);
            };
            virtual ~RegMat() { 
               for (int i = 0; i<_N; ++i) {
                  delete(_regs[i]);
                  _regs[i]=NULL;
               }
               delete[](_regs);
            };
            void inline reset() { 
               for (int i = 0; i<_N; ++i) _regs[i]->reset();
            };
            void inline prox(const Matrix<T>& x, Matrix<T>& y, const T lambda) {
               y.copy(x);
               int i;
               if (_transpose) {
#pragma omp parallel for private(i) 
                  for (i = 0; i<_N; ++i) {
                     Vector<T> colx, coly;
                     x.copyRow(i,colx);
                     _regs[i]->prox(colx,coly,lambda);
                     y.setRow(i,coly);
                  }
               } else {
#pragma omp parallel for private(i) 
                  for (i = 0; i<_N; ++i) {
                     Vector<T> colx, coly;
                     x.refCol(i,colx);
                     y.refCol(i,coly);
                     _regs[i]->prox(colx,coly,lambda);
                  }
               }
            };
            virtual bool is_subgrad() const { 
               bool ok=true;
               for (int i = 0; i<_N; ++i) 
                  ok=ok && _regs[i]->is_subgrad();
               return ok;
            };
            void inline sub_grad(const Matrix<T>& x, Matrix<T>& y) const {
               y.resize(x.m(),x.n());
               Vector<T> colx, coly, cold;
               if (_transpose) {
                  for (int i = 0; i<_N; ++i) {
                     x.copyRow(i,colx);
                     _regs[i]->sub_grad(colx,coly);
                     y.setRow(i,coly);
                  }
               } else {
                  for (int i = 0; i<_N; ++i) {
                     x.refCol(i,colx);
                     y.refCol(i,coly);
                     _regs[i]->sub_grad(colx,coly);
                  }
               }
            };
            T inline eval(const Matrix<T>& x) const { 
               T sum = 0;
               int i;
#pragma omp parallel for private(i) 
               for (i = 0; i<_N; ++i) {
                  Vector<T> col;
                  if (_transpose) {
                     x.copyRow(i,col);
                  } else {
                     x.refCol(i,col);
                  }
#pragma omp critical
                  sum += _regs[i]->eval(col);
               }
               return sum;
            };
            void inline fenchel(const Matrix<T>& input, T& val, T& scal) const {
               Vector<T> col;
               val = 0;
               scal = 1.0;
               for (int i = 0; i<_N; ++i) {
                  if (_transpose) {
                     input.copyRow(i,col);
                  } else {
                     input.refCol(i,col);
                  }
                  T val2 = 0;
                  T scal2 = 1.0;
                  _regs[i]->fenchel(col,val2,scal2);
                  scal=MIN(scal,scal2);
                  val += val2;
               }
            };
            virtual bool is_fenchel() const {
               bool ok=true;
               for (int i = 0; i<_N; ++i) 
                  ok = ok && _regs[i]->is_fenchel();
               return ok;
            };

         protected:
            int _N;
            Reg** _regs;
            bool _transpose;
      };

   template <typename T>
      struct MixedL1L2_L1 {
         typedef ComposeProx<T, Matrix<T>, MixedL1L2<T>, RegMat<T, Lasso<T> >, false> type;
      };

   template <typename T>
      struct MixedL1LINF_L1 {
         typedef ComposeProx<T, Matrix<T>, MixedL1LINF<T>, RegMat<T, Lasso<T> >, false> type;
      };

   template <typename T>
      class SpecGraphMat : public Regularizer<T,Matrix<T> > {
         public:
            SpecGraphMat(const ParamReg<T>& param) : Regularizer<T,Matrix<T> >(param) {  };
            virtual ~SpecGraphMat() { delete(_graphlasso); };

            virtual void dummy() = 0;

            void inline reset() {  _graphlasso->reset(); };

            void inline prox(const Matrix<T>& x, Matrix<T>& y, const T lambda) {
               Vector<T> xv, yv;
               x.toVect(xv);
               y.resize(x.m(),x.n());
               y.toVect(yv);
               _graphlasso->prox(xv,yv,lambda);
            }
            T inline eval(const Matrix<T>& X) const { 
               Vector<T> xv;
               X.toVect(xv);
               return _graphlasso->eval(xv);
            }

            void inline fenchel(const Matrix<T>& input, T& val, T& scal) const {
               Vector<T> inv;
               input.toVect(inv);
               _graphlasso->fenchel(inv,val,scal);
            };
            virtual bool is_fenchel() const {
               return _graphlasso->is_fenchel();
            };

         protected:
            GraphLasso<T>* _graphlasso;
      };

   template <typename T>
      class MixedL1LINFCR : public SpecGraphMat<T> {
         public:
            MixedL1LINFCR(const int m, const ParamReg<T>& param) : SpecGraphMat<T>(param) { 
               const int n = param.num_cols;
               const T l2dl1 = param.lambda2d1;
               GraphStruct<T> graph_st;
               graph_st.Nv=m*n;
               graph_st.Ng=m+n;
               T* weights = new T[graph_st.Ng];
               for (int i = 0; i<n; ++i) weights[i]=T(1.0);
               for (int i = 0; i<m; ++i) weights[i+n]=l2dl1;
               graph_st.weights=weights;

               mwSize* gv_jc = new mwSize[graph_st.Ng+1];
               mwSize* gv_ir = new mwSize[m*n*2];
               for (int i = 0; i<n; ++i) {
                  gv_jc[i]=i*m;
                  for (int j = 0; j<m; ++j)
                     gv_ir[i*m+j]=i*m+j;
               }
               for (int i = 0; i<m; ++i) {
                  gv_jc[i+n]=i*n+n*m;
                  for (int j = 0; j<n; ++j)
                     gv_ir[i*n+n*m+j]=j*m+i;
               }
               gv_jc[m+n]=2*m*n;
               graph_st.gv_jc=gv_jc;
               graph_st.gv_ir=gv_ir;

               mwSize* gg_jc = new mwSize[graph_st.Ng+1];
               mwSize* gg_ir = new mwSize[1];
               for (int i = 0; i< graph_st.Ng+1; ++i) gg_jc[i]=0;
               graph_st.gg_jc=gg_jc;
               graph_st.gg_ir=gg_ir;

               ParamReg<T> param_lasso = param;
               param_lasso.graph_st = &graph_st;
               this->_graphlasso = new GraphLasso<T>(param_lasso);

               delete[](weights);
               delete[](gv_jc);
               delete[](gv_ir);
               delete[](gg_jc);
               delete[](gg_ir);
            };
            virtual ~MixedL1LINFCR() { };
            virtual void dummy() { };
      };


   template <typename T>
      class TreeMult : public SpecGraphMat<T> {
         public:
            TreeMult(const ParamReg<T>& param) : SpecGraphMat<T>(param) { 
               const TreeStruct<T>& tree_st=*(param.tree_st);
               const int N = param.num_cols;
               const T l1dl2 = param.lambda2d1;
               GraphStruct<T> graph_st;
               int Nv=tree_st.Nv;
               if (param.intercept) ++Nv;
               int Ng=tree_st.Ng;
               graph_st.Nv=Nv*N;
               graph_st.Ng=Ng*(N+1);
               T* weights=new T[graph_st.Ng];
               for (int i = 0; i<N+1; ++i)
                  for (int j = 0; j<Ng; ++j)
                     weights[i*Ng+j]=tree_st.weights[j];
               for (int j = 0; j<Ng; ++j)
                  weights[N*Ng+j]*=l1dl2;
               graph_st.weights=weights;

               int nzmax_tree=0;
               for (int i = 0; i<Ng; ++i)
                  nzmax_tree += tree_st.N_own_variables[i];
               int nzmax_v=nzmax_tree*N;
               mwSize* gv_jc = new mwSize[graph_st.Ng+1];
               mwSize* gv_ir = new mwSize[nzmax_v];
               int count=0;
               for (int i = 0; i<N; ++i) {
                  for (int j = 0; j<Ng; ++j) {
                     gv_jc[i*Ng+j]=count;
                     for (int k = 0; k<tree_st.N_own_variables[j]; ++k) {
                        gv_ir[gv_jc[i*Ng+j] + k] =Nv*i+tree_st.own_variables[j]+k;
                        ++count;
                     }
                  }
               }
               for (int i = 0; i<Ng+1; ++i) {
                  gv_jc[N*Ng+i]=count;
               }
               graph_st.gv_jc=gv_jc;
               graph_st.gv_ir=gv_ir;

               mwSize* gg_jc = new mwSize[graph_st.Ng+1];
               int nzmax_tree2=tree_st.groups_jc[Ng];
               int nzmax2=nzmax_tree2*(N+1)+Ng*N;
               mwSize* gg_ir = new mwSize[nzmax2];
               count=0;
               for (int i = 0; i<N; ++i) {
                  for (int j = 0; j<Ng; ++j) {
                     gg_jc[i*Ng+j] = count;
                     for (int k = tree_st.groups_jc[j]; k<static_cast<int>(tree_st.groups_jc[j+1]); ++k) {
                        gg_ir[count++] = i*Ng+tree_st.groups_ir[k];
                     }
                  }
               }
               for (int i = 0; i<Ng; ++i) {
                  gg_jc[N*Ng+i] = count;
                  for (int j = tree_st.groups_jc[i]; j<static_cast<int>(tree_st.groups_jc[i+1]); ++j) {
                     gg_ir[count++] = N*Ng+tree_st.groups_ir[j];
                  }
                  for (int j = 0; j<N; ++j) {
                     gg_ir[count++] = j*Ng+i;
                  }
               }
               gg_jc[(N+1)*Ng]=nzmax2;

               graph_st.gg_jc=gg_jc;
               graph_st.gg_ir=gg_ir;
               //              param.graph_st=&graph_st;
               ParamReg<T> param_lasso = param;
               param_lasso.graph_st=&graph_st;
               this->_graphlasso = new GraphLasso<T>(param_lasso);

               delete[](weights);
               delete[](gv_ir);
               delete[](gv_jc);
               delete[](gg_ir);
               delete[](gg_jc);
            };
            virtual void dummy() { };
            virtual ~TreeMult() { };
      };

   template <typename T>
      class GraphMult : public SpecGraphMat<T> {
         public:
            GraphMult(const ParamReg<T>& param) : SpecGraphMat<T>(param) { 
               const GraphStruct<T>& graph_st=*(param.graph_st);
               const int N = param.num_cols;
               const T l1dl2 = param.lambda2d1;
               GraphStruct<T> g_st;
               int Nv=graph_st.Nv;
               int Ng=graph_st.Ng;
               g_st.Nv=Nv*N;
               g_st.Ng=Ng*(N+1);
               T* weights=new T[g_st.Ng];
               for (int i = 0; i<N+1; ++i)
                  for (int j = 0; j<Ng; ++j)
                     weights[i*Ng+j]=graph_st.weights[j];
               for (int j = 0; j<Ng; ++j)
                  weights[N*Ng+j]*=l1dl2;
               g_st.weights=weights;
               int nzmax_graph=graph_st.gv_jc[Ng]; //just corrected to gv
               int nzmax_v=nzmax_graph*N;
               mwSize* gv_jc = new mwSize[g_st.Ng+1];
               mwSize* gv_ir = new mwSize[nzmax_v];
               int count=0;
               for (int i = 0; i<N; ++i) {
                  for (int j = 0; j<Ng; ++j) {
                     gv_jc[i*Ng+j]=count;
                     for (int k = graph_st.gv_jc[j]; k<graph_st.gv_jc[j+1]; ++k) {
                        gv_ir[count++] =Nv*i+graph_st.gv_ir[k]; 
                     }
                  }
               }
               for (int i = 0; i<Ng+1; ++i) {
                  gv_jc[N*Ng+i]=count;
               }
               g_st.gv_jc=gv_jc;
               g_st.gv_ir=gv_ir;

               mwSize* gg_jc = new mwSize[g_st.Ng+1];
               int nzmax_tree2=graph_st.gg_jc[Ng];
               int nzmax2=nzmax_tree2*(N+1)+Ng*N;
               mwSize* gg_ir = new mwSize[nzmax2];
               count=0;
               for (int i = 0; i<N; ++i) {
                  for (int j = 0; j<Ng; ++j) {
                     gg_jc[i*Ng+j] = count;
                     for (int k = graph_st.gg_jc[j]; k<graph_st.gg_jc[j+1]; ++k) {
                        gg_ir[count++] = i*Ng+graph_st.gg_ir[k];
                     }
                  }
               }
               for (int i = 0; i<Ng; ++i) {
                  gg_jc[N*Ng+i] = count;
                  for (int j = graph_st.gg_jc[i]; j<static_cast<int>(graph_st.gg_jc[i+1]); ++j) {
                     gg_ir[count++] = N*Ng+graph_st.gg_ir[j];
                  }
                  for (int j = 0; j<N; ++j) {
                     gg_ir[count++] = j*Ng+i;
                  }
               }
               gg_jc[(N+1)*Ng]=nzmax2;

               g_st.gg_jc=gg_jc;
               g_st.gg_ir=gg_ir;
               ParamReg<T> param_lasso = param;
               param_lasso.graph_st = &g_st;
               this->_graphlasso = new GraphLasso<T>(param_lasso);

               delete[](weights);
               delete[](gv_ir);
               delete[](gv_jc);
               delete[](gg_ir);
               delete[](gg_jc);
            };
            virtual void dummy() { };
            virtual ~GraphMult() { };
      };

   template <typename T, typename D, typename E>
      T duality_gap(Loss<T,D,E>& loss, Regularizer<T,D>& regularizer, const D& x, 
            const T lambda, T& best_dual, const bool verbose = false) {
         if (!regularizer.is_fenchel() || !loss.is_fenchel()) {
            cerr << "Error: no duality gap available" << endl;
            exit(1);
         }
         T primal= loss.eval(x)+lambda*regularizer.eval(x);
         bool intercept=regularizer.is_intercept();
         D grad1, grad2;
         loss.var_fenchel(x,grad1,grad2,intercept); 
         grad2.scal(-T(1.0)/lambda);
         T val=0;
         T scal=1.0;
         regularizer.fenchel(grad2,val,scal);
         T dual = -lambda*val;
         grad1.scal(scal);
         dual -= loss.fenchel(grad1);
         dual = MAX(dual,best_dual);
         T delta= primal == 0 ? 0 : (primal-dual)/abs<T>(primal);
         if (verbose) {
            cout << "Relative duality gap: " << delta << endl;
            flush(cout);
         }
         best_dual=dual;
         return delta;
      }

   template <typename T, typename D, typename E>
      T duality_gap(Loss<T,D,E>& loss, Regularizer<T,D>& regularizer, const D& x, 
            const T lambda, const bool verbose = false) {
         T best_dual=-INFINITY;
         return duality_gap(loss,regularizer,x,lambda,best_dual,verbose);
      }

   template <typename T>
      void dualityGraph(const Matrix<T>& X, const Matrix<T>& D, const Matrix<T>& alpha0,
            Vector<T>& res, const ParamFISTA<T>& param, 
            const GraphStruct<T>* graph_st) {
         Regularizer<T>* regularizer=new GraphLasso<T>(*graph_st,
               param.intercept,param.resetflow,param.pos,param.clever);
         Loss<T>* loss;
         switch (param.loss) {
            case SQUARE: loss=new SqLoss<T>(D);  break;
            case LOG:  loss = new LogLoss<T>(D); break;
            case LOGWEIGHT:  loss = new LogLoss<T,true>(D); break;
            default: cerr << "Not implemented"; exit(1);
         }
         Vector<T> Xi;
         X.refCol(0,Xi);
         loss->init(Xi);
         Vector<T> alpha0i;
         alpha0.refCol(0,alpha0i);
         regularizer->reset();
         res[0]=loss->eval(alpha0i)+param.lambda*regularizer->eval(alpha0i);
         res[1]=duality_gap(*loss,*regularizer,alpha0i,param.lambda);
         delete(loss);
         delete(regularizer);
      }

   template <typename T>
      void writeLog(const int iter, const T time, const T primal, const T dual,
            char* name) {
         std::ofstream f;
         f.precision(12);
         f.flags(std::ios_base::scientific);
         f.open(name, ofstream::app);
         f << iter << " " << primal << " " << dual << " " << time << std::endl;
         f.close();
      };


   template <typename T, typename D, typename E>
      void subGradientDescent_Generic(Loss<T,D,E>& loss, Regularizer<T,D>& regularizer, const D& x0, D& x, 
            Vector<T>& optim_info,
            const ParamFISTA<T>& param) {
         D grad;
         D sub_grad;
         const T lambda=param.lambda;
         const int it0 = MAX(1,param.it0);
         const bool duality = loss.is_fenchel() && regularizer.is_fenchel();
         optim_info.set(-1);
         T best_dual=-INFINITY;
         T rel_duality_gap=-INFINITY;
         Timer time;
         time.start();
         int it;
         for (it = 1; it<=param.max_it; ++it) {
            /// print loss
            if (param.verbose && ((it % it0) == 0)) {
               time.stop();
               T los=loss.eval(x) + lambda*regularizer.eval(x);    
               optim_info[0]=los;
               T sec=time.getElapsed();
               cout << "Iter: " << it << ", loss: " << los << ", time: " << sec << " ";
               if (param.log) 
                  writeLog(it,sec,los,best_dual,param.logName);
               if (param.verbose) 
                  cout << endl;
               flush(cout);
               time.start();
            }

            /// compute gradient
            loss.grad(x,grad);
            regularizer.sub_grad(x,sub_grad);
            T step = param.sqrt_step ? param.a/(param.b+sqrt(static_cast<T>(it))) : param.a/(param.b+(static_cast<T>(it)));
            x.add(grad,-step);
            x.add(sub_grad,-lambda*step);
            if (duality && ((it % it0) == 0)) {
               time.stop();
               rel_duality_gap=duality_gap(loss,regularizer,x,lambda,best_dual,param.verbose);
               optim_info[1]=best_dual;
               optim_info[2]=rel_duality_gap;
               if (rel_duality_gap < param.tol) break;
               time.start();
            }
         }
         if ((it % it0) != 0 || !param.verbose) {
            T los=loss.eval(x) + lambda*regularizer.eval(x);    
            optim_info[0]=los;
            if (duality) {
               rel_duality_gap=duality_gap(loss,regularizer,x,lambda,best_dual,param.verbose);
               optim_info[1]=best_dual;
               optim_info[2]=rel_duality_gap;
            }
         }
         optim_info[3]=it;
      }

   template <typename T, typename D, typename E>
      void ISTA_Generic(Loss<T,D,E>& loss, Regularizer<T,D>& regularizer, const D& x0, D& x, Vector<T>& optim_info,
            const ParamFISTA<T>& param) {
         const int it0 = MAX(1,param.it0);
         const T lambda=param.lambda;
         T L=param.L0;
         T Lold=L;
         x.copy(x0);
         D grad, tmp, prox, old;

         const bool duality = loss.is_fenchel() && regularizer.is_fenchel();
         optim_info.set(-1);
         Timer time;
         time.start();
         T rel_duality_gap=-INFINITY;

         int it;
         T best_dual=-INFINITY;
         for (it = 1; it<=param.max_it; ++it) {
            /// print loss
            if (param.verbose && ((it % it0) == 0)) {
               time.stop();
               T los=loss.eval(x) + lambda*regularizer.eval(x);
               optim_info[0]=los;
               T sec=time.getElapsed();
               cout << "Iter: " << it << ", loss: " << los << ", time: " << sec << ", L: " << L;
               flush(cout);
               if (param.log) 
                  writeLog(it,sec,los,best_dual,param.logName);
               time.start();
            }

            /// compute gradient
            loss.grad(x,grad);
            int iter=1;
            while (iter < param.max_iter_backtracking) {
               prox.copy(x);
               prox.add(grad,-T(1.0)/L);
               regularizer.prox(prox,tmp,lambda/L);

               Lold=L;
               if (loss.test_backtracking(x,grad,tmp,L)) {
                  break;
               }
               L *= param.gamma;
               if (param.verbose && ((it % it0) == 0)) 
                  cout << " " << L;
               ++iter;
            }
            if (param.verbose && ((it % it0) == 0)) 
               cout << endl;
            old.copy(x);
            x.copy(tmp);
            if (duality) {
               if ((it % it0) == 0) {
                  time.stop();
                  rel_duality_gap=duality_gap(loss,regularizer,x,lambda,best_dual,param.verbose);
                  optim_info[1]=best_dual;
                  optim_info[2]=rel_duality_gap;
                  if (rel_duality_gap < param.tol) break;
                  time.start();
               }
            } else {
               old.sub(x);
               if (sqrt(old.nrm2sq()/MAX(EPSILON,x.nrm2sq())) < param.tol) break;
            }
         }
         T los=loss.eval(x) + lambda*regularizer.eval(x);    
         optim_info[0]=los;
         T sec=time.getElapsed();
         if (param.verbose) {
            cout << "Iter: " << it << ", loss: " << los << ", time: " << sec << ", L: " << L << endl;
            flush(cout);
         }
         if (duality) {
            rel_duality_gap=duality_gap(loss,regularizer,x,lambda,best_dual,param.verbose);
            optim_info[1]=best_dual;
            optim_info[2]=rel_duality_gap;
         }
         optim_info[3]=it;
      }

   template <typename T, typename D, typename E>
      void FISTA_Generic(Loss<T,D,E>& loss, Regularizer<T,D>& regularizer, const D& x0, D& x, Vector<T>& optim_info,
            const ParamFISTA<T>& param) {
         const int it0 = MAX(1,param.it0);
         const T lambda=param.lambda;
         T L=param.L0;
         T t = 1.0;
         T Lold=L;
         T old_t;
         D y, grad, prox, tmp;
         y.copy(x0);
         x.copy(x0);

         const bool duality = loss.is_fenchel() && regularizer.is_fenchel();
         T rel_duality_gap=-INFINITY;
         optim_info.set(-1);

         Timer time;
         time.start();

         int it;
         T best_dual=-INFINITY;
         for (it = 1; it<=param.max_it; ++it) {
            /// print loss
            if (param.verbose && ((it % it0) == 0)) {
               time.stop();
               T los=loss.eval(x) + lambda*regularizer.eval(x);    
               optim_info[0]=los;
               T sec=time.getElapsed();
               cout << "Iter: " << it << ", loss: " << los << ", time: " << sec << ", L: " << L;
               flush(cout);
               if (param.log) 
                  writeLog(it,sec,los,best_dual,param.logName);
               time.start();
            }

            /// compute gradient
            loss.grad(y,grad);

            int iter=1;

            while (iter < param.max_iter_backtracking) {
               prox.copy(y);
               prox.add(grad,-T(1.0)/L);
               regularizer.prox(prox,tmp,lambda/L);
               Lold=L;
               if (param.fixed_step || loss.test_backtracking(y,grad,tmp,L)) break;
               L *= param.gamma;
               if (param.verbose && ((it % it0) == 0)) 
                  cout << " " << L;
               ++iter;
            }
            if (param.verbose && ((it % it0) == 0)) 
               cout << endl;

            prox.copy(x);
            prox.sub(tmp);
            x.copy(tmp);
            old_t=t;
            t=(1.0+sqrt(1+4*t*t))/2;
            y.copy(x);
            y.add(prox,(1-old_t)/t);
            if (duality) {
               if ((it % it0) == 0) {
                  time.stop();
                  rel_duality_gap=duality_gap(loss,regularizer,x,lambda,best_dual,param.verbose);
                  optim_info[1]=best_dual;
                  optim_info[2]=rel_duality_gap;
                  if (rel_duality_gap < param.tol) break;
                  time.start();
               }
            } else {
               if (sqrt(prox.nrm2sq()/MAX(EPSILON,x.nrm2sq())) < param.tol) break;
            }
         }
         T los=loss.eval(x) + lambda*regularizer.eval(x);
         optim_info[0]=los;
         T sec=time.getElapsed();
         if (param.verbose) {
            cout << "Iter: " << it << ", loss: " << los << ", time: " << sec << ", L: " << L << endl;
            flush(cout);
         }
         if (duality) {
            rel_duality_gap=duality_gap(loss,regularizer,x,lambda,best_dual,param.verbose);
            optim_info[1]=best_dual;
            optim_info[2]=rel_duality_gap;
         }
         optim_info[3]=it;
      };

   template <typename T>
      T LagrangianADMM(const SplittingFunction<T, Matrix<T> >& loss, const SplittingFunction<T, SpMatrix<T> >& reg, 
            const T lambda, const T gamma, const Vector<T>& w,  const Matrix<T>& splitted_loss, const SpMatrix<T>& splitted_reg,
            const Matrix<T>& multi_loss, const SpMatrix<T>& multi_reg, T& los, const T* weights = NULL) {
         const int n_reg=reg.num_components();
         //T loss_val = loss.eval(w) + lambda*reg.eval(w);
         T lagrangian = loss.eval_split(splitted_loss) + lambda*reg.eval_split(splitted_reg);
         Matrix<T> tmp;
         tmp.copy(splitted_loss);
         tmp.addVecToCols(w,-T(1.0));
         T add =0.5*gamma*tmp.normFsq();
         lagrangian += add;
         los+=add;
         if (n_reg > 0) {
            SpMatrix<T> stmp;
            stmp.copy(splitted_reg);
            stmp.addVecToCols(w,-T(1.0));
            add=0.5*gamma*stmp.normFsq();
            lagrangian += add;
            los+=add;
            lagrangian -= multi_reg.dot_direct(stmp);
         }
         lagrangian -= multi_loss.dot(tmp);
         return lagrangian;
      };


   template <typename T>
      void update_multipliers_ADMM(Vector<T>& w,
            const Matrix<T>& splitted_w_loss,
            const Matrix<T>& multipliers_w_loss,
            const SpMatrix<T>& splitted_w_reg,
            const SpMatrix<T>& multipliers_w_reg,
            const T gamma) {
         Vector<T> mean(w.n());
         splitted_w_loss.sum_cols(mean);
         w.copy(mean);
         multipliers_w_loss.sum_cols(mean);
         w.add(mean,-T(1.0)/gamma);   
         Vector<T> number_occurences(w.n());
         number_occurences.set(splitted_w_loss.n());
         const int n_reg=splitted_w_reg.n();
         if (n_reg > 0) {
            SpVector<T> col;
            mean.setZeros();
            for (int i = 0; i<n_reg; ++i) {
               splitted_w_reg.refCol(i,col);
               mean.add(col);
               for (int j = 0; j<col.L(); ++j) 
                  number_occurences[col.r(j)]++;
            }
            w.add(mean);   
            mean.setZeros();
            for (int i = 0; i<n_reg; ++i) {
               multipliers_w_reg.refCol(i,col);
               mean.add(col);
            }
            w.add(mean,-T(1.0)/gamma);
         };
         w.div(number_occurences);
      };


   template <typename T>
      void update_multipliers_weighted_ADMM(Vector<T>& w,
            const Matrix<T>& splitted_w_loss,
            const Matrix<T>& multipliers_w_loss,
            const SpMatrix<T>& splitted_w_reg,
            const SpMatrix<T>& multipliers_w_reg,
            const T gamma, const T* inner_weights) {
         Vector<T> mean(w.n());
         splitted_w_loss.sum_cols(mean);
         w.copy(mean);
         multipliers_w_loss.sum_cols(mean);
         w.add(mean,-T(1.0)/gamma);   
         Vector<T> number_occurences(w.n());
         number_occurences.set(splitted_w_loss.n());
         const int n_reg=splitted_w_reg.n();
         if (n_reg > 0) {
            SpVector<T> col;
            mean.setZeros();
            for (int i = 0; i<n_reg; ++i) {
               splitted_w_reg.refCol(i,col);
               for (int j = 0; j<col.L(); ++j) {
                  mean[col.r(j)]+=inner_weights[j]*col.v(j);
                  number_occurences[col.r(j)]+=inner_weights[j]*inner_weights[j];
               }
            }
            w.add(mean);   
            mean.setZeros();
            for (int i = 0; i<n_reg; ++i) {
               multipliers_w_reg.refCol(i,col);
               for (int j = 0; j<col.L(); ++j) 
                  mean[col.r(j)]+=inner_weights[j]*col.v(j);
            }
            w.add(mean,-T(1.0)/gamma);
         };
         w.div(number_occurences);
      };

   template <typename T>
      void ADMM(const SplittingFunction<T, Matrix<T> >& loss, const SplittingFunction<T, SpMatrix<T> >& reg, 
            const Vector<T>& w0, Vector<T>& w, Vector<T>& optim_info,
            const ParamFISTA<T>& param) {
         const T gamma = param.c; 
         const int n_reg=reg.num_components();
         const int it0 = MAX(1,param.it0);
         const T lambda=param.lambda;

         w.copy(w0);
         Matrix<T> splitted_w_loss;
         SpMatrix<T> splitted_w_reg;
         Matrix<T> multipliers_w_loss;
         SpMatrix<T> multipliers_w_reg;
         loss.init_split_variables(multipliers_w_loss);
         reg.init_split_variables(multipliers_w_reg);
         splitted_w_loss.copy(multipliers_w_loss);
         splitted_w_loss.addVecToCols(w);
         if (n_reg > 0) {
            splitted_w_reg.copy(multipliers_w_reg);
            splitted_w_reg.addVecToCols(w);
         }

         Timer time;
         time.start();
         int it=0;
         T los;
         T old_los=INFINITY;

         for (it = 0; it<param.max_it; ++it) {

            if (((it % it0) == 0)) {
               time.stop();
               if (param.is_inner_weights) {
                  los= loss.eval(w)+lambda*reg.eval_weighted(w,splitted_w_reg,
                        param.inner_weights);
               } else {
                  los= loss.eval(w)+lambda*reg.eval(w);
               }
               optim_info[0]=los;
               T sec=time.getElapsed();
               optim_info[2]=sec;
               if (param.verbose) {
                  cout << "Iter: " << it << ", loss: " << los << ", time: " << sec << endl;
                  flush(cout);
                  if (param.log) 
                     writeLog(it,sec,los,T(0),param.logName);
               }
               time.start();
            }
            if (param.is_inner_weights) {
               /// update w
               update_multipliers_weighted_ADMM(w,splitted_w_loss,multipliers_w_loss,splitted_w_reg,multipliers_w_reg,gamma,param.inner_weights);

               /// update the splitting variables
               splitted_w_loss.copy(multipliers_w_loss);
               splitted_w_loss.scal((1.0)/gamma);
               splitted_w_loss.addVecToCols(w);
               loss.prox_split(splitted_w_loss,T(1.0)/gamma);
               if (n_reg > 0) {
                  splitted_w_reg.copy(multipliers_w_reg);
                  splitted_w_reg.scal((1.0)/gamma);
                  splitted_w_reg.addVecToColsWeighted(w,param.inner_weights);
                  reg.prox_split(splitted_w_reg,lambda/gamma);
               }

               /// update  multipliers
               multipliers_w_loss.addVecToCols(w,gamma);
               multipliers_w_loss.add(splitted_w_loss,-gamma);
               if (n_reg > 0) {
                  multipliers_w_reg.addVecToColsWeighted(w,param.inner_weights,
                        gamma);
                  multipliers_w_reg.add_direct(splitted_w_reg,-gamma);
               }
            } else {
               /// update w
               update_multipliers_ADMM(w,splitted_w_loss,multipliers_w_loss,splitted_w_reg,multipliers_w_reg,gamma);

               /// update the splitting variables
               splitted_w_loss.copy(multipliers_w_loss);
               splitted_w_loss.scal((1.0)/gamma);
               splitted_w_loss.addVecToCols(w);
               loss.prox_split(splitted_w_loss,T(1.0)/gamma);
               if (n_reg > 0) {
                  splitted_w_reg.copy(multipliers_w_reg);
                  splitted_w_reg.scal((1.0)/gamma);
                  splitted_w_reg.addVecToCols(w);
                  reg.prox_split(splitted_w_reg,lambda/gamma);
               }

               /// update  multipliers
               multipliers_w_loss.addVecToCols(w,gamma);
               multipliers_w_loss.add(splitted_w_loss,-gamma);
               if (n_reg > 0) {
                  multipliers_w_reg.addVecToCols(w,gamma);
                  multipliers_w_reg.add_direct(splitted_w_reg,-gamma);
               }
            }

            /// stopping criterion
            if ((it % it0) == 0) {
               if (it > 0 && (old_los-los)/old_los < param.tol) break;
               old_los=los;
            }
         }
         if (param.is_inner_weights) {
            los= loss.eval(w)+lambda*reg.eval_weighted(w,splitted_w_reg,
                  param.inner_weights);
         } else {
            los= loss.eval(w)+lambda*reg.eval(w);
         }
         optim_info[0]=los;
         optim_info[3]=it;
      };

   template <typename T>
      void update_multipliers_LinADMM(Vector<T>& w,
            const SpMatrix<T>& splitted_w_reg,
            const SpMatrix<T>& multipliers_w_reg,
            const T gamma, const T delta) {
         Vector<T> mean(w.n());
         Vector<T> number_occurences(w.n());
         number_occurences.set(delta);
         const int n_reg=splitted_w_reg.n();
         if (n_reg > 0) {
            SpVector<T> col;
            mean.setZeros();
            for (int i = 0; i<n_reg; ++i) {
               splitted_w_reg.refCol(i,col);
               mean.add(col);
               for (int j = 0; j<col.L(); ++j) 
                  number_occurences[col.r(j)]+=gamma;
            }
            mean.scal(gamma);
            for (int i = 0; i<n_reg; ++i) {
               multipliers_w_reg.refCol(i,col);
               mean.add(col);
            }
            w.add(mean);
         };
         w.div(number_occurences);
      };


   template <typename T>
      void LinADMM(const SplittingFunction<T, Matrix<T> >& loss, const SplittingFunction<T, SpMatrix<T> >& reg, 
            const Vector<T>& w0, Vector<T>& w, Vector<T>& optim_info,
            const ParamFISTA<T>& param) {
         const T gamma = param.c; 
         const int n_reg=reg.num_components();
         const int it0 = MAX(1,param.it0);
         const T lambda=param.lambda;

         w.copy(w0);
         SpMatrix<T> primal_reg;
         SpMatrix<T> dual_reg;
         reg.init_split_variables(dual_reg);
         if (n_reg > 0) {
            primal_reg.copy(dual_reg);
            primal_reg.addVecToCols(w);
         }
         Vector<T> prim_loss;
         loss.init_prim_var(prim_loss);
         Vector<T> dual_loss;
         dual_loss.copy(prim_loss);

         Timer time;
         time.start();
         int it=0;
         T los;
         T old_los=INFINITY;

         for (it = 0; it<param.max_it; ++it) {
            /*w.print("w");
              prim_loss.print("z");
              dual_loss.print("nu");
              primal_reg.print("zg");
              dual_reg.print("nug");*/

            if (((it % it0) == 0)) {
               time.stop();
               los= loss.eval(w)+lambda*reg.eval(w);
               optim_info[0]=los;
               T sec=time.getElapsed();
               optim_info[2]=sec;
               if (param.verbose) {
                  cout << "Iter: " << it << ", loss: " << los << ", time: " << sec << endl;
                  flush(cout);
                  if (param.log) 
                     writeLog(it,sec,los,T(0),param.logName);
               }
               time.start();
            }
            /// update primal_loss variables
            loss.prox_prim_var(prim_loss,dual_loss,w,gamma);

            /// update primal_reg variables
            if (n_reg > 0) {
               primal_reg.copy(dual_reg);
               primal_reg.scal(-(1.0)/gamma);
               primal_reg.addVecToCols(w);
               reg.prox_split(primal_reg,lambda/gamma);
            }

            /// update w
            loss.compute_new_prim(w,prim_loss,dual_loss,gamma,param.delta);
            update_multipliers_LinADMM(w,primal_reg,dual_reg,gamma,param.delta);

            /// update  multipliers
            if (n_reg > 0) {
               dual_reg.addVecToCols(w,-gamma);
               dual_reg.add_direct(primal_reg,gamma);
            }
            loss.add_mult_design_matrix(w,dual_loss,-gamma);
            dual_loss.add(prim_loss,gamma);

            /// stopping criterion
            if ((it % it0) == 0) {
               if (it > 0 && (old_los-los)/old_los < param.tol) break;
               old_los=los;
            }
         }
         los= loss.eval(w)+lambda*reg.eval(w);
         optim_info[0]=los;
         optim_info[3]=it;
      };

   template <typename T>
      SplittingFunction<T, SpMatrix<T> >* setRegularizerADMM(const ParamFISTA<T>& param,
            const GraphStruct<T>* graph_st = NULL,
            const TreeStruct<T>* tree_st = NULL) {
         SplittingFunction<T, SpMatrix<T> >* reg;
         ParamReg<T> param_reg;
         param_reg.pos=param.pos;
         param_reg.intercept=param.intercept;
         param_reg.tree_st=const_cast<TreeStruct<T>* >(tree_st);
         param_reg.graph_st=const_cast<GraphStruct<T>* >(graph_st);
         param_reg.resetflow=param.resetflow;
         param_reg.clever=param.clever;
         switch (param.regul) {
            case GRAPH: param_reg.linf=true; reg=new GraphLasso<T>(param_reg); break;
            case GRAPH_L2: param_reg.linf=false; reg=new GraphLasso<T>(param_reg); break;
            case NONE: reg=new None<T>(); break;
            default: cerr << "Not implemented"; exit(1);
         }
         return reg;
      };

   template <typename T>
      Regularizer<T>* setRegularizerVectors(const ParamFISTA<T>& param,
            const GraphStruct<T>* graph_st = NULL,
            const TreeStruct<T>* tree_st = NULL,
            const GraphPathStruct<T>* graph_path_st=NULL) {
         ParamReg<T> param_reg;
         param_reg.pos=param.pos;
         param_reg.intercept=param.intercept;
         param_reg.lambda=param.lambda;
         param_reg.lambda2d1=param.lambda2/param.lambda;
         param_reg.lambda3d1=param.lambda3/param.lambda;
         param_reg.size_group=param.size_group;
         param_reg.tree_st=const_cast<TreeStruct<T>* >(tree_st);
         param_reg.graph_st=const_cast<GraphStruct<T>* >(graph_st);
         param_reg.graph_path_st=const_cast<GraphPathStruct<T>* >(graph_path_st);
         param_reg.resetflow=param.resetflow;
         param_reg.clever=param.clever;
         param_reg.ngroups=param.ngroups;
         param_reg.groups=param.groups;
         Regularizer<T>* reg;
         switch (param.regul) {
            case L0: reg=new Lzero<T>(param_reg); break;
            case L1: reg=new Lasso<T>(param_reg); break;
            case L1CONSTRAINT: reg=new LassoConstraint<T>(param_reg); break;
            case L2: reg=new normL2<T>(param_reg); break;
            case LINF: reg=new normLINF<T>(param_reg); break;
            case RIDGE: reg=new Ridge<T>(param_reg); break;
            case ELASTICNET: reg=new typename ElasticNet<T>::type(param_reg); break;
            case FUSEDLASSO: reg=new FusedLasso<T>(param_reg); break;
            case TREE_L0: reg=new TreeLzero<T>(param_reg); break;
            case TREE_L2: param_reg.linf=false; reg=new TreeLasso<T>(param_reg); break;
            case TREE_LINF: param_reg.linf=true; reg=new TreeLasso<T>(param_reg); break;
            case GRAPH: param_reg.linf=true; reg=new GraphLasso<T>(param_reg); break;
            case GRAPH_RIDGE: param_reg.linf=true; reg=new typename GraphLassoRidge<T>::type(param_reg); break;
            case GRAPH_L2: param_reg.linf=false; reg=new GraphLasso<T>(param_reg); break;
            case TRACE_NORM_VEC: reg=new ProxMatToVec<T, TraceNorm<T> >(param_reg); break;
            case RANK_VEC: reg=new ProxMatToVec<T, Rank<T> >(param_reg); break;
            case GROUPLASSO_L2: reg=new typename GroupLassoL2<T>::type(param_reg); break;
            case GROUPLASSO_LINF: reg=new typename GroupLassoLINF<T>::type(param_reg); break;
            case GROUPLASSO_L2_L1: reg=new typename GroupLassoL2_L1<T>::type(param_reg); break;
            case GROUPLASSO_LINF_L1: reg=new typename GroupLassoLINF_L1<T>::type(param_reg); break;
            case GRAPH_PATH_L0: reg = new GraphPathL0<T>(param_reg); break;
            case GRAPH_PATH_CONV: reg = new GraphPathConv<T>(param_reg); break;
            case NONE: reg=new None<T>(); break;
            default: cerr << "Not implemented"; exit(1);
         }
         return reg;
      };

   template <typename T>
      Regularizer<T, Matrix<T> >* setRegularizerMatrices(const ParamFISTA<T>& param,
            const int m, const int n,
            const GraphStruct<T>* graph_st = NULL,
            const TreeStruct<T>* tree_st = NULL,
            const GraphPathStruct<T>* graph_path_st=NULL) {
         ParamReg<T> param_reg;
         param_reg.transpose=param.transpose;
         param_reg.pos=param.pos;
         param_reg.intercept=param.intercept;
         param_reg.lambda2d1=param.lambda2/param.lambda;
         param_reg.lambda3d1=param.lambda3/param.lambda;
         param_reg.size_group=param.size_group;
         param_reg.num_cols=param.transpose ? m : n;
         param_reg.tree_st=const_cast<TreeStruct<T>* >(tree_st);
         param_reg.graph_st=const_cast<GraphStruct<T>* >(graph_st);
         param_reg.resetflow=param.resetflow;
         param_reg.clever=param.clever;
         Regularizer<T, Matrix<T> >* reg;
         switch (param.regul) {
            case L0: reg=new RegMat<T, Lzero<T> >(param_reg); break;
            case L1: reg=new RegMat<T, Lasso<T> >(param_reg); break;
            case L1CONSTRAINT: reg=new RegMat<T, LassoConstraint<T> >(param_reg); break;
            case L2: reg=new RegMat<T, normL2<T> >(param_reg); break;
            case LINF: reg=new RegMat<T, normLINF<T> >(param_reg); break;
            case RIDGE: reg=new RegMat<T, Ridge<T> >(param_reg); break;
            case ELASTICNET: reg=new RegMat<T, typename ElasticNet<T>::type >(param_reg); break;
            case FUSEDLASSO: reg=new RegMat<T, FusedLasso<T> >(param_reg); break;
            case L1L2: reg=new MixedL1L2<T>(param_reg); break;
            case L1LINF: reg=new MixedL1LINF<T>(param_reg); break;
            case TRACE_NORM: reg=new TraceNorm<T>(param_reg); break;
            case RANK: reg=new Rank<T>(param_reg); break;
            case L1L2_L1: reg=new typename MixedL1L2_L1<T>::type(param_reg); break;
            case L1LINF_L1: reg=new typename MixedL1LINF_L1<T>::type(param_reg); break;
            case TREE_L0: reg=new RegMat<T, TreeLzero<T> >(param_reg); break;
            case TREE_L2: param_reg.linf=false; reg=new RegMat<T, TreeLasso<T> >(param_reg); break;
            case TREE_LINF:  param_reg.linf=true; reg=new RegMat<T, TreeLasso<T> >(param_reg); break;
            case GRAPH: reg=new RegMat<T, GraphLasso<T> >(param_reg); break;
            case TREEMULT: reg = new TreeMult<T>(param_reg); break;
            case GRAPHMULT: reg=new GraphMult<T>(param_reg); break;
            case L1LINFCR: reg = new MixedL1LINFCR<T>(m,param_reg); break;
            case GRAPH_PATH_L0: reg = new RegMat<T, GraphPathL0<T> >(param_reg); break;
            case GRAPH_PATH_CONV: reg = new RegMat<T, GraphPathConv<T> >(param_reg); break;
            case NONE: reg=new RegMat<T, None<T> >(param_reg); break;
            default: cerr << "not implemented"; exit(1);
         }
         return reg;
      }

   template <typename T>
      void print_info_solver(const ParamFISTA<T>& param) {
         if (param.verbose) {
            print_loss(param.loss);
            print_regul(param.regul);
            if (param_for_admm(param)) {
               if (param.admm || param.lin_admm) {
                  if (param.lin_admm) {
                     cout << "Linearized ADMM algorithm" << endl;
                  } else {
                     cout << "ADMM algorithm" << endl;
                  }
               } 
            } else {
               if (param.ista) {
                  cout << "ISTA algorithm" << endl;
               } else if (param.subgrad) {
                  cout << "Subgradient descent" << endl;
               } else {
                  cout << "FISTA algorithm" << endl;
               }
               if ((param.regul == GRAPH || param.regul == TREEMULT ||
                        param.regul == GRAPHMULT || param.regul==L1LINFCR) &&
                     param.clever) 
                  cout << "Projections with arc capacities" << endl;
               if (param.intercept) cout << "with intercept" << endl;
               if (param.log && param.logName) {
                  cout << "log activated " << endl;
                  cout << param.logName << endl;
                  cout << endl;
               }
            }
            flush(cout);
         }
      };


   template <typename T>
      void solver_admm(const Matrix<T>& X, const Matrix<T>& alpha0,
            Matrix<T>& alpha, Matrix<T>& optim_info, SplittingFunction<T, SpMatrix<T> >** regularizers,
            SplittingFunction<T, Matrix<T> >** losses, const ParamFISTA<T>& param) {
         const int M = X.n();
         optim_info.resize(4,M);
         
         int i1;
#pragma omp parallel for private(i1) 
         for (i1 = 0; i1< M; ++i1) {
#ifdef _OPENMP
            int numT=omp_get_thread_num();
#else
            int numT=0;
#endif
            Vector<T> Xi;
            X.refCol(i1,Xi);
            losses[numT]->init(Xi);
            Vector<T> alpha0i;
            alpha0.refCol(i1,alpha0i);
            Vector<T> alphai;
            alpha.refCol(i1,alphai);
            regularizers[numT]->reset();
            Vector<T> optim_infoi;
            optim_info.refCol(i1,optim_infoi);
            if (param.admm || param.lin_admm) {
               if (param.lin_admm) {
                  LinADMM(*(losses[numT]),*(regularizers[numT]),alpha0i,alphai,optim_infoi,param);
               } else {
                  ADMM(*(losses[numT]),*(regularizers[numT]),alpha0i,alphai,optim_infoi,param);
               }
            } 
         }
      }


   template <typename T>
      void solver_aux1(const Matrix<T>& X, const Matrix<T>& alpha0,
            Matrix<T>& alpha, Matrix<T>& optim_info, Regularizer<T, Vector<T> >** regularizers,
            Loss<T, Vector<T> >** losses, const ParamFISTA<T>& param) {
         const int M = X.n();
         if (param.verbose) {
            const bool duality = losses[0]->is_fenchel() && regularizers[0]->is_fenchel();
            if (duality) cout << "Duality gap via Fenchel duality" << endl;
            if (!param.ista && param.subgrad && !regularizers[0]->is_subgrad()) {
               cerr << "Subgradient algorithm is not implemented for this combination of loss/regularization" << endl;
               exit(1);
            }
            cout << "Timings reported do not include loss and fenchel evaluation" << endl;
            flush(cout);
         }
         optim_info.resize(4,M);

         int i1;
#pragma omp parallel for private(i1) 
         for (i1 = 0; i1< M; ++i1) {
#ifdef _OPENMP
            int numT=omp_get_thread_num();
#else
            int numT=0;
#endif
            Vector<T> Xi;
            X.refCol(i1,Xi);
            losses[numT]->init(Xi);
            Vector<T> alpha0i;
            alpha0.refCol(i1,alpha0i);
            Vector<T> alphai;
            alpha.refCol(i1,alphai);
            regularizers[numT]->reset();
            Vector<T> optim_infoi;
            optim_info.refCol(i1,optim_infoi);
            if (param.ista) {
               ISTA_Generic(*(losses[numT]),*(regularizers[numT]),alpha0i,alphai,optim_infoi,param);
            } else if (param.subgrad) {
               subGradientDescent_Generic(*(losses[numT]),*(regularizers[numT]),alpha0i,alphai,optim_infoi,param);
            } else {
               FISTA_Generic(*(losses[numT]),*(regularizers[numT]),alpha0i,alphai,optim_infoi,param);
            }
         }
      }

   template <typename T>
      void solver_aux2(const Matrix<T>& X, const Matrix<T>& alpha0,
            Matrix<T>& alpha, Matrix<T>& optim_info, Regularizer<T, Matrix<T> >** regularizers,
            Loss<T, Matrix<T> >** losses, const ParamFISTA<T>& param) {
         const int M = X.n();
         if (param.verbose) {
            const bool duality = losses[0]->is_fenchel() && regularizers[0]->is_fenchel();
            if (duality) cout << "Duality gap via Fenchel duality" << endl;
            flush(cout);
         }

         optim_info.resize(4,M);

         int i2;
#pragma omp parallel for private(i2) 
         for (i2 = 0; i2< M; ++i2) {
#ifdef _OPENMP
            int numT=omp_get_thread_num();
#else
            int numT=0;
#endif
            Vector<T> Xi;
            X.refCol(i2,Xi);
            losses[numT]->init(Xi);
            const int N = alpha0.n()/X.n();
            Matrix<T> alpha0i;
            alpha0.refSubMat(i2*N,N,alpha0i);
            Matrix<T> alphai;
            alpha.refSubMat(i2*N,N,alphai);
            regularizers[numT]->reset();
            Vector<T> optim_infoi;
            optim_info.refCol(i2,optim_infoi);
            if (param.ista) {
               ISTA_Generic(*(losses[numT]),*(regularizers[numT]),alpha0i,alphai,optim_infoi,param);
            } else if (param.subgrad) {
               subGradientDescent_Generic(*(losses[numT]),*(regularizers[numT]),alpha0i,alphai,optim_infoi,param);
            } else {
               FISTA_Generic(*(losses[numT]),*(regularizers[numT]),alpha0i,alphai,optim_infoi,param);
            }
         }
      }

   /// AbstractMatrixB is basically either SpMatrix or Matrix
   template <typename T>
      void solver(const Matrix<T>& X, const AbstractMatrixB<T>& D, const Matrix<T>& alpha0,
            Matrix<T>& alpha, const ParamFISTA<T>& param1, Matrix<T>& optim_info,
            const GraphStruct<T>* graph_st = NULL, 
            const TreeStruct<T>* tree_st = NULL,
            const GraphPathStruct<T>* graph_path_st=NULL) {
         print_info_solver(param1);

         int num_threads=MIN(X.n(),param1.num_threads);
         num_threads=init_omp(num_threads);
         ParamFISTA<T> param=param1;
         param.copied=true;
         if (param_for_admm(param)) {
            if (num_threads > 1) param.verbose=false;
            SplittingFunction<T>** losses = new SplittingFunction<T>*[num_threads];
            SplittingFunction<T, SpMatrix<T> >** regularizers= new SplittingFunction<T, SpMatrix<T> >*[num_threads];
            for (int i = 0; i<num_threads; ++i) {
               regularizers[i]=setRegularizerADMM(param,graph_st,tree_st);
               switch (param.loss) {
                  case SQUARE: losses[i]=new SqLoss<T>(D); break;
                  case HINGE: losses[i] = new HingeLoss<T>(D); break;
                  default: cerr << "Not implemented" << endl; exit(1);
               }
            }
            solver_admm(X, alpha0,  alpha, optim_info, regularizers,losses,param);
            for (int i = 0; i<num_threads; ++i) {
               delete(losses[i]);
               delete(regularizers[i]);
            }
            delete[](losses);
            delete[](regularizers);

         } else {
            Matrix<T> G;
            if (param.loss==HINGE) {
               cerr << "Loss only implemented for ADMM" << endl;
               return;
            }
            if (param.compute_gram && (param.loss==SQUARE)) D.XtX(G);
            if (!loss_for_matrices(param.loss) && !(param.transpose || regul_for_matrices(param.regul))) {
               if (num_threads > 1) param.verbose=false;
               Loss<T>** losses = new Loss<T>*[num_threads];
               Regularizer<T>** regularizers= new Regularizer<T>*[num_threads];
               for (int i = 0; i<num_threads; ++i) {
                  regularizers[i]=setRegularizerVectors(param,graph_st,tree_st,graph_path_st);
                  switch (param.loss) {
                     case SQUARE: if (param.compute_gram) {
                                     losses[i]=new SqLoss<T>(D,G); 
                                  } else {
                                     losses[i]=new SqLoss<T>(D); 
                                  }
                                  break;
                     case SQUARE_MISSING: losses[i]=new SqLossMissing<T>(D);  break;
                     case LOG:  losses[i] = new LogLoss<T>(D); break;
                     case LOGWEIGHT:  losses[i] = new LogLoss<T,true>(D); break;
                     default: cerr << "Not implemented"; exit(1);
                  }
               }

               solver_aux1(X, alpha0,  alpha, optim_info, regularizers,losses,param);
               for (int i = 0; i<num_threads; ++i) {
                  delete(losses[i]);
                  losses[i]=NULL;
                  delete(regularizers[i]);
                  regularizers[i]=NULL;
               }
               delete[](losses);
               delete[](regularizers);

            } else if (loss_for_matrices(param.loss) && param.loss != CUR) {
               if (num_threads > 1) param.verbose=false;
               Loss<T, Matrix<T> >** losses = new Loss<T, Matrix<T> >*[num_threads];
               Regularizer<T, Matrix<T> >** regularizers= new Regularizer<T, Matrix<T> >*[num_threads];
               const int N = alpha0.n()/X.n();
               for (int i = 0; i<num_threads; ++i) {
                  regularizers[i]=setRegularizerMatrices(param,alpha0.m(),N,graph_st,tree_st,graph_path_st);
                  switch (param.loss) {
                     case MULTILOG:  losses[i] = new MultiLogLoss<T>(D); break;
                     default: cerr << "Not implemented"; exit(1);
                  }
               }
               solver_aux2(X, alpha0,  alpha, optim_info, regularizers,losses,param);

               for (int i = 0; i<num_threads; ++i) {
                  delete(losses[i]);
                  losses[i]=NULL;
                  delete(regularizers[i]);
                  regularizers[i]=NULL;
               }

               delete[](losses);
               delete[](regularizers);
            } else {
               /// (loss not for matrices and regul for matrices) or CUR
               Loss<T, Matrix<T>, Matrix<T> >* loss;
               Regularizer<T, Matrix<T> >* regularizer;
               switch (param.loss) {
                  case SQUARE: if (param.compute_gram) {
                                  loss=new SqLossMat<T>(D,G); 
                               } else {
                                  loss=new SqLossMat<T>(D); 
                               }
                               break;
                  case SQUARE_MISSING: loss=new LossMat<T, SqLossMissing<T> >(X.n(),D);  break;
                  case LOG:  loss = new LossMat<T, LogLoss<T,false> >(X.n(),D); break;
                  case LOGWEIGHT:  loss = new LossMat<T, LogLoss<T,true> >(X.n(),D); break;
                  case CUR:  loss = new LossCur<T>(D); break; 
                  default: cerr << "Not implemented"; exit(1);
               }
               regularizer=setRegularizerMatrices(param,alpha0.m(),alpha0.n(),graph_st,tree_st,graph_path_st);
               if (param.verbose) {
                  const bool duality = loss->is_fenchel() && regularizer->is_fenchel();
                  if (duality) cout << "Duality gap via Fenchel duality" << endl;
               }
               loss->init(X);
               optim_info.resize(4,1);
               Vector<T> optim_infoi;
               optim_info.refCol(0,optim_infoi);
               if (param.ista) {
                  ISTA_Generic(*loss,*regularizer,alpha0,alpha,optim_infoi,param);
               } else if (param.subgrad) {
                  subGradientDescent_Generic(*loss,*regularizer,alpha0,alpha,optim_infoi,param);
               } else {
                  FISTA_Generic(*loss,*regularizer,alpha0,alpha,optim_infoi,param);
               }
               delete(regularizer);
               delete(loss);
            }
         }
      };

   template <typename T>
      void PROX(const Matrix<T>& alpha0,
            Matrix<T>& alpha, const ParamFISTA<T>& param, 
            Vector<T>& val_loss,
            const GraphStruct<T>* graph_st = NULL, 
            const TreeStruct<T>* tree_st = NULL,
            const GraphPathStruct<T>* graph_path_st = NULL) {
         if (param.verbose) {
            print_regul(param.regul);
            if ((param.regul == GRAPH || param.regul == TREEMULT ||
                     param.regul == GRAPHMULT || param.regul==L1LINFCR) &&
                  param.clever) 
               cout << "Projections with arc capacities" << endl;
            if (param.intercept) cout << "with intercept" << endl;
            flush(cout);
         }
         int num_threads=MIN(alpha.n(),param.num_threads);
         num_threads=init_omp(num_threads);
         const int M = alpha.n();
         if (!graph_st && param.regul==GRAPH) {
            cerr << "Graph structure should be provided" << endl;
            return;
         }

         if (!regul_for_matrices(param.regul)) {
            Regularizer<T>** regularizers= new Regularizer<T>*[num_threads];
            for (int i = 0; i<num_threads; ++i) 
               regularizers[i]=setRegularizerVectors(param,graph_st,tree_st,graph_path_st);

            int i;
            if (param.eval)
               val_loss.resize(M);
#pragma omp parallel for private(i) 
            for (i = 0; i< M; ++i) {
#ifdef _OPENMP
               int numT=omp_get_thread_num();
#else
               int numT=0;
#endif
               Vector<T> alpha0i;
               alpha0.refCol(i,alpha0i);
               Vector<T> alphai;
               alpha.refCol(i,alphai);
               regularizers[numT]->reset();
               regularizers[numT]->prox(alpha0i,alphai,param.lambda);
               if (param.eval)
                  val_loss[i]=regularizers[numT]->eval(alphai);
            }
            for (i = 0; i<num_threads; ++i) {
               delete(regularizers[i]);
               regularizers[i]=NULL;
            }
            delete[](regularizers);

         } else {
            /// regul for matrices
            if (param.eval)
               val_loss.resize(1);
            Regularizer<T, Matrix<T> >* regularizer;
            regularizer=setRegularizerMatrices(param,alpha0.m(),alpha0.n(),graph_st,tree_st,graph_path_st);
            regularizer->prox(alpha0,alpha,param.lambda);
            if (param.eval)
               val_loss[0]=regularizer->eval(alpha);
            delete(regularizer);
         }
      };

   template <typename T>
      void EvalGraphPath(const Matrix<T>& alpha0,
            const ParamFISTA<T>& param, 
            Vector<T>& val_loss,
            const GraphPathStruct<T>* graph_path_st,
            SpMatrix<T>* paths = NULL) {
         if (param.verbose) {
            print_regul(param.regul);
            if (param.intercept) cout << "with intercept" << endl;
            if (param.eval_dual_norm) cout << "Evaluate the dual norm only" << endl;
            flush(cout);
         }
         int num_threads=MIN(alpha0.n(),param.num_threads);
         num_threads=init_omp(num_threads);
         const int M = alpha0.n();

         if (!regul_for_matrices(param.regul)) {
            Regularizer<T>** regularizers= new Regularizer<T>*[num_threads];
            for (int i = 0; i<num_threads; ++i) 
               regularizers[i]=setRegularizerVectors<T>(param,NULL,NULL,graph_path_st);

            int i;
            val_loss.resize(M);
#pragma omp parallel for private(i) 
            for (i = 0; i< M; ++i) {
#ifdef _OPENMP
               int numT=omp_get_thread_num();
#else
               int numT=0;
#endif
               Vector<T> alphai;
               alpha0.refCol(i,alphai);
               regularizers[numT]->reset();
               if (i==0 && paths) {
                  if (param.eval_dual_norm) {
                     val_loss[i]=regularizers[numT]->eval_dual_norm_paths(alphai,*paths);
                  } else {
                     val_loss[i]=regularizers[numT]->eval_paths(alphai,*paths);
                  }
               } else {
                  if (param.eval_dual_norm) {
                     val_loss[i]=regularizers[numT]->eval_dual_norm(alphai);
                  } else {
                     val_loss[i]=regularizers[numT]->eval(alphai);
                  }
               }
            }
            for (i = 0; i<num_threads; ++i) {
               delete(regularizers[i]);
               regularizers[i]=NULL;
            }
            delete[](regularizers);

         } else {
            cerr << "Not implemented" << endl;
            return;
         }
      };
}


#endif
