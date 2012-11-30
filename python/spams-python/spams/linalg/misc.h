/*!
 * \file
 *                toolbox Linalg 
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File misc.h
 * \brief Contains miscellaneous functions */


#ifndef MISC_H
#define MISC_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"

#ifdef WINDOWS
#define isnan _isnan
#define isinf !_finite
#endif

using namespace std;


/// a useful debugging function
static inline void stop();
/// seed for random number generation
static int seed = 0;
/// first random number generator from Numerical Recipe
template <typename T> static inline T ran1(); 
/// standard random number generator 
template <typename T> static inline T ran1b(); 
/// random sampling from the normal distribution
template <typename T> static inline T normalDistrib();
/// reorganize a sparse table between indices beg and end,
/// using quicksort
template <typename T>
static void sort(int* irOut, T* prOut,int beg, int end);
template <typename T>
static void quick_sort(int* irOut, T* prOut,const int beg, const int end, const bool incr);
/// template version of the power function
template <typename T>
T power(const T x, const T y);
/// template version of the fabs function
template <typename T>
T abs(const T x);
/// template version of the fabs function
template <typename T>
T sqr(const T x);
template <typename T>
T sqr_alt(const T x);
/// template version of the fabs function
template <typename T>
T sqr(const int x) {
   return sqr<T>(static_cast<T>(x));
}

template <typename T>
T exp_alt(const T x);
template <typename T>
T log_alt(const T x);

/// a useful debugging function
/*static inline void stop() {
   cout << "Appuyez sur entrée pour continuer...";
   cin.ignore( numeric_limits<streamsize>::max(), '\n' );
};*/
static inline void stop() {
   printf("Appuyez sur une touche pour continuer\n");
   getchar();
}

/// first random number generator from Numerical Recipe
template <typename T> static inline T ran1() {
   const int IA=16807,IM=2147483647,IQ=127773,IR=2836,NTAB=32;
   const int NDIV=(1+(IM-1)/NTAB);
   const T EPS=3.0e-16,AM=1.0/IM,RNMX=(1.0-EPS);
   static int iy=0;
   static int iv[NTAB];
   int j,k;
   T temp;

   if (seed <= 0 || !iy) {
      if (-seed < 1) seed=1;
      else seed = -seed;
      for (j=NTAB+7;j>=0;j--) {
         k=seed/IQ;
         seed=IA*(seed-k*IQ)-IR*k;
         if (seed < 0) seed += IM;
         if (j < NTAB) iv[j] = seed;
      }
      iy=iv[0];
   }
   k=seed/IQ;
   seed=IA*(seed-k*IQ)-IR*k;
   if (seed < 0) seed += IM;
   j=iy/NDIV;
   iy=iv[j];
   iv[j] = seed;
   if ((temp=AM*iy) > RNMX) return RNMX;
   else return temp;
};

/// standard random number generator 
template <typename T> T ran1b() {
   return static_cast<T>(rand())/RAND_MAX;
}

/// random sampling from the normal distribution
template <typename T>
static inline T normalDistrib() {
   static bool iset = true;
   static T gset;

   T fac,rsq,v1,v2;
   if (iset) {
      do {
         v1 = 2.0*ran1<T>()-1.0;
         v2 = 2.0*ran1<T>()-1.0;
         rsq = v1*v1+v2*v2;
      } while (rsq >= 1.0 || rsq == 0.0);
      fac = sqrt(-2.0*log(rsq)/rsq);
      gset = v1*fac;
      iset = false;
      return v2*fac;
   } else {
      iset = true;
      return gset;
   }
};

/// reorganize a sparse table between indices beg and end,
/// using quicksort
template <typename T>
static void sort(int* irOut, T* prOut,int beg, int end) {
   int i;
   if (end <= beg) return;
   int pivot=beg;
   for (i = beg+1; i<=end; ++i) {
      if (irOut[i] < irOut[pivot]) {
         if (i == pivot+1) {
            int tmp = irOut[i];
            T tmpd = prOut[i];
            irOut[i]=irOut[pivot];
            prOut[i]=prOut[pivot];
            irOut[pivot]=tmp;
            prOut[pivot]=tmpd;
         } else {
            int tmp = irOut[pivot+1];
            T tmpd = prOut[pivot+1];
            irOut[pivot+1]=irOut[pivot];
            prOut[pivot+1]=prOut[pivot];
            irOut[pivot]=irOut[i];
            prOut[pivot]=prOut[i];
            irOut[i]=tmp;
            prOut[i]=tmpd;
         }
         ++pivot;
      }
   }
   sort(irOut,prOut,beg,pivot-1);
   sort(irOut,prOut,pivot+1,end);
}
template <typename T>
static void quick_sort(int* irOut, T* prOut,const int beg, const int end, const bool incr) {
   if (end <= beg) return;
   int pivot=beg;
   if (incr) {
      const T val_pivot=prOut[pivot];
      const int key_pivot=irOut[pivot];
      for (int i = beg+1; i<=end; ++i) {
         if (prOut[i] < val_pivot) {
            prOut[pivot]=prOut[i];
            irOut[pivot]=irOut[i];
            prOut[i]=prOut[++pivot];
            irOut[i]=irOut[pivot];
            prOut[pivot]=val_pivot;
            irOut[pivot]=key_pivot;
         } 
      }
   } else {
      const T val_pivot=prOut[pivot];
      const int key_pivot=irOut[pivot];
      for (int i = beg+1; i<=end; ++i) {
         if (prOut[i] > val_pivot) {
            prOut[pivot]=prOut[i];
            irOut[pivot]=irOut[i];
            prOut[i]=prOut[++pivot];
            irOut[i]=irOut[pivot];
            prOut[pivot]=val_pivot;
            irOut[pivot]=key_pivot;
         } 
      }
   }
   quick_sort(irOut,prOut,beg,pivot-1,incr);
   quick_sort(irOut,prOut,pivot+1,end,incr);
}


/// template version of the power function
template <>
inline double power(const double x, const double y) {
   return pow(x,y);
};
template <>
inline float power(const float x, const float y) {
   return powf(x,y);
};

/// template version of the fabs function
template <>
inline double abs(const double x) {
   return fabs(x);
};
template <>
inline float abs(const float x) {
   return fabsf(x);
};

/// template version of the fabs function
template <>
inline double sqr(const double x) {
   return sqrt(x);
};
template <>
inline float sqr(const float x) {
   return sqrtf(x);
};

template <>
inline double exp_alt(const double x) {
   return exp(x);
};
template <>
inline float exp_alt(const float x) {
   return expf(x);
};

template <>
inline double log_alt(const double x) {
   return log(x);
};
template <>
inline float log_alt(const float x) {
   return logf(x);
};


template <>
inline double sqr_alt(const double x) {
   return sqrt(x);
};
template <>
inline float sqr_alt(const float x) {
   return sqrtf(x);
};

static inline int init_omp(const int numThreads) {
   int NUM_THREADS;
#ifdef _OPENMP
   NUM_THREADS = (numThreads == -1) ? MIN(MAX_THREADS,omp_get_num_procs()) : numThreads;
   omp_set_nested(0);
   omp_set_dynamic(0);
   omp_set_num_threads(NUM_THREADS);
#else
   NUM_THREADS = 1;
#endif
   return NUM_THREADS;
}


#endif
