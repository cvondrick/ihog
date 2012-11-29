/*!
 * \file
 *                toolbox Linalg 
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File utils.h
 * \brief Contains various variables and class timer */


#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifdef HAVE_MKL   // obsolete
//#include <mkl_cblas.h>
//#else
//#include "cblas.h"
#endif
#ifdef USE_BLAS_LIB
//#include "blas.h"
#else
#include "cblas.h"  // dependency upon cblas libraries has been removed in a recent version
#endif

#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MATLAB_MEX_FILE
typedef int mwSize;
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

// MIN, MAX macros
#define MIN(a,b) (((a) > (b)) ? (b) : (a))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define SIGN(a) (((a) < 0) ? -1.0 : 1.0)
#define ABS(a) (((a) < 0) ? -(a) : (a))
// DEBUG macros
#define PRINT_I(name) printf(#name " : %d\n",name);
#define PRINT_F(name) printf(#name " : %g\n",name);
#define PRINT_S(name) printf("%s\n",name);
#define FLAG(a) printf("flag : %d \n",a);

// ALGORITHM constants
#define EPSILON 10e-10
#ifndef INFINITY
#define INFINITY 10e20
#endif
#define EPSILON_OMEGA 0.001
#define TOL_CGRAD 10e-6
#define MAX_ITER_CGRAD 40


#ifdef _MSC_VER

#include <time.h>
#include <windows.h>
#define random rand
#define srandom srand

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif


struct timezone
{
   int  tz_minuteswest; /* minutes W of Greenwich */
   int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval *tv, struct timezone *tz)
{
   FILETIME ft;
   unsigned __int64 tmpres = 0;
   static int tzflag = 0;

   if (NULL != tv)
   {
      GetSystemTimeAsFileTime(&ft);

      tmpres |= ft.dwHighDateTime;
      tmpres <<= 32;
      tmpres |= ft.dwLowDateTime;

      tmpres /= 10;  /*convert into microseconds*/
      /*converting file time to unix epoch*/
      tmpres -= DELTA_EPOCH_IN_MICROSECS;
      tv->tv_sec = (long)(tmpres / 1000000UL);
      tv->tv_usec = (long)(tmpres % 1000000UL);
   }

   if (NULL != tz)
   {
      if (!tzflag)
      {
         _tzset();
         tzflag++;
      }
      tz->tz_minuteswest = _timezone / 60;
      tz->tz_dsttime = _daylight;
   }

   return 0;
}

#else
#include <sys/time.h>
#endif


#include "linalg.h"

using namespace std;

/// Class Timer 
class Timer {
   public:
      /// Empty constructor
      Timer();
      /// Destructor
      ~Timer();

      /// start the time
      void inline start() { _running=true;
         gettimeofday(_time1,NULL); };
         /// stop the time
         void inline stop() { 
            gettimeofday(_time2,NULL);
            _running=false;
            _cumul+=  static_cast<double>((_time2->tv_sec - (_time1->tv_sec))*1000000 + _time2->tv_usec-_time1->tv_usec)/1000000.0;
         };
         /// reset the timer
         void inline reset() { _cumul=0;  
            gettimeofday(_time1,NULL); };
            /// print the elapsed time
            void inline printElapsed();
            /// print the elapsed time
            double inline getElapsed() const;

   private:
            struct timeval* _time1;
            struct timeval* _time2;
            bool _running;
            double _cumul;
};

/// Constructor
Timer::Timer() :_running(false) ,_cumul(0) {
   _time1 = (struct timeval*)malloc(sizeof(struct timeval));
   _time2 = (struct timeval*)malloc(sizeof(struct timeval));
};

/// Destructor
Timer::~Timer() {
   free(_time1);
   free(_time2);
}

/// print the elapsed time
inline void Timer::printElapsed() {
   if (_running) {
      gettimeofday(_time2,NULL);
      cerr << "Time elapsed : " << _cumul + static_cast<double>((_time2->tv_sec -
               _time1->tv_sec)*1000000 + _time2->tv_usec-_time1->tv_usec)/1000000.0 << endl;
   } else {
      cerr << "Time elapsed : " << _cumul << endl;
   }
};

/// print the elapsed time
double inline Timer::getElapsed() const {
   if (_running) {
      gettimeofday(_time2,NULL);
      return _cumul + 
         static_cast<double>((_time2->tv_sec -
                  _time1->tv_sec)*1000000 + _time2->tv_usec-
               _time1->tv_usec)/1000000.0;
   } else {
      return _cumul;
   }
}


#endif
