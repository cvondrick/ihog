#include "cppspams.h"
#include <time.h>

static struct timespec tstart, tend;

float delta_t(struct timespec &t1,struct timespec &t2) {
  float sec = (float)(t2.tv_sec - t1.tv_sec);
  float ms = (float)(t2.tv_nsec - t1.tv_nsec) / 1000000.; 
  float t = (sec * 1000. + ms) / 1000.;
  return t;
}


void test_omp() {
  std::cout << "OMP" << std::endl;
  int m(64), n(100000), p(200);
   Matrix<double> X(m,n);
   X.setAleat();
   double* prD = new double[m*p];
   Matrix<double> D(prD,m,p); 
   D.setAleat(); 
   D.normalize();
   const int L = 10;
   double eps = 1.0;
   double lambda = 0.;
   SpMatrix<double> spa;
   clock_gettime(CLOCK_REALTIME,&tstart);
   cppOMP(X,D,spa,&L,&eps,&lambda);
   clock_gettime(CLOCK_REALTIME,&tend);
   float nbs = X.n() / delta_t(tstart,tend);
   std::cout << nbs << " signals processed per second." << std::endl;
   delete[](prD);

}

void test_lasso() {
  std::cout << "LASSO" << std::endl;
  int m = 100;
  int n = 100000;
  int p = 200;
   /// external allocation for the matrix
   double* prD = new double[m*p];
   Matrix<double> D(prD,m,p); 
   D.setAleat(); 
   D.normalize();

   /// Allocate a matrix of size m x p
   Matrix<double> D2(m,p); 
   D2.setAleat();
   D2.normalize();

   Matrix<double> X(m,n);
   X.setAleat();
   
   /// create empty sparse matrix
   Matrix<double> *path;

   clock_gettime(CLOCK_REALTIME,&tstart);
   SpMatrix<double> *spa = cppLasso(X,D,&path,false,10,0.15);  // first simple example
   clock_gettime(CLOCK_REALTIME,&tend);
   float nbs = X.n() / delta_t(tstart,tend);
   std::cout << nbs << " signals processed per second." << std::endl;
   delete[](prD);
   delete spa;
}

struct progs {
  const char *name;
  void (*prog)();
} progs[] = { "omp", test_omp,
	      "lasso", test_lasso,
};
int main(int argc, char** argv) {
  
  for(int i = 1;i < argc;i++) {
    bool found = false;
    for (int j = 0;j < (sizeof(progs) / sizeof(struct progs));j++) {
      if(strcmp(argv[i],progs[j].name) == 0) {
	found = true;
	(progs[j].prog)();
	break;
      }
    }
    if(! found) {
      std::cout << "!! " << argv[i] << " not found" << std::endl;
      return 1;
    }
  }
  return 0;
}
