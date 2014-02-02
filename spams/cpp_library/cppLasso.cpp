#include "linalg.h"

int main(int argc, char** argv) {
   int m = 10;
   int p = 20;
   /// external allocation for the matrix
   double* prD = new double[m*p];
   Matrix<double> D(prD,m,p); 
   D.setAleat(); 
   D.normalize();

   /// Allocate a matrix of size m x p
   Matrix<double> D2(m,p); 
   D2.setAleat();
   D2.normalize();

   int n = 100;
   Matrix<double> X(m,n);
   X.setAleat();
   
   /// create empty sparse matrix
   SpMatrix<double> spa;

   lasso2(X,D,spa,10,0.15);  // first simple example

   /// extern allocation for the matrix D requires
   /// manually unallocating prD
   delete[](prD);
}
