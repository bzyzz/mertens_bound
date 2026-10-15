
/*
This program was written by Phong Nguyen (2024).

It rounds a real matrix into an integer matrix so that lattice reduction software can be used

Compilation:
 g++  -std=c++11   -o RoundMatrix RoundMatrix.cpp   -lgmp    -lm -lntl
 */

#include "NTL/LLL.h"
#include "NTL/HNF.h"

#include <string.h>
#include <iomanip>
#include <fstream>
#include "NTL/fileio.h"
#include <NTL/mat_RR.h>
#include <NTL/vec_ZZ.h>

 using namespace NTL;
 using namespace std;


int main(int argc, char * argv[])
{

  long i,j,prec=5*NTL_ZZ_NBITS;
  double factor = 0.99;


  for (i=1;i<argc;i++) {
    if (!strcmp(argv[i],"-h")) {
      cerr << "Options disponibles: \n";
      cerr << "-h     : Aide.\n";
      exit(1);
    }
    else {
      cerr << "Option inconnue !!!\n" << argv[i] << "\n";
      exit(1);
    }
  }



  mat_ZZ M;
  mat_RR F;


 cin >> F;
  long n,m;
  n=F.NumRows();
  m=F.NumCols();

  M.SetDims(n,m);
  for (i=1;i<=n;i++)
    for (j=1;j<=m;j++)
      RoundToZZ(M(i,j),F(i,j));

  cout << M << endl;

}
