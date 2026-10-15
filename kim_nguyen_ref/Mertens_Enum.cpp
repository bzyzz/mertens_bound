#include "NTL/LLL.h"
#include <NTL/mat_RR.h>
#include <string.h>
#include <iomanip>
#include <fstream>

 using namespace NTL;
 using namespace std;


/* This program does lattice enumeration tailored for the Mertens lattice,
 using the NTL library

It is based on the algorithm
 described in:

 [LN13]
 Mingjie Liu, Phong Q. Nguyen:
Solving BDD by Enumeration: An Update. CT-RSA 2013: 293-309


 which is the BDD adaptation of the algorithm described in
 the full version of:

 @inproceedings{DBLP:conf/eurocrypt/GamaNR10,
 author    = {Nicolas Gama and
 Phong Q. Nguyen and
 Oded Regev},
 title     = {Lattice Enumeration Using Extreme Pruning},
 booktitle = {EUROCRYPT},
 year      = {2010},
 pages     = {257-278},
 ee        = {http://dx.doi.org/10.1007/978-3-642-13190-5_13},
 crossref  = {DBLP:conf/eurocrypt/2010},
 bibsource = {DBLP, http://dblp.uni-trier.de}
 }

 This program was written by Phong Nguyen (2024),
 with some parts being adapted from older code by Nicolas Gama (2010)

Example:
./Mertens_Enum -prec 300 -gh 1.19 -lin 3 < SR140-130-100-30.B88-ENUM
./Mertens_Enum -prec 300 -gh 1.19 -lin 3 < SR140-130-100-30.B88-ENUM2

 */


vec_RR loggamma;
vec_RR heurist;  // log du rayon de l'heuristique gaussienne
vec_RR rayon;    // carre du rayon de l'heuristique gaussienne
vec_RR unitball; // volume de la boule unite




void ModifiedGS(const mat_ZZ& B, mat_RR& mu, vec_RR& c)
// renvoie les mu(i,j)
// c(i) = ||b*_i||^2
{
    long n = B.NumCols();
    long k = B.NumRows();

    mat_RR r;
    vec_RR b,bsb;
    ZZ z1,z2;
    // bsb(i) = ||b*(i)||^2/||b(i)||^2;

    mu.SetDims(k,k);
    r.SetDims(k,k);
    c.SetLength(k);

    long i,j,l;

    for (i=1;i<=k;i++)
        for (j=1;j<=i;j++) {
            // cerr << "(i,j) = " << i << " " << j << endl;
            InnerProduct(z1,B(i),B(j));
            conv(r(i,j),z1);
            for (l=1;l<j;l++)
                r(i,j) = r(i,j)-(mu(j,l)*r(i,l));
            mu(i,j) = r(i,j)/r(j,j);

            if (j==i)
                c(i) = r(i,i);
        }

}


/* Double LLL */

double Current_Time, Tmp_Time;

void Init_Time() { Current_Time = GetTime(); }


void Display_Time(double t)
{
  cerr << "[Time = " << setw(10) << t << "s = " <<
    setw(3) << (long) (t/3600) << ":" <<
    setw(2) << (long) ((((long) t) % 3600)/60)
    << ":" << setw(2) << ((long) t) % 60 << "]\n";
}

void Elapsed_Time()
{
  Tmp_Time = GetTime()-Current_Time;
  Display_Time(Tmp_Time);
}

void Update_Time()
{
  Elapsed_Time();
  Init_Time();
}

void Display_Enum(RR number)
{
    cerr << "2^" << log(number)/log(to_RR(2)) << " nodes = " << number/power(to_RR(2),24) << " secs =  " << number/(86400*power(to_RR(2),24)) << " days\n";
}




int BDD_Enum(const mat_ZZ & B, const mat_RR & mu, const vec_RR & gsc, const vec_RR & radius, const vec_ZZ& target)
/*
    INPUT:  (mu,gs) should be the output of ModifiedGS
            on the extended matrix = B concatenated with the target vector t

            so all the information on t is the last row of mu

            gsc(i) = ||b*i||^2 should be zero for the last index i.

            radius = bounding function non-squared
            boundcoeff = upperbound on y
   OUTPUT: The basis coefficients
            The lattice vector
 */
{
    Init_Time();

    static long i,j,k;

    if (mu.NumRows() != mu.NumCols()) {
        cerr << "Non-square matrix" << endl;
        exit(1);
    }

    if (mu.NumRows() != gsc.length()) {
        cerr << "mu incompatible with gsc" << endl;
    }

    long m = mu.NumRows()-1;

    // we convert everything into double

    static  double mu_fp[500][500];
    static double gsc_fp[500];

    if (m >= 499) {
        cerr << "Dimension too high" << endl;
        exit(1);
    }

    for (i=1;i<=(m+1);i++)
        for (j=1;j<=(m+1);j++)
            conv(mu_fp[i][j],mu(i,j));

    for (i=1;i<=m;i++)
        conv(gsc_fp[i],gsc(i));

    static double bounds[500]; // squared bounds for the bounding function

    if (radius.length() != m) {
        cerr << "Bounding function not compatible with mu" << endl;
        exit(1);
    }
    for (i=1;i<=m;i++) {
        conv(bounds[i],sqr(radius(i)));
    }




    // the target t is given implicitly by \sum_i mu[m+1,i] b_i^*

    static double sigma[500][500];
    int r[600];
    static double rho[600];
    static double v[600];
    static double c[600];
    static double w[600];

    ZZ zz1,zz2,zz3;

    // Initialization

    for (i=0;i<=m;i++)
        r[i] = i;

    for (i=1;i<=(m+1);i++)
        for (j=1;j<=m;j++)
            sigma[i][j] = 0;


    rho[m+1] = 0;


    // Initialize with Babai's point
    for (k=m;k>=1;k--) {
        for (i=m;i>=(k+1);i--)
            sigma[i][k] = sigma[i+1][k] - (v[i]*mu_fp[i][k]);
        c[k] = mu_fp[m+1][k] + sigma[k+1][k];
        v[k] = rint(c[k]);
        w[k] = 1;
        rho[k] = rho[k+1] + (c[k]-v[k])*(c[k]-v[k])*gsc_fp[k];
    }

    // THIS IS THE LOOP

    k = 1;

    long nb_sol;

    nb_sol = 0;

    while (1) {
        rho[k] = rho[k+1] + (c[k]-v[k])*(c[k]-v[k])*gsc_fp[k];

        if (rho[k] <= bounds[m+1-k]) {
            if (k == 1) {
                nb_sol++;
                cerr << "BDD SOLUTION FOUND #" << nb_sol;
                Elapsed_Time();

                vec_ZZ lat;
                lat = conv<ZZ>(v[1])*B(1);
                for (i=2;i<=m;i++)
                    lat += conv<ZZ>(v[i])*B(i);

                cerr << "sqnorm #" << nb_sol << " = " << rho[1]  << endl;
                // cerr << "LATTICE VECTOR = " << lat << endl;
                cout << lat(lat.length()) << " " << rho[1] << endl; // affiche la dernière coordonnée


                if (nb_sol < 20) {
                  cerr << endl << endl << endl;
                  cerr << "basis-coefficients #" << nb_sol << " = [ ";
                  for (i=1;i<=m;i++)
                    cerr << v[i] << " ";
                  cerr << " ] " << endl;
                  cerr << "Coordinates #" << nb_sol << " = [ ";
                  for (i=1;i<=m;i++)
                    cerr << lat(i) << " ";
                  cerr << " ] " << endl;
                  cerr << endl << endl << "Last coordinate #" << nb_sol << " = " << lat(lat.length())  << endl;
                  cerr << "Size of coeffs #" << nb_sol << "= ";
                  for (i=1;i<=m;i++) {
                    cerr << NumBits(lat(i)) << " ";
                  }
                  cerr << endl;
                  cerr << "Size of target-latticevector #" << nb_sol << " = ";
                  clear(zz1);
                  for (i=1;i<=m;i++) {
                    cerr << NumBits(lat(i)-target(i)) << " ";
                    sqr(zz2,lat(i)-target(i));
                    zz1 += zz2;
                  }
                  cerr << endl;
                  cerr << "Squared distance #" << nb_sol << "= " << to_RR(zz1)  << " #bits = " << NumBits(zz1) << endl;

                  clear(zz1);
                  for (i=1;i<m;i++) {
                    sqr(zz2,lat(i)-target(i));
                    zz1 += zz2;
                  }
                  cerr << "Squared distance without last coordinate #" << nb_sol << " = " << to_RR(zz1) << " #bits = " << NumBits(zz1) << endl;
                  cerr << "Difference #" << nb_sol << "= [";
                  for (i=1;i<=m;i++)
                    cerr << " " << lat(i)-target(i);
                  cerr << "]" << endl;
                }

                k++; // go up the tree

                    r[k-1] = k;
                    if (v[k] > c[k])
                        v[k] -= w[k];
                    else
                        v[k] += w[k];
                    w[k]++;

            }
            else {
                k--; // going down the tree
                r[k-1] = max(r[k-1],r[k]);
                for (i=r[k];i>=(k+1);i--)
                        sigma[i][k] = sigma[i+1][k] - (v[i]*mu_fp[i][k]);
                c[k] = mu_fp[m+1][k] + sigma[k+1][k];
                v[k] = rint(c[k]);
                w[k] = 1;
            }
        }
        else {
            k++; // going up the tree
            if (k == (m+1)) {
                cerr << "NOTHING";
                Elapsed_Time();
                exit(1);
            }
            r[k-1] = k;
            if (v[k] > c[k])
                v[k] -= w[k];
            else
                v[k] += w[k];
            w[k]++;
        }
    }
}

int main(int argc, char * argv[])
{
    long i,j,k;

    /* On recupere les arguments */
    long prec = 150, quad = 0, rr = 0, pure = 0, chk = 0, dim = 50, bit = 8, dim_fin = 1000, index= 100, index_dep = 100,       index_fin = 100, lin = 0, square = 0, CVP = 0, succ = 0, cost = 0, dyna = 0, autom = 0, search = 0;
    double factor = 0.9, herm=1.01, toler =1,  minproba = 0.01, GH = 0;
    long kind = 0, all = 0, debug = 0, unit = 0;
    double r3 = 0, maxi = 0;
    double r2 = 0;



    for (i=1;i<argc;i++) {
        if (!strcmp(argv[i],"-h")) {
            cerr << "Performing a pruned enumeration" << endl;
            cerr << "cin = Reduced basis + target vector + (optional) bounding function (non-squared radius given as vector) " << endl;
            cerr << "Options disponibles: \n";
            cerr << "-h     : Aide.\n";
            cerr << "-sqr   : the bounding function is given as squared radius (default  = non-squared.\n";
            cerr << "-gh    : multiply the bounding function by the factor times the Gaussian heuristic.\n";
            cerr << "-GH    : multiply the sqrt(bounding function) by the factor times the Gaussian heuristic.\n";
            cerr << "-r     : square of the enumeration radius.\n";
            cerr << "-R     : enumeration radius in double.\n";
            cerr << "-factor: speedup factor when searching.\n";
            cerr << "-lin   : choose linear pruning with k-1 times the standard deviation" << endl;
            cerr << "if there is no -gh/GH -r/R, then the bounding function is absolute.\n";

            exit(1);
        }
        else if (!strcmp(argv[i],"-r")) {
            i++; r2 = atof(argv[i]); }
        else if (!strcmp(argv[i],"-gh")) {
            i++;  square = 0; GH  = atof(argv[i]); }
        else if (!strcmp(argv[i],"-max")) {
            i++;  maxi  = atof(argv[i]); }
        else if (!strcmp(argv[i],"-sqr")) {
            square  = 1; }
        else if (!strcmp(argv[i],"-lin")) {
            i++; lin = atol(argv[i]); }
        else if (!strcmp(argv[i],"-GH")) {
            i++;  square = 1; GH  = atof(argv[i]);  }
        else if (!strcmp(argv[i],"-R")) {
            i++; r3 = atof(argv[i]); }
        else if (!strcmp(argv[i],"-factor")) {
            i++; factor = atof(argv[i]); }
        else if (!strcmp(argv[i],"-proba")) {
            i++; minproba = atof(argv[i]);
            cerr << "MINIMUM PROBABILITY = " << minproba << endl;}
        else if (!strcmp(argv[i],"-seed")) {
            i++; SetSeed(to_ZZ(atol(argv[i]))); }
        else if (!strcmp(argv[i],"-prec")) {
            i++; prec = atol(argv[i]); }
        else {
            cerr << "Option inconnue !!!\n" << argv[i] << "\n";
            exit(1);
        }
    }


    cerr << "Precision RR = " << prec << endl;
    RR::SetPrecision(prec);


    RR constante = to_RR(2)*ComputePi_RR()*exp(to_RR(1));
    RR rr1,rr2,rr3,rr4,rr5;
    mat_ZZ M;

    cerr << "Reading the reduced basis" << endl;
    cin >> M;
    dim = M.NumRows();
    cerr << "Dim = " << dim << endl;

    cerr << "Reading the target vector" << endl;
    vec_RR target2;
    vec_ZZ target;
    cin >> target2;
    target.SetLength(target2.length());
    for (i=1;i<=target.length();i++)
      RoundToZZ(target(i),target2(i));



    cerr << "Computing the Gaussian heuristic." << endl;

    dim_fin = dim;

    loggamma.SetLength(max(2,dim_fin)); // loggamma(i) = log(gamma(1+i/2))
    heurist.SetLength(max(2,dim_fin));

    // Calcul de la fonction gamma

    // Gamma(3/2) = sqrt(Pi)/2
    loggamma(1) = log(ComputePi_RR())/2-log(to_RR(2));
    // Gamma(2) = 1
    clear(loggamma(2));

    // Gamma(1+x) = x Gamma(x)

    for (i=3;i<=dim_fin;i++)
        loggamma(i) = loggamma(i-2)+log(1+to_RR(i-2)/2);


    // vol unit-ball
    // vn = pi^(n/2)/gamma(1+n/2) = pi^(n/2)/exp(loggamma(n))

    unitball.SetLength(dim); // unitball(i) = volume of the unit-ball in dim i
    RR sqrtpi;
    SqrRoot(sqrtpi,ComputePi_RR());
    unitball(1) = to_RR(1);
    for (i=2;i<=dim;i++)
        div(unitball(i),power(sqrtpi,i),exp(loggamma(i)));

    for (i=2;i<=dim_fin;i++)
        heurist(i) = loggamma(i)/i-log(ComputePi_RR())/2; // heurist(i) = log( heuristique gaussienne ) = log (1/v_n^(1/n))

    rayon.SetLength(max(2,dim_fin));
    for (i=2;i<=dim_fin;i++)
        rayon(i) = exp(2*heurist(i)); // rayon(i) = squared(heuristique gaussienne);

    rayon(1) = to_RR(1)/to_RR(4);
    heurist(1) = log(rayon(1))/2;



    vec_RR bounding;
    if (lin) {
        cerr << "We choose linear bounding function with  " << lin-1 << " times the standard deviation.\n";
        bounding.SetLength(dim);
        rr4 = to_RR(1);
        for (i=1;i<=dim;i++) {
            rr1 = to_RR(i)/to_RR(dim);
            rr2 = (lin-1)*sqrt(to_RR(i)*to_RR(dim-i)/(to_RR(dim)/to_RR(2)+1))/to_RR(dim);
            add(rr3,rr1,rr2);
            SqrRoot(bounding(i),min(rr3,rr4));
        }
    }
    else {
        cerr << "We read the bounding function" << endl;
        cin >> bounding;

        if (bounding.length() != dim) {
            cerr << "The bounding function has length " << bounding.length() << " : incompatible" << endl;
            exit(1);
        }

    }


    cerr << "Computing Gram-Schmidt" << endl;

    mat_RR mu;
    vec_RR c;
    ModifiedGS(M,mu,c);


    cerr << "Computing volume" << endl;
    vec_RR volume; // volume du reseau projete: vol(i) = ||b*i|| ... ||b*n||
    volume.SetLength(dim);
    SqrRoot(volume(dim),c(dim));
    for (i=dim-1;i>=1;i--)
        mul(volume(i),volume(i+1),SqrRoot(c(i)));


    RR myradius;
    vec_RR radius;
    if (r2 > 0) {
        cerr << "Squared radius = " << r2 << endl;
        myradius = SqrRoot(to_RR(r2));
    }
    else if (r3 > 0) {
        cerr << "Double radius = " << r3 << endl;
        myradius = to_RR(r3);
    }
    else if (GH > 0) {
        cerr << "On va utiliser " << GH << " fois l'heuristique Gaussienne" << endl;
        pow(rr1,volume(1),to_RR(1)/to_RR(dim)); // rr1 = vol^1/n
        exp(rr2,heurist(dim));
        rr2 *= to_RR(GH);
        mul(myradius,rr1,rr2); // myradius = GaussianHeuristic(rayon)*GH;
    }

    cerr << "Enumeration radius = " << myradius << endl;

    cerr << "Creating the radius function" << endl;

    radius.SetLength(dim);

    if (IsZero(myradius)) {
        // no radius has been set
        cerr << "We're taking an absolute bounding function" << endl;
        if (square) {
            cerr << "We're taking square roots" << endl;
            for (i=1;i<=dim;i++)
                SqrRoot(radius(i),bounding(i));
        }
        else {
            cerr << "There is no square root" << endl;
            for (i=1;i<=dim;i++)
                radius(i) = bounding(i);
        }
    }
    else {
        cerr << "We're taking a relative bounding function" << endl;
        if (square){
            cerr << "We're taking square roots" << endl;
            for (i=1;i<=dim;i++) {
                SqrRoot(rr1,bounding(i));
                mul(radius(i),rr1,myradius);
            }
        }
        else {
            cerr << "There is no square root" << endl;
            for (i=1;i<=dim;i++) {
                mul(radius(i),myradius,bounding(i));
            }
        }
    }

    cerr << "The radius function is " << radius << endl;

    long dim2;


    mat_ZZ M2;
    cerr << "Building the extended matrix" << endl;

    long m = M.NumCols();

    if (target.length() != m) {
        cerr << "Target incompatible with the reduced basis" << endl;
        exit(1);
    }

    M2.SetDims(dim+1,m);

    for (i=1;i<=dim;i++)
        M2(i) = M(i);
    M2(dim+1) = target;

    mat_RR mu2;
    vec_RR c2;

    ModifiedGS(M2,mu2,c2);

    cerr << "Launching BDD enumeration" << endl;
    BDD_Enum(M,mu2,c2,radius,target);
}
