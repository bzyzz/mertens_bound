/* This file is public domain.
 Author: Phong Nguyen.
 This arb-based program creates the Mertens lattice
*/

#include <string.h>
#include "arb.h"
#include "acb.h"
#include "arb_hypgeom.h"


#include "flint/profiler.h"
#define DIM_MAX 15000
#define nombre 14399
#define SUM_MAX 10000



int main(int argc, char *argv[])
{
    arb_t x,y,my_pi,cexp;
    slong i, j,k, prec, digits, condense, num_threads, dim, sum, interv,nb_y,rank; // sum = nb terms in the Fourier sum
    slong nu, nuy, nut;
    int constant;

    arb_t *alpha = NULL; // modulus
    arb_t *psi = NULL; // argument
    arb_t *gamma = NULL; // Imaginary part of the zero
    arb_t *alphastar = NULL;
    arb_t *sqrtalphastar = NULL;


  /*

    arb_t alpha[DIM_MAX]; // modulus
    arb_t psi[DIM_MAX]; // argument
    arb_t gamma[DIM_MAX]; // Imaginary part of the zero
    arb_t alphastar[DIM_MAX];
    arb_t sqrtalphastar[DIM_MAX];

  */

    char chaine[20000];

    nb_y = 3;
    nu = 100;
    nuy = 90;
    nut = 0;
    rank = 60;

    condense = 20;
    dim = 20;
    interv = 10;
    double d_cexp = 1;

    if (argc < 2)
    {
        flint_printf("usage: MakeLattice digits [-weight 0 = Pinch, 1 = TR ] [-nu generalbit] [-nuy bit for y] [-nut bit for coeff]  [-threads n] [-limit seuil] \n");
        flint_printf("Create the lattice of the Mertens conjecture\n");
        return 1;
    }

    digits = atol(argv[1]);

    num_threads = 1;
    condense = 20;
    constant = 0;
    sum = 1000;

    double limit = 0.92;

    for (i = 2; i < argc; i++)
    {

       if (!strcmp(argv[i], "-y")) {
            nb_y = atol(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-limit")) {
            limit = atof(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-rank")) {
            rank = atol(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-weight")) {
            d_cexp = atof(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-nu")) {
            nu = atol(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-nuy")) {
            nuy = atol(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-nut")) {
            nut = atol(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-threads")) {
            num_threads = atol(argv[i+1]);
            i++;
        }
        else    {
                flint_printf("unknown option\n");
                flint_abort();
            }
    }

    flint_fprintf(stderr,"nb y = %wd\n", nb_y);


    FILE *fichier;


    flint_set_num_threads(num_threads);

    arb_init(my_pi);

    prec = digits * 3.3219280948873623 + 5;

    flint_fprintf(stderr,"precision = %wd bits... \n", prec);
    flint_fprintf(stderr,"nu = %wd \n", nu);
    flint_fprintf(stderr,"nuy = %wd \n", nuy);
    flint_fprintf(stderr,"nut = %wd \n", nut);
    flint_fprintf(stderr,"rank = %wd \n", rank);

    fflush(stderr);

    TIMEIT_ONCE_START
    arb_const_pi(my_pi, prec);
    arb_fprintn(stderr,my_pi, digits, ARB_STR_CONDENSE * condense);

    flint_fprintf(stderr,"\nAllocating memory\n");
    alpha =  (arb_t *) malloc(sizeof(arb_t)*nombre);
    if (alpha == NULL) {
        flint_fprintf(stderr,"Not enough space for alpha\n");
        exit(0);
    }
    psi =  (arb_t *) malloc(sizeof(arb_t)*nombre);
    if (psi == NULL) {
        flint_fprintf(stderr,"Not enough space for psi\n");
        exit(0);
    }
    gamma =  (arb_t *) malloc(sizeof(arb_t)*nombre);
    if (gamma == NULL) {
        flint_fprintf(stderr,"Not enough space for gamma\n");
        exit(0);
    }
    alphastar =  (arb_t *) malloc(sizeof(arb_t)*nombre);
    if (alphastar == NULL) {
        flint_fprintf(stderr,"Not enough space for alphastar\n");
        exit(0);
    }
    sqrtalphastar =  (arb_t *) malloc(sizeof(arb_t)*nombre);
    if (sqrtalphastar == NULL) {
        flint_fprintf(stderr,"Not enough space for sqrtalphastar\n");
        exit(0);
    }

    flint_fprintf(stderr,"reading gamma\n");
    fichier = fopen("zeroes.txt", "r");
    for (i = 0; i < nombre; i++) {
        arb_init(gamma[i]);
        fgets(chaine,12000,fichier);
        arb_set_str(gamma[i],chaine,prec);
    }
    fclose(fichier);
    flint_fprintf(stderr,"done reading gamma\n");

    flint_fprintf(stderr,"reading alpha\n");
    fichier = fopen("alpha.txt", "r");
    for (i = 0; i < nombre; i++) {
      arb_init(alpha[i]);
      fgets(chaine,12000,fichier);
      arb_set_str(alpha[i],chaine,prec);
    }
    fclose(fichier);
    flint_fprintf(stderr,"done reading alpha\n");

    flint_fprintf(stderr,"reading psi\n");
    fichier = fopen("beta.txt", "r");
    for (i = 0; i < nombre; i++) {
      arb_init(psi[i]);
      fgets(chaine,12000,fichier);
      arb_set_str(psi[i],chaine,prec);
    }
    fclose(fichier);
    flint_fprintf(stderr,"done reading psi\n");
    fflush(stdout);

   arb_t ell,x1,x2,x3,x4,x5,x6,x7,x8,produit,res,my_gamma;
   flint_fprintf(stderr,"\n");

     arb_init(y);
     arb_init(x);


  arb_init(cexp);
  if (d_cexp > 0) {
    flint_fprintf(stderr,"StR weight: ");
    // arb_set_d(cexp,-0.000000003); // -3*10^-9
    arb_set_str(cexp,"-0.000000003",prec);
  }
  else if (d_cexp == 0) {
    flint_fprintf(stderr,"Pinch weight: ");
    //arb_set_d(cexp,-0.0000015); // -1.5e-6
    arb_set_str(cexp,"-0.0000015",prec);

  }
  else {
    flint_fprintf(stderr,"User weight: ");
    arb_set_d(cexp,d_cexp); // -1.5e-6
  }
  arb_fprintd(stderr,cexp, 20);
  flint_fprintf(stderr,"\n");


 arb_t s,s1,s2,s3,t,t1,t2,t3;
arb_init(s);
arb_init(s1);
arb_init(s2);
arb_init(s3);
arb_init(t);
arb_init(t1);
arb_init(t2);
arb_init(t3);

arb_init(x);
arb_init(y);

arb_t seuil;
arb_init(seuil);
arb_set_d(seuil,1);


   flint_fprintf(stderr,"Upper bound\n");
   arb_zero(s);
   for (j=0;j<nombre;j++) {
     arb_init(alphastar[j]);
     arb_sqr(s1,gamma[j],prec);
     arb_mul(s2,cexp,s1,prec); // s2 = constant*gammma^2
     arb_exp(s3,s2,prec); // s3 = exp(constant*gammma^2)
     arb_mul(alphastar[j],s3,alpha[j],prec);
     if (arb_ge(alphastar[j],seuil)) {
       flint_fprintf(stderr,"Big-alpha* %d = ",j);
       arb_fprintd(stderr,alphastar[j], 20);
       flint_fprintf(stderr," Gamma = ");
       arb_fprintd(stderr,gamma[j], 20);
       flint_fprintf(stderr," Alpha = ");
       arb_fprintd(stderr,alpha[j], 20);
       flint_fprintf(stderr," constant*gammma^2 = ");
       arb_fprintd(stderr,s2, 20);
       flint_fprintf(stderr," exp(constant*gammma^2) = ");
       arb_fprintd(stderr,s3, 20);
       flint_fprintf(stderr,"\n");
     }
     arb_add(s,s,alphastar[j],prec);
     if (j<100) {
     flint_fprintf(stderr,"Coeff-alpha %d = ",j);
     arb_fprintd(stderr,alpha[j], 20);
     flint_fprintf(stderr," alphastar %d = ",j);
     arb_fprintd(stderr,alphastar[j], 20);
     flint_fprintf(stderr," exponential = ");
     arb_fprintd(stderr,s3, 20);
     flint_fprintf(stderr," Partial sum = ");
     arb_fprintd(stderr,s, 20);
     flint_fprintf(stderr,"\n");
     }
   }
   arb_add(t,s,s,prec);
   arb_abs(s,t); // absolute value
   flint_fprintf(stderr,"Upper bound = ");
   arb_fprintd(stderr,t, 20);
   flint_fprintf(stderr,"\n");
   fflush(stderr);

   flint_fprintf(stderr,"Sorting the alphastar\n");

   long imax;
   arb_t swaparb;
   arb_zero(s);

    for (i=0;i<500;i++) {
      imax = i;
      for (j=i+1;j<nombre;j++) {
        if (arb_ge(alphastar[j],alphastar[imax]))
          imax = j;
      }
      if (imax != i) {
        arb_swap(alphastar[i],alphastar[imax]);
        arb_swap(alpha[i],alpha[imax]);
        arb_swap(psi[i],psi[imax]);
        arb_swap(gamma[i],gamma[imax]);
      }
      arb_init(sqrtalphastar[i]);
      arb_sqrt(sqrtalphastar[i],alphastar[i],prec);
      if (i<100) {
      flint_fprintf(stderr,"Sort %d index %d alpha* = ",i,imax);
      arb_fprintd(stderr,alphastar[i], 20);
      flint_fprintf(stderr," vs ");
      arb_fprintd(stderr,alphastar[imax], 20);
      flint_fprintf(stderr," alpha =  ");
      arb_fprintd(stderr,alpha[i], 20);
      flint_fprintf(stderr," vs ");
      arb_fprintd(stderr,alpha[imax], 20);
      flint_fprintf(stderr,"\n");
    }
    }


    flint_printf("Creating the target vector\n");
    flint_printf("[ ");
    for (i=0;i<rank;i++) {
      // arb_sqrt(s1,alphastar[i],prec);
      arb_mul_2exp_si(s2,psi[i],nu);
      arb_mul(s3,sqrtalphastar[i],s2,prec); // s2 = sqrtalphastar[i]*psi[i]*2^nu
      arb_floor(t,s3,prec);
      arb_printn(t,arb_bits(t)/2.5, ARB_STR_NO_RADIUS);
      flint_printf(" ");
    }
    flint_printf("0 ]\n");


    flint_printf("Creating the shifted target vector\n");
    flint_printf("[ ");
    for (i=0;i<rank;i++) {
      // arb_sqrt(s1,alphastar[i],prec);
      arb_add(t1,psi[i],my_pi,prec);
      arb_mul_2exp_si(s2,t1,nu); // s2 = (psi[i]+pi)*2^nu
      arb_mul(s3,sqrtalphastar[i],s2,prec); // s3 = sqrtalphastar[i] * (psi[i]+pi)*2^nu
      arb_floor(t,s3,prec);
      arb_printn(t,arb_bits(t)/2.5, ARB_STR_NO_RADIUS);
      flint_printf(" ");
    }
    flint_printf("0 ]\n");



    flint_printf("Creating the lattice\n");
    flint_printf("[ ");
    // 1st row
    flint_printf("[ ");
    for (i=0;i<rank;i++) {
      // arb_sqrt(s1,alphastar[i],prec);
    //  arb_add(t1,gamma[i],my_pi,prec); // pourquoi ajouter pi ?
      arb_mul_2exp_si(s2,gamma[i],nuy); // s2 = gamma[y]*2^nuy
      arb_mul(s3,sqrtalphastar[i],s2,prec); // s3 = sqrtalphastar[i] * gamma[y]*2^nuy
      arb_floor(t,s3,prec);
      arb_printn(t,arb_bits(t)/2.5, ARB_STR_NO_RADIUS);
      flint_printf(" ");
    }
    arb_one(t1);
    arb_mul_2exp_si(t,t1,nut); // t = 2^nut
    arb_printn(t,nut+2, ARB_STR_NO_RADIUS);
    flint_printf("]\n");
    // diagonal matrix_dif
    for (j=0;j<rank;j++) {
      flint_printf("[ ");
      for (k=0;k<j;k++)
        flint_printf("0 ");
      arb_mul_2exp_si(t1,sqrtalphastar[j],nu+1);
      arb_mul(t2,t1,my_pi,prec);
      arb_floor(t,t2,prec);
      arb_printn(t,arb_bits(t)/2.5, ARB_STR_NO_RADIUS);
      flint_printf(" ");

      for (k=j+1;k<rank+1;k++)
        flint_printf("0 ");
      flint_printf("]\n");
    }
    flint_printf("]\n");


    TIMEIT_ONCE_STOP

    // SHOW_MEMORY_USAGE


    flint_printf("\n");
    fflush(stdout);


    arb_clear(my_pi);
    arb_clear(x); arb_clear(y); arb_clear(cexp);


    for (i=0;i<nombre;i++) {
      arb_clear(alpha[i]);
      arb_clear(psi[i]);
      arb_clear(gamma[i]);
      arb_clear(alphastar[i]);
      arb_clear(sqrtalphastar[i]);

  }

  free(alpha);  free(psi); free(gamma); free(alphastar); free(sqrtalphastar);

    flint_cleanup_master();
    return 0;
}
