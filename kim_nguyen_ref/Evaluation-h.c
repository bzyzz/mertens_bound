/* This file is public domain.
   Author: Phong Nguyen.
   Evaluate at y

   Examples:
  ./Evaluation-h 100 -y 14 -weight 1 -in Test-ST.txt
  ./Evaluation-h 100 -y 5 -weight 0 -in Test-Pinch.txt


   Compilation:
   gcc -larb -lflint -lgmp Evaluation-h.c -o Evaluation-h

*/

#include <string.h>
#include "arb.h"
#include "acb.h"
#include "arb_hypgeom.h"


#include "flint/profiler.h"


#define SUM_MAX 10000
#define SUBDIM_MAX 1000


int main(int argc, char *argv[])
{
    arb_t x,y,my_pi,cexp;
    slong i, j,k, prec, digits, condense, num_threads, dim, sum, interv,nb_y,rank; // sum = nb terms in the Fourier sum
    slong nu, nuy, nut, exposant;
    int constant;
    long long nombre = 14399;

    arb_t *alpha = NULL; // modulus
    arb_t *psi = NULL; // argument
    arb_t *gamma = NULL; // Imaginary part of the zero
    arb_t *alphastar = NULL;
    arb_t *sqrtalphastar = NULL;


    arb_t errors[SUBDIM_MAX]; // modulus

    char *nomfichier;
    nomfichier = "z.txt";

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
        flint_printf("usage: Evaluation-h digits [-threads n] [-y number of y values] [-weight 1 for the h-StR function, 0 for the h-Pinch function, negative for different values] [-in NOMFICHIER]\n");
        flint_printf("Evaluates the h function on y values \n");
        return 1;
    }

    digits = atol(argv[1]);

    num_threads = 1;
    condense = 20;
    constant = 0;
    sum = 1000;

    double limit = 0.92;

    i = 2;

    while (i< argc) {
       if (!strcmp(argv[i], "-y")) {
            nb_y = atol(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-limit")) {
            limit = atof(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-weight")) {
            d_cexp = atof(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-threads")) {
            num_threads = atol(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i],"-in")) {
          i++; nomfichier = argv[i]; }
        else   {
               flint_printf(argv[i]);
                flint_printf(" : unknown option \n");
                flint_abort();
        }
        i++;
    }

    flint_fprintf(stderr,"nb y = %wd\n", nb_y);

    FILE *fichier;





    flint_set_num_threads(num_threads);

    arb_init(my_pi);

    prec = digits * 3.3219280948873623 + 5;

    flint_fprintf(stderr,"precision = %wd bits... \n", prec);

    fflush(stdout);

    TIMEIT_ONCE_START
    arb_const_pi(my_pi, prec);
    arb_fprintn(stderr,my_pi, digits, ARB_STR_CONDENSE * condense);


    arb_init(cexp);

    if (d_cexp > 0) {
      flint_fprintf(stderr,"\nUsing the StR function with weight: ");
      // arb_set_d(cexp,-0.000000003); // -3*10^-9
      arb_set_str(cexp,"-0.000000003",prec);
      nombre = 98625;
    }
    else if (d_cexp == 0) {
      flint_fprintf(stderr,"\nUsing the Pinch function with weight: ");
      //arb_set_d(cexp,-0.0000015); // -1.5e-6
      arb_set_str(cexp,"-0.0000015",prec);
      nombre = 14399;
    }
    else {
      flint_fprintf(stderr,"\nUsing user weight: ");
      arb_set_d(cexp,d_cexp); // -1.5e-6
    }
    arb_fprintd(stderr,cexp, 20);
    flint_fprintf(stderr,"\n");


    flint_fprintf(stderr,"\nAllocating memory for %wd entries\n",nombre);
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

    for (i = 0; i < dim; i++)
         arb_init(errors[i]);

    for (i = 0; i < nombre; i++)
    {
         arb_init(gamma[i]);
    //    arb_load_file(rho[i],fichier);

        //fscanf(fichier,"%s",chaine);
        fgets(chaine,12000,fichier);
        /*
        if (i<2)
          printf("%s\n",chaine);
          */
        arb_set_str(gamma[i],chaine,prec);
    }
    fclose(fichier);
    flint_fprintf(stderr,"done reading gamma\n");

    flint_fprintf(stderr,"reading alpha\n");
    fichier = fopen("alpha.txt", "r");
    for (i = 0; i < nombre; i++)
    {
     arb_init(alpha[i]);
  //   arb_load_file(alpha[i],fichier);
  fgets(chaine,12000,fichier);
  arb_set_str(alpha[i],chaine,prec);
    }
    fclose(fichier);
    flint_fprintf(stderr,"done reading alpha\n");

    flint_fprintf(stderr,"reading psi\n");
    fichier = fopen("beta.txt", "r");
    for (i = 0; i < nombre; i++)
    {
     arb_init(psi[i]);
     fgets(chaine,12000,fichier);
     arb_set_str(psi[i],chaine,prec);
    }
    fclose(fichier);
    flint_fprintf(stderr,"done reading psi\n");
    fflush(stdout);



// void arb_pow_ui(arb_t y, const arb_t b, ulong e, slong prec)

   arb_t ell,x1,x2,x3,x4,x5,x6,x7,x8,produit,res,my_gamma;
/*
Computing l = \sum_j=0^{n-1} max(alpha_j^2,beta_j^2)
          s = \sum_j=0^{n-1) alpha_j^2+beta_j^2+\alpha_j \beta_j
ARF_RND_NEAR
*/

flint_fprintf(stderr,"\n");



     arb_init(y);
     arb_init(x);


 arb_t s,s1,s2,s3,t,t1,t2,t3,t4,t5,t6,t7,t8;
arb_init(s);
arb_init(s1);
arb_init(s2);
arb_init(s3);
arb_init(t);
arb_init(t1);
arb_init(t2);
arb_init(t3);
arb_init(t4);
arb_init(t5);
arb_init(t6);arb_init(t7);arb_init(t8);




arb_init(x);
arb_init(y);

arb_t seuil;
arb_init(seuil);
arb_set_d(seuil,limit);



arb_zero(s);
for (j=0;j<nombre;j++) {
  arb_init(alphastar[j]);
  arb_sqr(s1,gamma[j],prec);
  arb_mul(s2,cexp,s1,prec); // s2 = constant*gammma^2
  arb_exp(s3,s2,prec); // s3 = exp(constant*gammma^2)
  arb_mul(alphastar[j],s3,alpha[j],prec);
}

   arb_zero(s);




    flint_fprintf(stderr,"Evaluating the h-function \n\n");
    arb_t as,as1,as2,as3,e1,e2,e3;
    arb_t arb_undemi,arb_error;
    arb_t norme2,normemax;


   arb_zero(s);
   arb_init(arb_undemi);
   arb_init(arb_error);
   arb_set_d(arb_undemi,0.5); // 0.5
   arb_set_d(arb_error,0.001); // 0.5

   flint_fprintf(stderr,"Opening ");
   flint_fprintf(stderr,nomfichier);
   flint_fprintf(stderr,"\n");


   fichier = fopen(nomfichier, "r");


   for (k = 0; k < nb_y; k++)
   {
    fgets(chaine,12000,fichier);
    arb_set_str(y,chaine,prec); // reading y
    arb_zero(s);
    for (j=0;j<nombre;j++) {
      arb_mul(t1,gamma[j],y,prec); // t1 = gamma*y
      arb_sub(t2,t1,psi[j],prec); // t2 = angle = (gamma[j]*y-psi[j])
      arb_cos(t3,t2,prec); // t3 = cos(angle)
      arb_mul(t,alphastar[j],t3,prec); // t = terme de la somme (sans facteur 2)
      // t = alpha*_j*cos(gamma_j*y-psi_j)
      arb_add(s,s,t,prec); // s est la vraie somme, sans facteur 2
    }
    arb_add(t,s,s,prec); // full-h
//    arb_printn(t, 10, ARB_STR_CONDENSE * condense);
    flint_printf("h = ");
    arb_printn(t, 40,ARB_STR_CONDENSE * condense); // on affiche h
    flint_printf(" for y = ");
//    arb_printn(y, 40, ARB_STR_CONDENSE * condense);
    arb_printn(y, 40, ARB_STR_CONDENSE * condense); // on affiche y
    flint_printf("\n");

    fflush(stdout);
  }


    TIMEIT_ONCE_STOP

    // SHOW_MEMORY_USAGE


    flint_printf("\n");
    fflush(stdout);


    arb_clear(my_pi);

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
