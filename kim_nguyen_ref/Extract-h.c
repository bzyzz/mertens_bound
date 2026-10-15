/*
   Author: Phong Nguyen.
   Extract values from enumeration at y
   Check whether approximation match the computation
   Check correlations
   write h_partial approx_h_partial ||_2 ||_infinity h y

   Example:
  ./Extract-h 100 -y 11754 -nu 130 -nuy 100 -nut 30 -exp 60 -dim 140 -in 140-119.IN -weight 1

  Compilation:
  gcc -larb -lflint -lgmp Extract-h.c -o Extract-h


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
        flint_printf("usage: Evaluation-h digits [-nu generalbit] [-nuy bit for y] [-nut bit for coeff]  [-threads n] [-limit seuil] [-in NOMFICHIER]\n");
        flint_printf("Display statistics h_partial ||_2 ||_inf h y \n");
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
        else if (!strcmp(argv[i], "-exp")) {
            exposant = atof(argv[i+1]);
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
        else if (!strcmp(argv[i], "-dim")) {
            dim = atof(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i], "-threads")) {
            num_threads = atol(argv[i+1]);
            i++;
        }
        else if (!strcmp(argv[i],"-in")) {
          i++; nomfichier = argv[i]; }
        else    {
                flint_printf("unknown option\n");
                flint_abort();
            }
    }

    flint_fprintf(stderr,"nb y = %wd\n", nb_y);
    flint_fprintf(stderr,"Dim = %wd\n", dim);
    flint_fprintf(stderr,"2-power = %wd\n", exposant);
    flint_fprintf(stderr,"nu  = %wd\n", nu);
    flint_fprintf(stderr,"nuy = %wd\n", nuy);
    flint_fprintf(stderr,"nut = %wd\n", nut);



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
   arb_t variance;
/*
Computing l = \sum_j=0^{n-1} max(alpha_j^2,beta_j^2)
          s = \sum_j=0^{n-1) alpha_j^2+beta_j^2+\alpha_j \beta_j
ARF_RND_NEAR
*/

flint_fprintf(stderr,"\n");



     arb_init(y);
     arb_init(x);
     arb_init(variance);




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


   flint_fprintf(stderr,"Upper bound\n");
   arb_zero(s);
   arb_zero(variance);
   for (j=0;j<nombre;j++) {
     arb_init(alphastar[j]);
     arb_sqr(s1,gamma[j],prec);
     arb_mul(s2,cexp,s1,prec); // s2 = constant*gammma^2
     arb_exp(s3,s2,prec); // s3 = exp(constant*gammma^2)
     arb_mul(alphastar[j],s3,alpha[j],prec);
     arb_sqr(s1,alphastar[j],prec);
     arb_addmul_ui(variance,s1,2,prec); // v += 2*sqr(alphastar)
     arb_add(s,s,alphastar[j],prec);

   }
   arb_add(t,s,s,prec);
   arb_abs(s,t); // absolute value

   arb_sqrt(s1,variance,prec);

   fflush(stderr);

   flint_fprintf(stderr,"Sorting the alphastar\n");

   long imax;
   arb_t swaparb;
   arb_zero(s);

    for (i=0;i<500;i++) {
      imax = i;
      // on cherche le maximum entre i et number
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
      /*
      if (i<140) {
      flint_printf("Sort %d = ",i);
      arb_printd(alphastar[i], 20);
      flint_printf("\n");
    }
    */
      arb_init(sqrtalphastar[i]);
      arb_sqrt(sqrtalphastar[i],alphastar[i],prec);
    }




    arb_zero(variance);
    arb_zero(s);


        arb_add(s1,my_pi,my_pi,prec); // as1 = 2pi


    flint_fprintf(stderr,"\n\nEvaluating special values\n 1=h_partial 2=approx_h_partial 3=||_2 4=||_infinity 5=sq_latt_distance_withoutlast coord 6=sq_latt_dist 7=sq_latt_infdistwithoutlastcoord 8= sq_latt_distinf 9=h 10=y \n\n");
    arb_t as,as1,as2,as3,e1,e2,e3;
    arb_t arb_undemi,arb_error;
    arb_t norme2,normemax;


   arb_zero(s);
   arb_init(as);
   arb_init(norme2);   // carré de la norme euclidienne
   arb_init(normemax); // carré de la norme max
   arb_init(as1);
   arb_init(as2);
   arb_init(as3);
   arb_init(e1);
   arb_init(e2);
   arb_init(e3);
   arb_init(arb_undemi);
   arb_init(arb_error);
   arb_set_d(arb_undemi,0.5); // 0.5
   arb_set_d(arb_error,0.001); // 0.5

   flint_fprintf(stderr,"Opening ");
   flint_fprintf(stderr,nomfichier);
   flint_fprintf(stderr,"\n");


   fichier = fopen(nomfichier, "r");

   arb_add(as1,my_pi,my_pi,prec); // as1 = 2pi

   for (k = 0; k < nb_y; k++)
   {
    fgets(chaine,12000,fichier);
    arb_set_str(x,chaine,prec);
    arb_mul_2exp_si(y,x,-exposant);

    arb_zero(s);
    arb_zero(as);
    arb_zero(norme2);
    arb_zero(normemax);

// il ne faut pas oublier de doubler les résultats
    arb_zero(s1); // s1 est la pire erreur
    arb_zero(e1); // e1 est la pire erreur du cosinus
    for (j=0;j<nombre;j++) {
      if (j==dim) {
        // on affiche la différence dans stderr



        // on affiche dans stdout
        arb_add(t,s,s,prec); // on double la somme
      //  arb_printn(t, 10, ARB_STR_CONDENSE * condense); // partial-h
      arb_printn(t, 10, ARB_STR_NO_RADIUS);
        flint_printf(" ");
      arb_add(t,as,as,prec); // on double la somme
      arb_printn(t, 10, ARB_STR_NO_RADIUS);
      flint_printf(" "); // approximation


        arb_add(t,norme2,norme2,prec);
    //    arb_printn(t, 10, ARB_STR_CONDENSE * condense);
    arb_printn(t, 10, ARB_STR_NO_RADIUS);
        flint_printf(" ");
        arb_add(t,normemax,normemax,prec);
//        arb_printn(t, 10, ARB_STR_CONDENSE * condense);
        arb_printn(t, 10, ARB_STR_NO_RADIUS);
        flint_printf(" ");


        // maintenant on rajoute la norme euclidienne

        arb_set(t1,x);
        arb_zero(t7);
        arb_zero(t8);

      // t1 est le multiplicateur
      for (i=0;i<dim;i++) {
        arb_mul(t2,sqrtalphastar[i],gamma[i],prec);
        arb_mul_2exp_si(t3,t2,nuy); // t3 = sqrtalphastar[i] * gamma[i] * 2^90
        arb_floor(t2,t3,prec); // we used floor to create the latticevol
        arb_mul(t3,t2,t1,prec); // on multiplie par le multiplicateur

        arb_mul(t,sqrtalphastar[i],psi[i],prec);
        arb_mul_2exp_si(t2,t,nu);
        arb_sub(t,t3,t2,prec); // t = 1st row * t1 - target
        // now we have to reduce modulo the diagonal


       arb_mul(t2,sqrtalphastar[i],as1,prec);
       arb_mul_2exp_si(t4,t2,nu);
       arb_floor(t3,t4,prec);
       //arb_set_round(t3,t4,prec);
       // t3 = round(sqrtalphastar[i] * 2pi * 2^120)

       // now we have to take t modulo t3

       arb_div(t4,t,t3,prec); // t4 = t/t3
       arb_floor(t5,t4,prec); // t5 = floor(t/t3)
       arb_mul(t6,t3,t5,prec);
       arb_sub(t4,t,t6,prec); // reste positif
       arb_mul_2exp_si(t5,t3,-1); // divise t3 par 2
       if (arb_ge(t4,t5))
        arb_sub(t4,t4,t3,prec);

      arb_sqr(t5,t4,prec);
      if (arb_ge(t5,t8)) // max_dist
        arb_set(t8,t5);
      arb_add(t7,t7,t5,prec);
     }
//     flint_fprintf(stderr,"Squared dist without last coordinate = ");
     arb_mul_2exp_si(t6,t7,-2*nu);
     arb_printn(t6, 10, ARB_STR_NO_RADIUS);
     flint_printf(" ");
     arb_sqr(t5,x,prec);
     arb_add(t7,t7,t5,prec);
     arb_mul_2exp_si(t6,t7,-2*nu);
  //   flint_fprintf(stderr,"Squared dist = ");
  // on rajoute les distances infinies
     arb_printn(t6, 10, ARB_STR_NO_RADIUS);
     flint_printf(" ");

     arb_mul_2exp_si(t6,t8,-2*nu);
     arb_printn(t6, 10, ARB_STR_NO_RADIUS);
     flint_printf(" ");

     if (arb_ge(t5,t8)) // max_dist
       arb_set(t8,t5);
      arb_mul_2exp_si(t6,t8,-2*nu);
      arb_printn(t6, 10, ARB_STR_NO_RADIUS);
      flint_printf(" ");




      }

      arb_mul(t1,gamma[j],y,prec); // t1 = gamma*y
      arb_sub(t2,t1,psi[j],prec); // t2 = angle = (gamma[j]*y-psi[j])
      arb_cos(t3,t2,prec); // t3 = cos(angle)
      // t2 modulo 2pi
      arb_div(as2,t2,as1,prec); // as2 = t2/(2pi)
      arb_floor(as3,as2,prec); // as3 = floor(t2/(2pi))
      arb_mul(as2,as3,as1,prec);
      arb_sub(as3,t2,as2,prec);
      if (arb_ge(as3,my_pi))
       arb_sub(as3,as3,as1,prec);


       // as3 = (gamma_i*y-psi_i) mod 2\pi (least residue in absolute value)




      arb_sqr(t2,as3,prec); // t2 = (interieur mod 2pi)^2 = [(gamma_i*y-psi_i) mod 2\pi]^2

      arb_mul_2exp_si(as2,t2,-1); // as2 = [(gamma_i*y-psi_i) mod 2\pi]^2/2
      arb_one(t1);
      arb_sub(as3,t1,as2,prec); // as3 = approx(term) qui remplace t3. On approche cos(x) par 1-x^2/2
      // as3 = 1-[(gamma_i*y-psi_i) mod 2\pi]^2/2
      arb_mul(t,alphastar[j],t3,prec); // t = terme de la somme (sans facteur 2)
      // t = alpha*_j*cos(gamma_j*y-psi_j)

    //  arb_mul(t,t1,alpha[j],prec);
      arb_add(s,s,t,prec); // s est la vraie somme, sans facteur 2
  //    arb_mul(t1,s3,as3,prec);

      arb_mul(as2,as3,alphastar[j],prec); // as2 = approx(term) qui remplace t
      // as2 = alpha*_j*     1-[(gamma_i*y-psi_i) mod 2\pi]^2/2
      arb_add(as,as,as2,prec); // as est la vraie somme approchée, sans facteur 2

        if (j < dim) {
          arb_mul(s3,t2,alphastar[j],prec);
          // s3 = alphastar[j] * [(gamma_i*y-psi_i) mod 2\pi]^2
          arb_max(normemax,normemax,s3,prec);
          arb_add(norme2,norme2,s3,prec);
        }



    }
    arb_add(t,s,s,prec); // full-h
//    arb_printn(t, 10, ARB_STR_CONDENSE * condense);
    arb_printn(t, 10,ARB_STR_NO_RADIUS); // on affiche h
    flint_printf(" ");
//    arb_printn(y, 40, ARB_STR_CONDENSE * condense);
    arb_printn(y, 40, ARB_STR_NO_RADIUS); // on affiche y

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
