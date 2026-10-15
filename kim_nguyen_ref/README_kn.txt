March 3, 2024
-------------

The files Test-ST.txt and Test-Pinch.txt
contain the best values of y given in the Kim-Nguyen manuscript
To check the values, run:
  ./Evaluation-h 100 -y 14 -weight 1 -in Test-ST.txt
  ./Evaluation-h 100 -y 5 -weight 0 -in Test-Pinch.txt


Each file .c or .cpp explains how to compile the source.

The programs Evaluation-h and MakeLattice
require the data files zeroes.txt, alpha.txt and beta.txt
which are numerical values related to zeroes of the Riemann zeta function,
and computed with arb

To do new experiments:
- run MakeLattice to create the Mertens lattice basis
- run RoundMatrix to round the lattice basis into an integer basis
- reduce the matrix using fplll
- run Mertens_Enum to perform lattice enumeration tailored for the Mertens basis
- run Extract-h to compute the h values derives from the lattice points

The files SR140-130-100-30.B88-ENUM and SR140-130-100-30.B88-ENUM2
are sample files to run Mertens_Enum
They contain a BKZ-88 reduced basis together with a target vector

The corresponding sample output of Mertens_Enum
are SR140-130-100-30.B88-ENUM-1.19.OUT
and SR140-130-100-30.B88-ENUM-1.19.OUT2


The files 140-119.IN and 140-119.OUT were produced as follows:
- The positive y values from the enumeration are filtered by:
cat SR140-130-100-30.B88-ENUM-1.19.OUT* | awk '{ if ($1 > 0) print $1}' > 140-119.IN 
- Then the h values are derived by:
./Extract-h 100 -y 11754 -nu 130 -nuy 100 -nut 30 -exp 60 -dim 140 -in 140-119.IN -weight 1 > 140-119.OUT

In the 9th and 10th columns of 140-119.OUT, one can see the best values of (h,y) obtained
To see the ordered list by h value:
sort -k9  140-119.OUT


