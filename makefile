test_nn: matrix.h mnist.h neural_nets.c
	gcc -Wall -pedantic -o test_nn neural_nets.c -lm -fopenmp

test_nn_omp: matrix_omp.h mnist.h neural_nets_omp.c
	gcc -Wall -pedantic -o test_nn_omp neural_nets_omp.c -lm -fopenmp