#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

int main(int argc, char *argv[])
{
    float *A, *B, *C;
    int m, n, k, i, j, loop_count;
    float alpha, beta;

    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
    loop_count = atoi(argv[4]);
    if (loop_count < 10){
        printf("Error: loop_count needs to be over 10, and a multiply of 10.\n");
        return 0;
    }
    printf (" =============== MKL ==================\n");
    printf (" Matrix multiplication C=A*B,\n"
            " matrix A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    A = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
    B = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    for (i = 0; i < (m*k); i++) {
        A[i] = (float)(1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (float)(1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 1.0;
    }

    SGEMM("N", "N", &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
    
    double time_start = dsecnd();
    double time_end, time_avg, gflop;
    gflop = (2.0*m*n*k)*1E-9;
    for (i = 1; i < loop_count + 1; i++) {
        SGEMM("N", "N", &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
        if(i % (loop_count/10) == 0){
            time_end = dsecnd();
            time_avg = (time_end - time_start) / (loop_count/10);
            printf("----- During %d to %d steps -----\n", i - (loop_count/10) + 1, i);
            printf(" Average time: %e secs \n", time_avg);
            printf(" GFlop/sec   : %.5f  \n", gflop/time_avg);
            time_start = dsecnd(); 
        }
    }

    printf("C[0, 0] = %f\n", C[0, 0]);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}

