#define PEAK_FLOPS 512

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <chrono> 
using namespace std::chrono; 

using namespace Eigen;

int main(int argc, char *argv[])
{
    Eigen::initParallel();
    int m, n, k, i, j, loop_count;
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
    loop_count = atoi(argv[4]);

    Eigen::MatrixXf A(m, k);
    Eigen::MatrixXf B(k, n);
    Eigen::MatrixXf C(m, n);
    
    printf (" =============== Eigen ================\n");
    printf (" Using %i CPU cores\n", Eigen::nbThreads( ));
    printf (" Matrix multiplication C=A*B,\n"
            " matrix A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);

    C.noalias() += A*B;
    auto time_start = high_resolution_clock::now(); 
    for (i = 0; i < loop_count; i++) {
      C.noalias() += A*B;
    }

    auto time_end = high_resolution_clock::now(); 
    auto t = duration_cast<microseconds>(time_end - time_start).count() * 1e-6; 
    double time_avg = t/loop_count;
    double gflop = (2.0*m*n*k)*1E-9;
    printf(" Total time  : %e secs \n", t);
    printf(" Average time: %e secs \n", time_avg);
    printf(" GFlop/sec   : %.5f  \n", gflop/time_avg);
    printf(" FLOPS Util. : %.5f  \n", gflop/time_avg/PEAK_FLOPS);

}
