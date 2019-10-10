#define PEAK_FLOPS 512

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include "mkldnn.hpp"

#include <chrono> 
using namespace std::chrono; 

using namespace mkldnn;
using dim_t = mkldnn::memory::dim;

int main(int argc, char *argv[])
{
    int i, j, loop_count, data_type;
    dim_t m, k, n;
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
    loop_count = atoi(argv[4]);
    data_type = atoi(argv[5]);
    
    //const dim_t m = 1000, k = 1000, n = 1000;
    float alpha, beta;

    //omp_set_num_threads(4);
    printf (" =============== MKLDNN ===============\n");
    //printf (" Using %i CPU cores\n", omp_get_num_threads());
    printf (" Matrix multiplication C=A*B,\n"
            " matrix A(%ldx%ld) and matrix B(%ldx%ld)\n\n", m, k, k, n);
    if (data_type == 0)
        printf (" Using float32.\n");
    else if (data_type == 1)
        printf (" Using int8(A), uint8(B) and int32(C).\n");

    alpha = 1.0; beta = 0.0;

    if (data_type == 0){
        std::vector<float> A( m*k, 1.0f);
        std::vector<float> B( k*n, 1.0f);
        std::vector<float> C( m*n, 1.0f);
        mkldnn_sgemm("N", "N", &m, &n, &k, &alpha, A.data(), &m, B.data(), &k, &beta, C.data(), &m);
        auto time_start = high_resolution_clock::now(); 
        for (i = 0; i < loop_count; i++) {
            mkldnn_sgemm("N", "N", &m, &n, &k, &alpha, A.data(), &m, B.data(), &k, &beta, C.data(), &m);
        }
        auto time_end = high_resolution_clock::now(); 
        printf("%f\n", C[0]);
    auto t = duration_cast<microseconds>(time_end - time_start).count() * 1e-6; 
    double time_avg = t/loop_count;
    double gflop = (2.0*m*n*k)*1E-9;
    printf(" Total time  : %e secs \n", t);
    printf(" Average time: %e secs \n", time_avg);
    printf(" GFlop/sec   : %.5f  \n", gflop/time_avg);
    printf(" FLOPS Util. : %.5f  \n", gflop/time_avg/PEAK_FLOPS);

    }
    else if (data_type == 1){
        int8_t ao=0, bo=0;
        int32_t co=0;
        std::vector<int8_t> A( m*k, 1);
        std::vector<uint8_t > B( k*n, 1);
        std::vector<int32_t> C( m*n, 1);
        mkldnn_gemm_s8u8s32("N", "N", "F", &m, &n, &k, &alpha, A.data(), &m, &ao, B.data(), &k, &bo, &beta, C.data(), &m, &co);
        auto time_start = high_resolution_clock::now(); 
        for (i = 0; i < loop_count; i++) {
            mkldnn_gemm_s8u8s32("N", "N", "F", &m, &n, &k, &alpha, A.data(), &m, &ao, B.data(), &k, &bo, &beta, C.data(), &m, &co);
        }
        auto time_end = high_resolution_clock::now(); 
        printf("%d\n", C[0]);
    auto t = duration_cast<microseconds>(time_end - time_start).count() * 1e-6; 
    double time_avg = t/loop_count;
    double gflop = (2.0*m*n*k)*1E-9;
    printf(" Total time  : %e secs \n", t);
    printf(" Average time: %e secs \n", time_avg);
    printf(" GFlop/sec   : %.5f  \n", gflop/time_avg);
    printf(" FLOPS Util. : %.5f  \n", gflop/time_avg/PEAK_FLOPS);
    }

    return 0;
}


///* mkl.h is required for dsecnd and SGEMM */
//#include <mkl.h>
///* initialization code is skipped for brevity (do a dummy dsecnd() call to improve accuracy of timing) */
//float alpha = 1.0, beta = 1.0;
///* first call which does the thread/buffer initialization */
//SGEMM("N", "N", &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
///* start timing after the first GEMM call */
//for (i=0; i<LOOP_COUNT; ++i)
//{
//     SGEMM("N", "N", &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
//}
//double time_end = dsecnd();
//double time_avg = (time_end - time_st)/LOOP_COUNT;
//double gflop = (2.0*m*n*k)*1E-9;
//printf("Average time: %e secs n", time_avg);
//printf("GFlop       : %.5f  n", gflop);
//printf("GFlop/sec   : %.5f  n," gflop/time_avg);&nbsp;
