export LD_LIBRARY_PATH=$MKLROOT/lib/intel64_lin/:$LD_LIBRARY_PATH
gcc -o mkl.o mkl.cpp -mavx -mfma -fopenmp -lmkl_rt -m64 -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./mkl.o 521 512 512 1000

export LD_LIBRARY_PATH=$MKLDNNROOT/lib:$LD_LIBRARY_PATH
g++ -o mkldnn_test.o -std=c++11 -fopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib mkldnn.cpp -lmkldnn
OMP_DISPLAY_ENV=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./mkldnn_test.o 128 128 128 100 1

g++ -O3 -std=c++11 -fopenmp -mavx -mfma -I $EIGENROOT eigen.cpp -o eigen.o
OMP_NUM_THREADS=1 ./eigen.o 4000 4000 4000 1 
