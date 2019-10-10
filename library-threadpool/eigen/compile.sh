# Modify eigen root
export EIGENROOT=eigen
g++ -O3 -std=c++11 -fopenmp -mavx -mfma -I $EIGENROOT eigen.cpp -o eigen.o
