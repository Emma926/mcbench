export LD_LIBRARY_PATH=$MKLROOT/lib/intel64_lin/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKLDNNROOT/lib:$LD_LIBRARY_PATH

MKL_PATH=mkl_m_2048_perfstat
#MKLDNN_PATH=mkldnn_m_128_perfstat
#EIGEN_PATH=eigen_m_128_perfstat

mkdir $MKL_PATH
#mkdir $MKLDNN_PATH
#mkdir $EIGEN_PATH

PERF_COUNTERS=instructions,cycles,L1-dcache-load-misses,L1-dcache-loads,l2_rqsts.miss,l2_rqsts.references,LLC-load-misses,LLC-store-misses,LLC-loads,LLC-stores,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.scalar_double

M=2048

for K in  256 512 1024 2048 4096 8192
do
for N in  256 512 1024 2048 4096 8192
do

  LOOP=10000000000

  echo $M $K $N
  NAME=m_$M-k_$K-n_$N

  OMP_NUM_THREADS=4 ./mkl.o $M $K $N $LOOP &
  pid=$!
  sleep 30
  perf stat -p $pid -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_1.err sleep 10
  perf stat -p $pid -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_2.err sleep 10
  perf stat -p $pid -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_3.err sleep 10
  perf stat -p $pid -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_4.err sleep 10
  perf stat -p $pid -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_5.err sleep 10
  kill -9 $pid

#  OMP_NUM_THREADS=4 ./mkldnn.o $M $K $N $LOOP 0 &
#  pid=$!
#  sleep 30
#  perf stat -p $pid -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_1.err sleep 10
#  perf stat -p $pid -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_2.err sleep 10
#  perf stat -p $pid -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_3.err sleep 10
#  perf stat -p $pid -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_4.err sleep 10
#  perf stat -p $pid -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_5.err sleep 10
#  kill -9 $pid
#  
#  OMP_NUM_THREADS=4 ./eigen.o $M $K $N $LOOP &
#  pid=$!
#  sleep 30
#  perf stat -p $pid -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_1.err sleep 10
#  perf stat -p $pid -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_2.err sleep 10
#  perf stat -p $pid -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_3.err sleep 10
#  perf stat -p $pid -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_4.err sleep 10
#  perf stat -p $pid -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_5.err sleep 10
#  kill -9 $pid

done
done
