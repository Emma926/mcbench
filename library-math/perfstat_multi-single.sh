export LD_LIBRARY_PATH=$MKLROOT/lib/intel64_lin/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKLDNNROOT/lib:$LD_LIBRARY_PATH

PERF_COUNTERS=instructions,cycles,L1-dcache-load-misses,L1-dcache-loads,l2_rqsts.miss,l2_rqsts.references,LLC-load-misses,LLC-store-misses,LLC-loads,LLC-stores,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.scalar_double

num_cores=4

MKL_PATH=mkl_perfstat_${num_cores}-single
MKLDNN_PATH=mkldnn_perfstat_${num_cores}-single
EIGEN_PATH=eigen_perfstat_${num_cores}-single 

mkdir $MKL_PATH
mkdir $MKLDNN_PATH
mkdir $EIGEN_PATH

for SIZE in 16384 #64 128 256 512 1024 2048 4096 8192 16384 32768 
do
  LOOP=1000000000

  NAME=size_$SIZE
  OMP_NUM_THREADS=1 ./mkl.o $SIZE $SIZE $SIZE $LOOP &
  p1=$!
  OMP_NUM_THREADS=1 ./mkl.o $SIZE $SIZE $SIZE $LOOP &
  p2=$!
  OMP_NUM_THREADS=1 ./mkl.o $SIZE $SIZE $SIZE $LOOP &
  p3=$!
  OMP_NUM_THREADS=1 ./mkl.o $SIZE $SIZE $SIZE $LOOP &
  p4=$!
  sleep 30
  perf stat -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_1.err -p $p4 sleep 10 
  perf stat -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_2.err -p $p4 sleep 10
  perf stat -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_3.err -p $p4 sleep 10
  perf stat -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_4.err -p $p4 sleep 10
  perf stat -e $PERF_COUNTERS -o $MKL_PATH/$NAME-try_5.err -p $p4 sleep 10
  kill -9 $p1
  kill -9 $p2
  kill -9 $p3
  kill -9 $p4
  
  OMP_NUM_THREADS=1 ./mkldnn.o $SIZE $SIZE $SIZE $LOOP 0 &
  p1=$!
  OMP_NUM_THREADS=1 ./mkldnn.o $SIZE $SIZE $SIZE $LOOP 0 &
  p2=$!
  OMP_NUM_THREADS=1 ./mkldnn.o $SIZE $SIZE $SIZE $LOOP 0 &
  p3=$!
  OMP_NUM_THREADS=1 ./mkldnn.o $SIZE $SIZE $SIZE $LOOP 0 &
  p4=$!
  sleep 30
  perf stat -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_1.err -p $p4 sleep 10 
  perf stat -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_2.err -p $p4 sleep 10 
  perf stat -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_3.err -p $p4 sleep 10 
  perf stat -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_4.err -p $p4 sleep 10 
  perf stat -e $PERF_COUNTERS -o $MKLDNN_PATH/$NAME-try_5.err -p $p4 sleep 10 
  kill -9 $p1
  kill -9 $p2
  kill -9 $p3
  kill -9 $p4
  
  OMP_NUM_THREADS=1 ./eigen.o $SIZE $SIZE $SIZE $LOOP &
  p1=$!
  OMP_NUM_THREADS=1 ./eigen.o $SIZE $SIZE $SIZE $LOOP &
  p2=$!
  OMP_NUM_THREADS=1 ./eigen.o $SIZE $SIZE $SIZE $LOOP &
  p3=$!
  OMP_NUM_THREADS=1 ./eigen.o $SIZE $SIZE $SIZE $LOOP &
  p4=$!
  sleep 30
  perf stat -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_1.err -p $p4 sleep 10 
  perf stat -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_2.err -p $p4 sleep 10 
  perf stat -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_3.err -p $p4 sleep 10 
  perf stat -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_4.err -p $p4 sleep 10 
  perf stat -e $PERF_COUNTERS -o $EIGEN_PATH/$NAME-try_5.err -p $p4 sleep 10 

  kill -9 $p1
  kill -9 $p2
  kill -9 $p3
  kill -9 $p4
  
done

