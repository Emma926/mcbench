export LD_LIBRARY_PATH=$MKLROOT/lib/intel64_lin/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKLDNNROOT/lib:$LD_LIBRARY_PATH

num_cores=1

MKL_PATH=mkl_td_${num_cores}-single
MKLDNN_PATH=mkldnn_td_${num_cores}-single
EIGEN_PATH=eigen_td_${num_cores}-single 

mkdir $MKL_PATH
mkdir $MKLDNN_PATH
mkdir $EIGEN_PATH

for SIZE in 128 256 512 1024 2048 4096 8192 16384 32768
do
  LOOP=100000000000

  OMP_NUM_THREADS=1 ./mkl.o $SIZE $SIZE $SIZE $LOOP &
  pid=$!
  sleep 30 
  perf stat --topdown -a -o $MKL_PATH/size_$SIZE-try_1.err -- sleep 10
  perf stat --topdown -a -o $MKL_PATH/size_$SIZE-try_2.err -- sleep 10
  perf stat --topdown -a -o $MKL_PATH/size_$SIZE-try_3.err -- sleep 10
  perf stat --topdown -a -o $MKL_PATH/size_$SIZE-try_4.err -- sleep 10
  perf stat --topdown -a -o $MKL_PATH/size_$SIZE-try_5.err -- sleep 10
  kill -9 $pid
  
  OMP_NUM_THREADS=1 ./mkldnn.o $SIZE $SIZE $SIZE $LOOP 0 &
  pid=$!
  sleep 30 
  perf stat --topdown -a -o $MKLDNN_PATH/size_$SIZE-try_1.err -- sleep 10
  perf stat --topdown -a -o $MKLDNN_PATH/size_$SIZE-try_2.err -- sleep 10
  perf stat --topdown -a -o $MKLDNN_PATH/size_$SIZE-try_3.err -- sleep 10
  perf stat --topdown -a -o $MKLDNN_PATH/size_$SIZE-try_4.err -- sleep 10
  perf stat --topdown -a -o $MKLDNN_PATH/size_$SIZE-try_5.err -- sleep 10
  kill -9 $pid
  
  OMP_NUM_THREADS=1 ./eigen.o $SIZE $SIZE $SIZE $LOOP &
  pid=$!
  sleep 30 
  perf stat --topdown -a -o $EIGEN_PATH/size_$SIZE-try_1.err -- sleep 10
  perf stat --topdown -a -o $EIGEN_PATH/size_$SIZE-try_2.err -- sleep 10
  perf stat --topdown -a -o $EIGEN_PATH/size_$SIZE-try_3.err -- sleep 10
  perf stat --topdown -a -o $EIGEN_PATH/size_$SIZE-try_4.err -- sleep 10
  perf stat --topdown -a -o $EIGEN_PATH/size_$SIZE-try_5.err -- sleep 10
  kill -9 $pid
  
done

