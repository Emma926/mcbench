export LD_LIBRARY_PATH=$MKLROOT/lib/intel64_lin/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKLDNNROOT/lib:$LD_LIBRARY_PATH

num_copies=4

MKL_PATH=mkl_bw_${num_copies}-single
MKLDNN_PATH=mkldnn_bw_${num_copies}-single
EIGEN_PATH=eigen_bw_${num_copies}-single 

mkdir $MKL_PATH
mkdir $MKLDNN_PATH
mkdir $EIGEN_PATH

for SIZE in 32768 #128 256 512 1024 2048 4096 8192 16384 
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
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_1.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_2.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_3.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_4.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_5.err -I 1000 sleep 10 
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
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_1.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_2.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_3.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_4.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_5.err -I 1000 sleep 10 
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
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_1.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_2.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_3.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_4.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_5.err -I 1000 sleep 10 
  kill -9 $p1
  kill -9 $p2
  kill -9 $p3
  kill -9 $p4
  
done

