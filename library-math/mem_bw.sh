export LD_LIBRARY_PATH=$MKLROOT/lib/intel64_lin/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKLDNNROOT/lib:$LD_LIBRARY_PATH

num_cores=1

MKL_PATH=mkl_bw_${num_cores}-single
MKLDNN_PATH=mkldnn_bw_${num_cores}-single
EIGEN_PATH=eigen_bw_${num_cores}-single 

mkdir $MKL_PATH
mkdir $MKLDNN_PATH
mkdir $EIGEN_PATH

for SIZE in 32768 #64 128 256 512 1024 2048 4096 8192 16384 
do
  LOOP=1000000000

  NAME=size_$SIZE

  OMP_NUM_THREADS=${num_cores} ./mkl.o $SIZE $SIZE $SIZE $LOOP &
  p4=$!
  sleep 30
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_1.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_2.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_3.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_4.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKL_PATH/$NAME-try_5.err -I 1000 sleep 10 
  kill -9 $p4
  
  OMP_NUM_THREADS=${num_cores} ./mkldnn.o $SIZE $SIZE $SIZE $LOOP 0 &
  p4=$!
  sleep 30
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_1.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_2.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_3.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_4.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $MKLDNN_PATH/$NAME-try_5.err -I 1000 sleep 10 
  kill -9 $p4
  
  OMP_NUM_THREADS=${num_cores} ./eigen.o $SIZE $SIZE $SIZE $LOOP &
  p4=$!
  sleep 30
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_1.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_2.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_3.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_4.err -I 1000 sleep 10 
  perf stat -a -e uncore_imc/data_reads/,uncore_imc/data_writes/ -o $EIGEN_PATH/$NAME-try_5.err -I 1000 sleep 10 
  kill -9 $p4
  
done

