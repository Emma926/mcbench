PERF_COUNTERS=instructions,cycles,L1-dcache-load-misses,L1-dcache-loads,l2_rqsts.miss,l2_rqsts.references,LLC-load-misses,LLC-store-misses,LLC-loads,LLC-stores,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.scalar_double

PATH1=tf_2d
mkdir $PATH1

for bs in 128 256 512 1024 2048 4096 8192 #16384
do
for node in  128 256 512 1024 2048 4096 8192 #16384
do
  LOOP=10000000000

  echo $bs $node
  NAME=Node_$node-BS_$bs
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python fc_tf.py --intra_threads 4 --batch_size $bs --node $node --loop_count $LOOP &
  pid=$!
  sleep 30
  perf stat -e $PERF_COUNTERS -o $PATH1/$NAME-try_1.err -p $pid sleep 10 
  perf stat -e $PERF_COUNTERS -o $PATH1/$NAME-try_2.err -p $pid sleep 10 
  perf stat -e $PERF_COUNTERS -o $PATH1/$NAME-try_3.err -p $pid sleep 10 
  perf stat -e $PERF_COUNTERS -o $PATH1/$NAME-try_4.err -p $pid sleep 10 
  perf stat -e $PERF_COUNTERS -o $PATH1/$NAME-try_5.err -p $pid sleep 10 
  kill -9 $pid

done
done

get_perf.py $PATH1
