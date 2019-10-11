PERF_COUNTERS=instructions,cycles,L1-dcache-load-misses,L1-dcache-loads,l2_rqsts.miss,l2_rqsts.references,LLC-load-misses,LLC-store-misses,LLC-loads,LLC-stores,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.scalar_single


PATH1=tf_1intra_mklthreads_perfstat
mkdir $PATH1

for thread in 1 24
do
for bs in 128 256 1024 2048 8192 16384  
do
  LOOP=100000000

  node=$bs
  echo $bs $node
  NAME=Node_$node-BS_$bs-mklthreads_$thread
  OMP_NUM_THREADS=$thread MKL_NUM_THREADS=$thread python fc_tf.py --intra_threads 1 --batch_size $bs --node $node --loop_count $LOOP & 
  pid=$!
  sleep 60
  perf stat -e $PERF_COUNTERS -o $PATH1/$NAME.err -p $pid sleep 30 
  kill -9 $pid

done
done

get_perf.py $PATH1
