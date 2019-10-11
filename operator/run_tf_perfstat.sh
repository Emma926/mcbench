PERF_COUNTERS=instructions,cycles,L1-dcache-load-misses,L1-dcache-loads,l2_rqsts.miss,l2_rqsts.references,LLC-load-misses,LLC-store-misses,LLC-loads,LLC-stores,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.scalar_single

export KMP_BLOCKTIME=30
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

path=tf_2socket_perfstat
mkdir $path

for bs in 128 256 512 1024 2048 4096 8192 16384
do
  LOOP=1000000

  node=$bs
  echo $bs $node

  export MKL_NUM_THREADS=24
  export OMP_NUM_THREADS=$MKL_NUM_THREADS
  intrathreads=$MKL_NUM_THREADS
  name=Node_$node-BS_$bs-mklthreads_${MKL_NUM_THREADS}-intrathreads_${intrathreads}
  OMP_NUM_THREADS=$MKL_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS python fc_tf.py --intra_threads $intrathreads --batch_size $bs --node $node --loop_count $LOOP & 
  pid=$!
  sleep 30
  perf stat -e $PERF_COUNTERS -o $path/$name.err -p $pid sleep 30 
  kill $pid

  export MKL_NUM_THREADS=48
  export OMP_NUM_THREADS=$MKL_NUM_THREADS
  intrathreads=$MKL_NUM_THREADS
  name=Node_$node-BS_$bs-mklthreads_${MKL_NUM_THREADS}-intrathreads_${intrathreads}
  OMP_NUM_THREADS=$MKL_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS python fc_tf.py --intra_threads $intrathreads --batch_size $bs --node $node --loop_count $LOOP & 
  pid=$!
  sleep 30
  perf stat -e $PERF_COUNTERS -o $path/$name.err -p $pid sleep 30 
  kill $pid

done

get_perf.py $PATH1
