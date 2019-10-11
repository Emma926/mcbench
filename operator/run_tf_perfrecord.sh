PATH1=tf_perfrecord
PATH2=tf_1intra_perfrecord
mkdir $PATH1
mkdir $PATH2

ncore=24
for SIZE in 512 4096 
do
  LOOP=50000000

  echo $SIZE

  NAME=size_$SIZE
  rm $PATH1/$NAME-core_*
  KMP_BLOCKTIME=1 OMP_NUM_THREADS=$ncore MKL_NUM_THREADS=$ncore python fc_tf.py --intra_threads $ncore --batch_size $SIZE --node $SIZE --loop_count $LOOP &
  pid=$!
  sleep 60
  for core in {0..4}
  do
    perf record -g -C $core -o $PATH1/$NAME-core_$core.par -F 10 sleep 30
  done
  kill $pid

  rm $PATH2/$NAME-core_*
  OMP_NUM_THREADS=$ncore MKL_NUM_THREADS=$ncore python fc_tf.py --intra_threads 1 --batch_size $SIZE --node $SIZE --loop_count $LOOP &
  pid=$!
  sleep 60
  for core in {0..4}
  do
    perf record -g -C $core -o $PATH2/$NAME-core_$core.par -F 10 sleep 30
  done
  kill $pid

done

flamegraph.py $PATH1
flamegraph.py $PATH2

get_perfrecord.py $PATH1
get_perfrecord.py $PATH2
