
PATH1=tf_1intra_mklthreads_perfrecord
mkdir $PATH1

for SIZE in 128 256 512 1024 2048 4096 8192 
do
for thread in 12 24
do 
  LOOP=10000000

  echo $SIZE

  NAME=size_$SIZE-mklthreads_$thread
  #rm $PATH1/$NAME-core_*
  OMP_NUM_THREADS=$thread MKL_NUM_THREADS=$thread python fc_tf.py --intra_threads 1 --batch_size $SIZE --node $SIZE --loop_count $LOOP &
  pid=$!
  sleep 60
  for core in {0..4}
  do
    perf record -g -C $core -o $PATH1/$NAME-core_$core.par -F 10 sleep 30
  done

  kill $pid

done
done

flamegraph.py $PATH1
get_perfrecord.py $PATH1
