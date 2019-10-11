ncore=24
PATH1=tf
PATH2=tf_1intra
mkdir $PATH1
mkdir $PATH2

for bs in 256 512 1024 2048 4096 8192 16384
do
  LOOP=1000

  node=$bs
  echo $bs $node
  NAME=Node_$node-BS_$bs
  OMP_NUM_THREADS=$ncore MKL_NUM_THREADS=$ncore python fc_tf.py --intra_threads $ncore --batch_size $bs --node $node --loop_count $LOOP 1>$PATH1/$NAME.out 2>$PATH1/$NAME.err 
  OMP_NUM_THREADS=$ncore MKL_NUM_THREADS=$ncore python fc_tf.py --intra_threads 1 --batch_size $bs --node $node --loop_count $LOOP 1>$PATH2/$NAME.out 2>$PATH2/$NAME.err 

done

python get_perf.py $PATH1
python get_perf.py $PATH2
