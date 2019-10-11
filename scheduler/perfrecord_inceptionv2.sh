export KMP_BLOCKTIME=30
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"

outpath=inceptionv2_perfrecord
mkdir $outpath

loop=100000000

for bs in 256
do

  mklthread=24
  asyncthread=1
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  rm $outpath/$name*
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread &
  pid=$!
  sleep 60
  for core in {0..4}
  do
    perf record -g -C $core -o $outpath/$name-core_$core.par -F 10 sleep 30
  done
  kill $pid

done


flamegraph.py $outpath
