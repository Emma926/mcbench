perfcounters=instructions,cycles,LLC-load-misses,LLC-store-misses,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.scalar_single

outpath='inceptionv2_threads_2sockets_flops_xeon'
mkdir $outpath

loop=10000

export KMP_BLOCKTIME=30
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"

for bs in 256 #1 4 16 64 128 256
do

  mklthread=48
  asyncthread=1
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread &
  pid=$!
  sleep 30
  perf stat -p $pid -e $perfcounters -o $outpath/$name.err sleep 30
  kill $pid

  mklthread=24
  asyncthread=2
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread &
  pid=$!
  sleep 30
  perf stat -p $pid -e $perfcounters -o $outpath/$name.err sleep 30
  kill $pid

  mklthread=16
  asyncthread=3
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread &
  pid=$!
  sleep 30
  perf stat -p $pid -e $perfcounters -o $outpath/$name.err sleep 30
  kill $pid

  mklthread=12
  asyncthread=4
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread &
  pid=$!
  sleep 30
  perf stat -p $pid -e $perfcounters -o $outpath/$name.err sleep 30
  kill $pid

done

get_perf.py ${outpath}

