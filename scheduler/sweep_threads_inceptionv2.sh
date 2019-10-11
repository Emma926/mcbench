export KMP_BLOCKTIME=30
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"

outpath='inceptionv2_threads_xeon'
mkdir $outpath

loop=10

for bs in 256 #1 4 16 64 128 256
do

  mklthread=24
  asyncthread=1
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/$name.out 2>${outpath}/$name.err

  mklthread=12
  asyncthread=2
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/$name.out 2>${outpath}/$name.err

  mklthread=8
  asyncthread=3
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/$name.out 2>${outpath}/$name.err

  mklthread=6
  asyncthread=4
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/$name.out 2>${outpath}/$name.err

done

get_perf.py ${outpath}

