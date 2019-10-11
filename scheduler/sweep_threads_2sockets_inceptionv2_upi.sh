perfcounter=uncore_imc_0/cas_count_read/,uncore_imc_0/cas_count_write/,uncore_imc_1/cas_count_read/,uncore_imc_1/cas_count_write/,uncore_imc_2/cas_count_read/,uncore_imc_2/cas_count_write/,uncore_imc_3/cas_count_read/,uncore_imc_3/cas_count_write/,uncore_imc_4/cas_count_read/,uncore_imc_4/cas_count_write/,uncore_imc_5/cas_count_read/,uncore_imc_5/cas_count_write/

outpath='inceptionv2_threads_2sockets_upi_xeon'
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
  perf stat -a -e $perfcounter -I 1000 -o $outpath/$name sleep 30
  kill $pid

  mklthread=24
  asyncthread=2
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread &
  pid=$!
  sleep 30
  perf stat -a -e $perfcounter -I 1000 -o $outpath/$name sleep 30
  kill $pid

  mklthread=16
  asyncthread=3
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread &
  pid=$!
  sleep 30
  perf stat -a -e $perfcounter -I 1000 -o $outpath/$name sleep 30
  kill $pid

  mklthread=12
  asyncthread=4
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread &
  pid=$!
  sleep 30
  perf stat -a -e $perfcounter -I 1000 -o $outpath/$name sleep 30
  kill $pid

done

get_perf.py ${outpath}

