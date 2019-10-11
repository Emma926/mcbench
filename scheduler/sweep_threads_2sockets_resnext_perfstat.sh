export KMP_BLOCKTIME=30
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"

PERF_COUNTERS=instructions,cycles,LLC-load-misses,LLC-store-misses,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.scalar_single


path=resnext_threads_2sockets_xeon
mkdir $path

for bs in 256
do

#mklthread=48
#asyncthread=1
#name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
#MKL_NUM_THREADS=$mklthread OMP_NUM_THREADS=$mklthread python imagenet_trainer.py --use_cpu --train_data null --batch_size $bs --async_threads $asyncthread &
#pid=$!
#sleep 30
#perf stat -e $PERF_COUNTERS -o $path/$name.err -p $pid sleep 60 
#kill $pid

mklthread=24
asyncthread=2
name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
MKL_NUM_THREADS=$mklthread  OMP_NUM_THREADS=$mklthread python imagenet_trainer.py --use_cpu --train_data null --batch_size $bs --async_threads $asyncthread &
pid=$!
sleep 30
perf stat -e $PERF_COUNTERS -o $path/$name.err -p $pid sleep 60 
kill $pid

mklthread=16
asyncthread=3
name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
MKL_NUM_THREADS=$mklthread  OMP_NUM_THREADS=$mklthread python imagenet_trainer.py --use_cpu --train_data null --batch_size $bs --async_threads $asyncthread &
pid=$!
sleep 30
perf stat -e $PERF_COUNTERS -o $path/$name.err -p $pid sleep 60 
kill $pid

mklthread=12
asyncthread=4
name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
MKL_NUM_THREADS=$mklthread  OMP_NUM_THREADS=$mklthread python imagenet_trainer.py --use_cpu --train_data null --batch_size $bs --async_threads $asyncthread &
pid=$!
sleep 30
perf stat -e $PERF_COUNTERS -o $path/$name.err -p $pid sleep 60 
kill $pid
 
done
