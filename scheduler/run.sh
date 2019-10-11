
export KMP_BLOCKTIME=30
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"

outpath='async_run'

mkdir ${outpath}

loop=5

for bs in 16 128
do

  mklthread=24
  asyncthread=1
  export MKL_NUM_THREADS=$mklthread
  export OMP_NUM_THREADS=$mklthread
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  python fc.py --batch_size $bs --node 512 --proto_type async_scheduling --warmup_steps 2 --train_steps 100 --async_threads $asyncthread 1>${outpath}/wl_fc-$name.out 2>${outpath}/wl_fc-$name.err
  python fc.py --batch_size $bs --node 4096 --proto_type async_scheduling --warmup_steps 2 --train_steps 10 --async_threads $asyncthread 1>${outpath}/wl_fc4k-$name.out 2>${outpath}/wl_fc4k-$name.err
  python pretrained_inceptionv1.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_inceptionv1-$name.out 2>${outpath}/wl_inceptionv1-$name.err
  python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_inceptionv2-$name.out 2>${outpath}/wl_inceptionv2-$name.err
  python pretrained_squeezenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_squeezenet-$name.out 2>${outpath}/wl_squeezenet-$name.err
  python pretrained_googlenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_googlenet-$name.out 2>${outpath}/wl_googlenet-$name.err
  python pretrained_caffenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_caffenet-$name.out 2>${outpath}/wl_caffenet-$name.err
  python pretrained_rcnn.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_rcnn-$name.out 2>${outpath}/wl_rcnn-$name.err
    
  mklthread=12
  asyncthread=2
  export MKL_NUM_THREADS=$mklthread
  export OMP_NUM_THREADS=$mklthread
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  python fc.py --batch_size $bs --node 512 --proto_type async_scheduling --warmup_steps 2 --train_steps 100 --async_threads $asyncthread 1>${outpath}/wl_fc-$name.out 2>${outpath}/wl_fc-$name.err
  python fc.py --batch_size $bs --node 4096 --proto_type async_scheduling --warmup_steps 2 --train_steps 10 --async_threads $asyncthread 1>${outpath}/wl_fc4k-$name.out 2>${outpath}/wl_fc4k-$name.err
  python pretrained_inceptionv1.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_inceptionv1-$name.out 2>${outpath}/wl_inceptionv1-$name.err
  python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_inceptionv2-$name.out 2>${outpath}/wl_inceptionv2-$name.err
  python pretrained_squeezenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_squeezenet-$name.out 2>${outpath}/wl_squeezenet-$name.err
  python pretrained_googlenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_googlenet-$name.out 2>${outpath}/wl_googlenet-$name.err
  python pretrained_caffenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_caffenet-$name.out 2>${outpath}/wl_caffenet-$name.err
  python pretrained_rcnn.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_rcnn-$name.out 2>${outpath}/wl_rcnn-$name.err
  
  mklthread=8
  asyncthread=3
  export MKL_NUM_THREADS=$mklthread
  export OMP_NUM_THREADS=$mklthread
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  python fc.py --batch_size $bs --node 512 --proto_type async_scheduling --warmup_steps 2 --train_steps 100 --async_threads $asyncthread 1>${outpath}/wl_fc-$name.out 2>${outpath}/wl_fc-$name.err
  python fc.py --batch_size $bs --node 4096 --proto_type async_scheduling --warmup_steps 2 --train_steps 10 --async_threads $asyncthread 1>${outpath}/wl_fc4k-$name.out 2>${outpath}/wl_fc4k-$name.err
  python pretrained_inceptionv1.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_inceptionv1-$name.out 2>${outpath}/wl_inceptionv1-$name.err
  python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_inceptionv2-$name.out 2>${outpath}/wl_inceptionv2-$name.err
  python pretrained_squeezenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_squeezenet-$name.out 2>${outpath}/wl_squeezenet-$name.err
  python pretrained_googlenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_googlenet-$name.out 2>${outpath}/wl_googlenet-$name.err
  python pretrained_caffenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_caffenet-$name.out 2>${outpath}/wl_caffenet-$name.err
  python pretrained_rcnn.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_rcnn-$name.out 2>${outpath}/wl_rcnn-$name.err

  mklthread=6
  asyncthread=4
  export MKL_NUM_THREADS=$mklthread
  export OMP_NUM_THREADS=$mklthread
  name=mklthread_${mklthread}-asyncthread_${asyncthread}-batchsize_$bs
  python fc.py --batch_size $bs --node 512 --proto_type async_scheduling --warmup_steps 2 --train_steps 100 --async_threads $asyncthread 1>${outpath}/wl_fc-$name.out 2>${outpath}/wl_fc-$name.err
  python fc.py --batch_size $bs --node 4096 --proto_type async_scheduling --warmup_steps 2 --train_steps 10 --async_threads $asyncthread 1>${outpath}/wl_fc4k-$name.out 2>${outpath}/wl_fc4k-$name.err
  python pretrained_inceptionv1.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_inceptionv1-$name.out 2>${outpath}/wl_inceptionv1-$name.err
  python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_inceptionv2-$name.out 2>${outpath}/wl_inceptionv2-$name.err
  python pretrained_squeezenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_squeezenet-$name.out 2>${outpath}/wl_squeezenet-$name.err
  python pretrained_googlenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_googlenet-$name.out 2>${outpath}/wl_googlenet-$name.err
  python pretrained_caffenet.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_caffenet-$name.out 2>${outpath}/wl_caffenet-$name.err
  python pretrained_rcnn.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread 1>${outpath}/wl_rcnn-$name.out 2>${outpath}/wl_rcnn-$name.err

done

get_perf.py ${outpath}

