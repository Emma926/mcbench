export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=$MKL_NUM_THREADS
asyncthread=1

outpath=inceptionv2_${MKL_NUM_THREADS}mklthread_${asyncthread}asyncthread_perfstat
#rm -rf $outpath
mkdir $outpath

bs=128
loop=500000000

taskset -c 0,1,2,3 python pretrained_inceptionv2.py --batch_size $bs --proto_type async_scheduling --steps $loop --async_threads $asyncthread &
pid=$!
sleep 60
start=`date +%s`
for try in {1..3000}
do
  perf stat -a --per-core sleep 0.001 2> $outpath/trace_$try.err
done
end=`date +%s`
kill $pid
runtime=$((end-start))
echo $runtime

outpathnew=inceptionv2_${MKL_NUM_THREADS}mklthread_${asyncthread}asyncthread_perfstat_percore
rm -rf $outpathnew
mkdir $outpathnew

for try in {1..3000}
do
  grep "Performance counter stats" $outpath/trace_$try.err > $outpathnew/core_0-trace_$try.err
  grep "S0-C0" $outpath/trace_$try.err >> $outpathnew/core_0-trace_$try.err
  grep "seconds" $outpath/trace_$try.err >> $outpathnew/core_0-trace_$try.err

  grep "Performance counter stats" $outpath/trace_$try.err > $outpathnew/core_1-trace_$try.err
  grep "S0-C1" $outpath/trace_$try.err >> $outpathnew/core_1-trace_$try.err
  grep "seconds" $outpath/trace_$try.err >> $outpathnew/core_1-trace_$try.err

  grep "Performance counter stats" $outpath/trace_$try.err > $outpathnew/core_2-trace_$try.err
  grep "S0-C2" $outpath/trace_$try.err >> $outpathnew/core_2-trace_$try.err
  grep "seconds" $outpath/trace_$try.err >> $outpathnew/core_2-trace_$try.err

  grep "Performance counter stats" $outpath/trace_$try.err > $outpathnew/core_3-trace_$try.err
  grep "S0-C3" $outpath/trace_$try.err >> $outpathnew/core_3-trace_$try.err
  grep "seconds" $outpath/trace_$try.err >> $outpathnew/core_3-trace_$try.err
done

get_perf.py $outpathnew trace
