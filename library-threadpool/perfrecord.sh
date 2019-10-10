
outpath=perfrecord_percore
mkdir $outpath

threads=64
tasks=1000000000

./pthread/pthread.o $threads $tasks &
pid=$!
sleep 10
for core in 1 2 3 4
do
perf record -g -C $core -o $outpath/wl_pthread-core-$core.par -F 10 sleep 10 
done

./eigen/eigen.o $threads $tasks &
pid=$!
sleep 10
for core in 1 2 3 4
do
perf record -g -C $core -o $outpath/wl_eigen-core-$core.par -F 10 sleep 10 
done

./folly/folly.o $threads $tasks &
pid=$!
sleep 10
for core in 1 2 3 4
do
perf record -g -C $core -o $outpath/wl_folly-core-$core.par -F 10 sleep 10 
done

