threads=64
tasks=100000

./pthread/pthread.o $threads $tasks
./eigen/eigen.o $threads $tasks
./folly/folly.o $threads $tasks
