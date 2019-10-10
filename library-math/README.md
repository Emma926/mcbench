# 1. Install libraries and tools

## MKL

https://software.intel.com/en-us/mkl/choose-download

## MKL-DNN

https://github.com/intel/mkl-dnn

## Eigen

https://eigen.tuxfamily.org/dox/GettingStarted.html

## Other tools

Flamegraph: 

https://github.com/brendangregg/FlameGraph


# 2. Modify env.sh
Modify libraries locations in env.sh and do `source env.sh`

# 3. Compile
```
bash compile.sh
```

# 4. Experiments

### Simple run and collect run-times
```
bash run.sh
```

### Collect single-threaded performance counters
```
bash perfstat-single.sh
```

### Run multiple copies of single-threaded process and collect performance counters for one process
```
bash perfstat_multi-single.sh
```

### Sweep three dimensions of matrix multiplication and collect performance counters
```
bash sweep_3dim_perfstat.sh 
```

### Collect top-level statistics of single-threaded workloads
```
bash td-single.sh
```

### Collect memory bandwidth of one process
```
bash mem_bw.sh
```

### Collect memory bandwidth of one process when running multiple processes
```
bash membw_multi-single.sh
```
