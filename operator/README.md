## Prerequisite

TensorFlow

## Experiments

### Simply run and collect run-times, with and without intra-op threads
```
bash run_tf.sh
``` 

### Sweep Matrix sizes and collect performance counters
```
bash run_tf_perfstat.sh 
```

### Sweep MKL threads and collect performance counters
```
bash run_tf_perfstat_mklthreads.sh 
```

### Sweep MKL threads and collect per-core stack trace
```
bash run_tf_perfrecord_mklthreads.sh 
```

### Run with and without intra-op threads and collect per-core stack trace
```
bash run_tf_perfrecord.sh
```

### Sweep batch size and node size and collect performance counters
```
bash run_tf_perfstat_2d.sh
```
