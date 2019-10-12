## Setup
Assuming conda is installed, run
```
bash setup.sh
```

## Get model graphs

Add these to the code:

```
from caffe2.python import net_drawer
g = net_drawer.GetPydotGraph(predict_net, rankdir="TB")
g.write_dot('test.dot')
```

and run
```
dot -Tpng test.dot -o test.png
```

## Experiments

### Varies asynchronous scheduling
```
bash run.sh
```

### Sweep batch size and async scheduling and collect performance or performance counters
```
bash sweep_threads_*.sh
```

### Stack trace of Inception v2
```
bash perfrecord_inceptionv2.sh
```

### Execution trace of Inception v2
```
bash perf_trace.sh
```
