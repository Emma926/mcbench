# mcbench
Mille Crepe Bench: multi-layer performance analysis for deep learning frameworks

This repository contains the scripts and workloads of our deep learning framework analysis paper. If
you find it useful, please cite [our paper](https://arxiv.org/abs/1908.04705).

```
Wang, Yu Emma, Carole-Jean Wu, Xiaodong Wang, Kim Hazelwood, and David Brooks. 
"Exploiting Parallelism Opportunities with Deep Learning Frameworks." 
arXiv preprint arXiv:1908.04705 (2019).
```

```
@article{wang2019exploiting,
  title={Exploiting Parallelism Opportunities with Deep Learning Frameworks},
  author={Wang, Yu Emma and Wu, Carole-Jean and Wang, Xiaodong and Hazelwood, Kim and Brooks, David},
  journal={arXiv preprint arXiv:1908.04705},
  year={2019}
}
```

![tftuner](https://github.com/Emma926/mcbench/blob/master/TF-Tuner.png)
This repository also contains TensorFlow Tuner in tftuner/, that improves TensorFlow models' performance on CPU
by 1.34x on average.


This figure summarizes the structure of the paper, as well as this repository.
![mcbench](https://github.com/Emma926/mcbench/blob/master/overview.png)



## Workloads
library-math/ contains matrix multiplication microbenchmarks written in C++.  
library-threadpool/ contains thread pool microbenchmarks written in C++.  
operator/ contains matrix multiplication operator microbenchmarks written with TensorFlow and Caffe2.   
scheduler/ contains pre-trained models from Caffe2 model zoo.  


## Setup

library-math/, library-threadpool/, operator/, and scheduler/ have their own setup instructions, please refer to README.md in each directory to install corresponding libraries.  

After installing, library-math/ and library-threadpool/ have compile.sh to compile the workloads.  

To use the tools in utils/, please install Flamegraph by doing the following in your chosen directory
```
git clone https://github.com/brendangregg/FlameGraph

```
and update the `fg_path` in utils/flamegraph.py. And then do
```
source env.sh
```

## Scripts and Experiments
Each directory contains bash scripts for experiments. Please refer to README.me in each directory for details.

After experiments, output directories are generated, and one can use tools in utils to parse the results.

## Utils

`utils/flamegraph.py` parses the `perf record` results into text files and flamegraphs. To use it anywhere is convinient for you, just simply run
```
flamegraph.py $DIR
```
where $DIR is the output directory from perf record experiments. It then generates .perf-folded files for the stack trace and .svg flamegraphs.  

`utils/get_perfrecord.py` parses the outputs from `utils/flamegraph.py`, the .perf-folded files, and store the ouputs in json files in data/. To use it anywhere, just run
```
get_perfrecord.py $DIR
```

`utils/get_perf.py` parses the performance from `perf stat`, top-level analysis, bandwidth and run-time experiments and generates data in json files stored in data/. Note that main memory bandwidth experiment needs to have 'bw' in the directory; top-level analysis needs to have 'td', and cross-socket memory bandwidth experiment needs to have 'upi'. To use it anywhere, run
```
get_perf.py $DIR
```

Yu Emma Wang  
10/8/2019
