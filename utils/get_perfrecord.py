#!/usr/bin/python
import os
import json
import sys

from parse_utils import merge_results, get_value

def check_keyword(line):
    for k in sequence:
        v = keywords[k]
        for i in v:
            if i in line:
                return k
    return 'other'


def which_framework(f):
    os.system('grep tensorflow ' + f + ' > tmpfile')
    if os.path.getsize("tmpfile") > 0:
        os.system('rm tmpfile')
        return 'tensorflow'
    os.system('grep caffe2 ' + f + ' > tmpfile')
    if os.path.getsize("tmpfile") > 0:
        os.system('rm tmpfile')
        return 'caffe2'
    os.system('rm tmpfile')
    return 'none'
  
trace_flag = False
if len(sys.argv) > 2:
    trace_flag = True
    trace_keyword = sys.argv[2]

path = sys.argv[1]

tf_sequence = ['mkl-compute','mkl-data-prep', 'barrier', 'omp', 'threadpool', 'tf_py', 'profile']
tf_keywords = {
    'mkl-compute': ['avx512', 'mkldnn' , 'kernel', 'mkl_blas_avx512_sgemm_kernel', 'mkl_blas_avx2_sgemm_kernel'],
    'mkl-data-prep':['mkl_blas_avx512_sgemm_scopy', 'mkl_blas_avx2_sgemm_scopy'],
    'barrier': ['kmp_barrier', 'kmp_wait_yield', 'kmp_yield'],
    'omp': ['kmp','omp_driver', 'kmp_invoke_microtask', 'libgomp' , 'omp_parallel'],
    'threadpool':['Eigen', 'eigen'],
    'tf_py':['tensorflow', 'python','python3', 'pyeval_evalframeex', '_pytype_lookup', '__pyx'],
    'profile':['SYSCALL', 'perf_pmu', 'syscall'],
    }

c2_sequence = ['mkl-compute','mkl-data-prep', 'barrier', 'omp', 'threadpool', 'c2_math', 'c2', 'py', 'profile']
c2_keywords = {
    'mkl-compute': ['mkldnn','mkl_avx', 'mklml_intel','mkl_blas_avx512_sgemm_kernel', 'mkl_blas_avx2_sgemm_kernel'],
    'mkl-data-prep':['mkl_blas_avx512_sgemm_scopy', 'mkl_blas_avx2_sgemm_scopy'],
    'barrier': ['kmp_barrier', 'kmp_wait_yield', 'kmp_yield'],
    'omp': ['kmp', 'omp_driver', 'kmp_invoke_microtask', 'libgomp' , 'omp_parallel'],
    'threadpool':['Eigen', 'eigen'],
    'c2_math':['caffe2::math'],
    'c2':['caffe2'],
    'py':['python', 'pyeval_evalframeex', '_pytype_lookup', '__pyx'],
    'profile':['SYSCALL', 'perf_pmu', 'syscall']
    }

label_all = []
data = {}
out_perf_all = {}

files = []
for f in os.listdir(path):
    if '.perf-folded' not in f:
        continue
    if '.old' in f:
        continue
    files.append(f)

if trace_flag:
    l = [int(get_value(i.split('.')[0], trace_keyword)) for i in files ]
    sorted_files = [x for _,x in sorted(zip(l,files))]
    files = sorted_files


for f in files:
  fw = which_framework(os.path.join(path, f))
  if fw <> 'none':
    break
print 'framework:', fw

if fw == 'tensorflow':
    sequence = tf_sequence
    keywords = tf_keywords 
if fw == 'caffe2':
    sequence = c2_sequence
    keywords = c2_keywords 

for k in keywords.keys():
    data[k] = []
data['other'] = []

for f in files:
    label_all.append(f.replace('.perf-folded', ''))
    for k in data.keys():
        data[k].append(0)
        
    fin = open(os.path.join(path, f), 'r')
    for line in fin:
        line = line.lower()
        k = check_keyword(line)
        samples = int(line.strip(' \n').split(' ')[-1])
        data[k][-1] += samples
        if samples > 20 and k == 'other':
            print(line)

data['labels'] = label_all
d = data
d = merge_results(d, 'try')

name = path.strip('/')

if os.path.isfile('data/' + name + '.json'):
    with open('data/' + name + '.json', 'r') as infile:
        din = json.load(infile)
    print(len(d['labels']) - len(din['labels']), 'more data points than last time.')

if not os.path.isdir('data'):
    os.makedirs('data')

with open('data/' + name + '.json', 'w') as outfile:
    json.dump(d, outfile)
    print('Results written in data/' + name + '.json')

print('Total data points: ', len(d['labels']))
print(d.keys())
