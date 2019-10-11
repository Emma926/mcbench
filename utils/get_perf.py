#!/usr/bin/python

import os
import json
import sys

from parse_utils import *

trace_flag = False
if len(sys.argv) > 2:
    trace_flag = True
    trace_keyword = sys.argv[2]

path = sys.argv[1]
files = get_file_names(os.listdir(path))
if trace_flag:
    l = [int(get_value(i, trace_keyword)) for i in files ]
    sorted_files = [x for _,x in sorted(zip(l,files))]
    files = sorted_files
        
label_all = []
perf_all = {}
scaled_perf_all = {}
out_perf_all = {}

for label in files:

    err_f = os.path.join(path, label + '.err')
    out_f = os.path.join(path, label + '.out')

    label_all.append(label)

    if os.path.isfile(out_f):
        out_perf = get_flops(out_f)
        if out_perf_all == {}:
            for k, v in iter(out_perf.items()):
                out_perf_all[k] = []
        for k, v in iter(out_perf.items()):
            out_perf_all[k].append(v)

    if os.path.isfile(err_f):

        if 'bw' in path:
            perf = get_memory_bandwidth(err_f)
        elif 'upi' in path:
            perf = get_upi_bandwidth(err_f)
        elif 'td' in path:
            if  'short' in path and 'single' in path:
                perf = get_topdown_short(err_f, single=True)
            elif 'short' in path:
                perf = get_topdown_short(err_f)
            else:
                perf = get_topdown(err_f)
        else:
            perf = get_perf_counter(err_f)
            
    
        if len(perf) > 1:
            if perf_all == {}:
                for k, v in perf.iteritems():
                    perf_all[k] = []
            for k, v in perf.iteritems():
                perf_all[k].append(v)
    
        if not 'bw' in path and not 'td' in path and not 'upi' in path:
            perf_scaled = scale_perf_counters(perf)
            if scaled_perf_all == {}:
                for k, v in perf_scaled.iteritems():
                    scaled_perf_all[k] = []
            for k, v in perf_scaled.iteritems():
                scaled_perf_all[k].append(v)


d = {
  'labels': label_all,
}

for k, v in iter(out_perf_all.items()):
    d[k] = v
for k, v in perf_all.iteritems():
    d[k] = v
for k, v in scaled_perf_all.iteritems():
    d[k] = v

if not check_d(d):
    exit()
#display_d(d, ['labels', 'GFLOPS'])
d = merge_results(d, 'try')
#display_d(d, ['labels', 'GFLOPS_m'])
if not check_d(d):
    exit()

name = path.strip('/')

if os.path.isfile('data/' + name + '.json'):
    with open('data/' + name + '.json', 'r') as infile:
        din = json.load(infile)
    print(len(d['labels']) - len(din['labels']), 'more data points than last time.')

if not os.path.isdir('data'):
    os.makedirs('data')

with open('data/' + name + '.json', 'w') as outfile:
    json.dump(d, outfile)
    print 'Results written in data/' + name + '.json'

print('Total data points: ', len(d['labels']))
print(d.keys())
