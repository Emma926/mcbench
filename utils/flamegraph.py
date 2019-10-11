#!/usr/bin/python
import sys
import os

path = sys.argv[1]
files = os.listdir(path)

# Modify this to Flamegraph Path
fg_path = 'FlameGraph'

for f in files:
    if not '.par' in f:
        continue
    perf_file = os.path.join(path, f)
    if os.path.isfile(perf_file.replace('.par', '.svg')):
        continue
    os.system('perf script -i ' + perf_file + '| ' + os.path.join(fg_path, 'stackcollapse-perf.pl') + ' > ' + perf_file.replace('.par', '.perf-folded'))
    os.system(os.path.join(fg_path, 'flamegraph.pl') + ' ' + perf_file.replace('.par', '.perf-folded') + ' > ' + perf_file.replace('.par', '.svg'))

