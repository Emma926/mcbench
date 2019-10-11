import re
import numpy as np


def get_file_names(files):
    names = set() 
    for f in files:
        names.add(f.split('.')[0])
    return sorted(list(names))


def get_perc(s):
    return float(s.strip('(%)')) / 100


def find_keyword(line, l):
    for i in l:
        if i in line:
            return i
    return None


def get_topdown_short(filename, single=False):
    fin = open(filename, 'r')
    results = {}
    l = [[],[],[],[]]
    for line in fin:
        line = line.strip(' \n')
        floats = re.findall('\d+\.\d+', line, re.M|re.I)
        if len(floats) == 4:
            for i in range(4):
                l[i].append(float(floats[i]))
    if single:
        ind = l[0].index(max(l[0]))
        results['retiring'] = l[0][ind]
        results['bad-speculation'] = l[1][ind]
        results['frontend-bound'] = l[2][ind]
        results['backend-bound'] = l[3][ind]
    else:
        results['retiring'] = np.mean(l[0])
        results['bad-speculation'] = np.mean(l[1])
        results['frontend-bound'] = np.mean(l[2])
        results['backend-bound'] = np.mean(l[3])
    return results
      

def get_topdown(filename):
    fin = open(filename, 'r')
    l2 = {'l2-Memory_Bound':[], 'l2-Core_Bound':[]}
    l3 = {'l3-L1':[], 'l3-L2':[], 'l3-L3':[], 'l3-DRAM':[]}

    for line in fin:
        if not 'Backend_Bound' in line:
            continue
        line = line.strip(' \n')
        floats = re.findall('\d+\.\d+', line, re.M|re.I)
        if not len(floats) == 2:
            continue
        line_split = " ".join(line.split()).split(" ")
        kw = line_split[2].strip(":")
        if len(kw.split(".")) == 2:
            if not 'l2-' + kw.split(".")[1] in l2:
                print("Error: ", kw.split(".")[1])
                return
            l2['l2-' + kw.split(".")[1]].append(float(floats[0]))
        elif len(kw.split(".")) == 3:
            if not 'Memory_Bound' in line:
                continue
            core = line_split[0].split('-')[0]
            if 'L1' in line:
                l3['l3-L1'].append(float(floats[0]))
            elif 'L2' in line:
                l3['l3-L2'].append(float(floats[0]))
            elif 'L3' in line:
                l3['l3-L3'].append(float(floats[0]))
            elif 'DRAM' in line:
                l3['l3-DRAM'].append(float(floats[0]))
            else:
                print('Unknown:', line)
            
    results = {}
    for k, v in l2.iteritems():
        if v == []:
          results[k] = 0
        else:
          results[k] = np.mean(v)
    for k, v in l3.iteritems():
        if v == []:
          results[k] = 0
        else:
          results[k] = np.mean(v)

    return results


def get_perf_counter(filename):
    fin = open(filename, 'r')
    results = {}
    keywords = [
        'instructions',
        'cycles',
        'branch-misses',
        'dTLB-load-misses',
        'dTLB-store-misses',
        'L1-dcache-load-misses',
        'L1-dcache-loads',
        'l2_rqsts.miss',
        'l2_rqsts.references',
        'LLC-load-misses',
        'LLC-store-misses',
        'LLC-loads',
        'LLC-stores',
        'mem-loads',
        'mem-stores',
        'energy-cores',
        'energy-pkg',
        'energy-ram',
        '128b_packed_double',
        '128b_packed_single',
        '256b_packed_double',
        '256b_packed_single',
        '512b_packed_double',
        '512b_packed_single',
        'scalar_double',
        'scalar_single',
        'fp_assist.any']

    start = 0
    for line in fin:
        if 'Performance counter stats' in line:
            start = 1
        if start == 0:
            continue
        line = line.strip('\n').replace(',', '')
        floats = re.findall('\d+\.\d+', line, re.M|re.I)
        ints = re.findall('\d+', line, re.M|re.I)
        percs = re.findall('(\d+\.\d+%)', line, re.M|re.I)
        kw = find_keyword(line, keywords)
        if kw is not None:
            if 'energy' in kw:
                if len(floats) > 0:
                    results[kw] = float(floats[0])
                else:
                    results[kw] = 0
            else:
                if len(ints) > 0:
                    results[kw] = int(ints[0])
                    if int(ints[0]) in [0,1,2,3] and int(ints[1]) in [0,1,2,3]:
                        results[kw] = int(ints[3])
                      
                else:
                    results[kw] = 0
        if 'seconds' in line:
            results['perf_seconds'] = float(floats[0])
    fin.close()
    return results


def get_memory_bandwidth(filename):
    fin = open(filename, 'r')
    
    reads = []
    writes = []
    times = []
    for line in fin:
        line = line.strip('\n').replace(',', '')
        floats = re.findall('\d+\.\d+', line, re.M|re.I)
        if 'data_reads' in line:
            times.append(float(floats[0]))
            reads.append(float(floats[1]))
        elif 'data_writes' in line:
            writes.append(float(floats[1]))

    diff = [times[i+1] - times[i] for i in range(len(times)-1)]
    diff = np.mean(diff)

    results = {}
    results['reads_gbyte_ps'] = np.mean(reads) / 1e3 / diff
    results['writes_gbyte_ps'] = np.mean(writes) / 1e3 / diff
    results['total_gbyte_ps'] = np.mean(np.add(reads, writes)) / 1e3 / diff
    return results


def get_upi_bandwidth(filename):
    fin = open(filename, 'r')
    
    reads = []
    writes = []
    times = []

    for line in fin:
        line = line.strip('\n').replace(',', '')
        floats = re.findall('\d+\.\d+', line, re.M|re.I)
        if 'uncore_imc_0/cas_count_read/' in line:
            times.append(float(floats[0]))
            reads.append(0)
            writes.append(0)
        if 'cas_count_read' in line:
            reads[-1] += float(floats[1])
        elif 'cas_count_write' in line:
            writes[-1] += float(floats[1])

    diff = [times[i+1] - times[i] for i in range(len(times)-1)]
    diff = np.mean(diff)

    results = {}
    results['upi_reads_gbyte_ps'] = np.mean(reads) / 1e3 / diff
    results['upi_writes_gbyte_ps'] = np.mean(writes) / 1e3 / diff
    results['upi_total_gbyte_ps'] = np.mean(np.add(reads, writes)) / 1e3 / diff
    return results


def scale_perf_counters(perf):
    results = {}
    if 'instructions' in perf and 'cycles' in perf:
        results['ipc'] = perf['instructions'] * 1.0 / perf['cycles']
    if 'branch-misses' in perf:
        results['branch_mpki'] = perf['branch-misses'] * 1000.0 / perf['instructions']
    
    if 'dTLB-load-misses' in perf:
        results['dTLB_mpki'] = (perf['dTLB-load-misses'] + perf['dTLB-store-misses']) \
                        * 1000.0 / perf['instructions']
    if 'L1-dcache-load-misses' in perf:
        results['L1_mpki'] = perf['L1-dcache-load-misses'] * 1000.0 / perf['instructions']
        if 'L1-dcache-loads' in perf:
            results['L1_miss_rate'] = perf['L1-dcache-load-misses'] * 1.0 / perf['L1-dcache-loads']
            results['L1_miss_ps'] = perf['L1-dcache-load-misses'] * 1.0 / perf['perf_seconds']
    if 'l2_rqsts.miss' in perf:
        results['L2_mpki'] = perf['l2_rqsts.miss'] * 1000.0 / perf['instructions']
        if 'l2_rqsts.references' in perf:
            results['L2_miss_rate'] = perf['l2_rqsts.miss'] * 1.0 / perf['l2_rqsts.references']
    if 'LLC-load-misses' in perf:
        results['LLC_mpki'] = (perf['LLC-load-misses'] + perf['LLC-store-misses']) \
                        * 1000.0 / perf['instructions']
        results['LLC_load_mpki'] = perf['LLC-load-misses'] * 1000.0 / perf['instructions']
        results['LLC_load_miss_gbyte_ps'] = perf['LLC-load-misses'] * 64.0 / 1e9 / perf['perf_seconds']
        if 'LLC-loads' in perf:
            results['LLC_miss_rate'] = (perf['LLC-load-misses'] + perf['LLC-store-misses']) * 1.0 \
                                      / (perf['LLC-loads'] + perf['LLC-stores'])
    #if 'mem-loads' in perf:
    #    results['gbyte_ps'] = (perf['mem-loads'] + perf['mem-stores'])*64 / perf['perf_seconds'] / 1e9
    #    results['load_gbyte_ps'] = perf['mem-loads']*64 / perf['perf_seconds'] / 1e9

    results['FLOPS'] = 0
    if '128b_packed_double' in perf:
      results['FLOPS'] += perf['128b_packed_double'] * 2
    if '128b_packed_single' in perf:
      results['FLOPS'] +=  perf['128b_packed_single'] * 4
    if '256b_packed_double' in perf:
      results['FLOPS'] += perf['256b_packed_double'] * 4
    if '256b_packed_single' in perf:
      results['FLOPS'] += perf['256b_packed_single'] * 8
    if '512b_packed_double' in perf:
      results['FLOPS'] += perf['512b_packed_double'] * 8
    if '512b_packed_single' in perf:
       results['FLOPS'] += perf['512b_packed_single'] * 16
    if 'scalar_double' in perf:
      results['FLOPS'] += perf['scalar_double']
    if 'scalar_single' in perf:
      results['FLOPS'] += perf['scalar_single']
    if 'fp_assist.any' in perf:
      results['FLOPS'] +=  perf['fp_assist.any']

    if not results['FLOPS'] == 0:
      results['FLOPS'] /= perf['perf_seconds']

    if 'L1-dcache-loads' in perf:
      results['L1_gbyte_ps'] = perf['L1-dcache-loads'] * 64 / 1e9 / perf['perf_seconds']
    if 'l2_rqsts.references' in perf:
      results['L2_gbyte_ps'] = perf['l2_rqsts.references'] * 64 / 1e9 / perf['perf_seconds']
    if 'LLC-loads' in perf:
      results['LLC_gbyte_ps'] = (perf['LLC-loads'] + perf['LLC-stores']) * 64 / 1e9 / perf['perf_seconds']
      results['LLC_read_gbyte_ps'] = perf['LLC-loads'] * 64 / 1e9 / perf['perf_seconds']

    if 'L1-dcache-loads' in perf and 'FLOPS' in results:
      results['flops_per_L1byte'] = results['FLOPS'] / (perf['L1-dcache-loads'] * 64 / perf['perf_seconds'])
    if 'l2_rqsts.references' in perf and 'FLOPS' in results:
      results['flops_per_L2byte'] = results['FLOPS'] / (perf['l2_rqsts.references'] * 64 / perf['perf_seconds'])
    if 'LLC-loads' in perf and 'FLOPS' in results:
      results['flops_per_LLCbyte'] = results['FLOPS'] / ((perf['LLC-loads'] + perf['LLC-stores'])* 64 / perf['perf_seconds'])

    return results


def get_flops(filename):
    fin = open(filename, 'r')
    flops = []
    latency = []
    for line in reversed(list(fin)):

        floats = re.findall('\d+\.\d+', line, re.M|re.I)
        floatse = re.findall("(\d+(\.\d+)?)[Ee](\+|-)(\d+)", line, re.M|re.I)
        if floats and 'GFlop' in line or 'GFLOPS' in line:
            flops.append(float(floats[0]))
        if 'time' in line:
            if floatse:
              latency.append(float(line.strip(' \n').split(' ')[-2]))
            else:
              latency.append(float(floats[0]))
    results = {}
    if not len(flops) == 0:
      if len(flops) > 3:
        del flops[0]
        del flops[-1]
      results['GFLOPS'] = np.mean(flops)
    if not len(latency) == 0:
      if len(latency) > 3:
        del latency[0]
        del latency[-1]
      results['latency'] = np.mean(latency)
    fin.close()
    return results


def get_performance(filename):
    fin = open(filename, 'r')
    results = {}
    count = 0
    for line in reversed(list(fin)):
        count += 1
        if count >= 50:
            break
        line = line.strip(' \n')
        floats = re.findall('\d+\.\d+', line, re.M|re.I)
        ints = re.findall('\d+', line, re.M|re.I)
        if floats and 'examples/sec' in line:
            results['example_per_sec'] = float(floats[0])
        if floats and 'global_step/sec' in line:
            results['step_per_sec'] = float(floats[0])
        if floats and 'Total time' in line:
            results['exe_time'] = float(floats[0])
        if ints and 'train_steps' in line:
            results['train_steps'] = int(ints[0])
    fin.close()
    return results


def op_caffe2(filename):
    results = {'other': 0}
    fin = open(filename, 'r')

    # For two strings AB and A, A should be after AB.
    ops = ['FCGradient', 'FC', 'MomentumSGDUpdate', 'Softmax', 'RmsProp']
    for k in ops:
        results[k] = 0
    for line in fin:
        if 'ms/run' not in line:
            continue
        line = line.strip(' \n')
        kw = find_keyword(line, ops)
        ints = re.findall('\d+', line, re.M|re.I)
        if kw is not None:
            results[kw] = float(line.split(' ')[0]) * float(ints[-1])
        if kw is None:
            results['other'] += float(line.split(' ')[0]) * float(ints[-1])
    fin.close()
    return results


def op_pytorch(filename):
    results = {}
    fin = open(filename, 'r')
    ops = ['AddmmBackward', 'addmm', 'mm']
    train_steps = 0
    for line in reversed(list(fin)):
        if train_steps == 0 and 'train_steps' in line:
            train_steps = int(line.strip(' \n').split(' ')[-1])
        if 'us' not in line:
            continue
        line = line.strip(' \n')
        kw = find_keyword(line, ops)
        floats = re.findall('\d+.\d+', line, re.M|re.I)
        if kw is not None and float(floats[0]) < 500:
            break
        if kw is not None:
            if kw not in results:
                results[kw] = 0
            results[kw] += float(floats[0]) / (train_steps * 1000.0)
    fin.close()
    return results


def get_value(l, kw):
    l_split = l.split('-')
    for i in l_split:
        if i.split('_')[0] == kw:
            return i.split('_')[1]


def rm_keyword(l, kw):
    ori_k = kw + '_' + get_value(l, kw)
    if '-' + ori_k in l:
      new_l = l.replace('-' + ori_k, '')
    elif ori_k + '-' in l:
      new_l = l.replace(ori_k + '-', '')
    return new_l


def merge_results(d, kw='try'):
    new_d = {}
    new_d['labels'] = []
    temp_d = {}

    if not kw in d['labels'][0]:
        return d

    keys = []
    for k in d.keys():
        if not k == 'labels':
            new_d[k + '_m'] = []
            new_d[k + '_std'] = []
            temp_d[k] = []
            keys.append(k)
    for l_ind, l in enumerate(d['labels']):
        new_l = rm_keyword(l, kw) 
        if not new_l in new_d['labels']:
            new_d['labels'].append(new_l)
            for k in keys:
                temp_d[k].append([])
            for k in keys:
                temp_d[k][-1].append(d[k][l_ind])
        else:
            ind = new_d['labels'].index(new_l)
            for k in keys:
                temp_d[k][ind].append(d[k][l_ind])
        #print('===============================')
        #print(l, d['FLOPS'][l_ind])
        #for ind, l in enumerate(new_d['labels']):
        #    print(l, temp_d['FLOPS'][ind])
    for ind, l in enumerate(new_d['labels']):
        for k in keys:
            new_d[k + '_m'].append(np.mean(temp_d[k][ind]))
            new_d[k + '_std'].append(np.std(temp_d[k][ind]))
    return new_d


def display_d(d, keys=['labels']):
    s = ''
    for k in keys:
        s += k + '\t'
    s += '\n'
    for ind, l in enumerate(d['labels']):
        for k in keys:
            s += str(d[k][ind]) + '\t'
        s += '\n'
    print(s)


def check_d(d):
    s = set()
    for k, v in iter(d.items()):
      s.add(len(v))
    if len(s) > 1:
        print("The data is not aligned.")
        for k, v in iter(d.items()):
            print(len(v), k)
        return False
    return True

