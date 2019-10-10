import time
import argparse
import caffe2

from caffe2.python import (
    brew,
    core,
    model_helper,
    workspace,
)

from caffe2.python.modeling import initializers

import os
import sys
import numpy as np


parser = argparse.ArgumentParser(description='FC Operator')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--node', type=int, default=128)
parser.add_argument('--loop_count', type=int, default=300)
FLAGS, _ = parser.parse_known_args()

layer = 4
# for best CPU performance
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

def GetInput():
    input_data = np.random.rand(FLAGS.batch_size, FLAGS.node).astype(np.float32)
    return input_data

def GetModel(model):
    sizes = [FLAGS.node]
    for _ in range(layer):
        sizes.append(FLAGS.node)
    sizes.append(FLAGS.node)

    net = brew.fc(model, "input", 'dense_in', dim_in=sizes[0], dim_out=sizes[1])
    for i in range(1, len(sizes) - 2):
        net = brew.fc(model, net, 'dense_{}'.format(i), dim_in=sizes[i], dim_out=sizes[i + 1])
    net = brew.fc(model, net, 'dense_out', dim_in=sizes[-2], dim_out=sizes[-1])
    #softmax = brew.softmax(model, net, 'softmax')
    return net

def main():

    workspace.ResetWorkspace()

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2',
            '--caffe2_net_async_thread_pool_size=1'])

    model = model_helper.ModelHelper(name="FC")

    input_data = GetInput()
    workspace.FeedBlob("input", input_data)

    out = GetModel(model)

    model.net.Proto().type = "simple"

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net, overwrite=True)

    print(" =============== CAFFE2 ==================")
    print(" [" + str(FLAGS.batch_size) + "x" + str(FLAGS.node) + "] * [" + str(FLAGS.node) + "x" + str(FLAGS.node) + "]")
    for arg in vars(FLAGS):
        print("***%s: %s" % (arg, getattr(FLAGS, arg)))
    if 'OMP_NUM_THREADS' in os.environ:
        print("***OMP_NUM_THREADS: ", os.environ['OMP_NUM_THREADS'])
    if 'MKL_NUM_THREADS' in os.environ:
        print("***MKL_NUM_THREADS: ", os.environ['MKL_NUM_THREADS'])

    flops = 2 * FLAGS.batch_size * FLAGS.node * FLAGS.node
    workspace.RunNet(model.net.Proto().name, num_iter=1)
    for i in range(10):
        t1 = time.time()
        workspace.RunNet(model.net.Proto().name, num_iter=FLAGS.loop_count/10)
        t2 = time.time()
        total_time = t2 - t1
        avg_time = total_time / (FLAGS.loop_count / 10) / (layer + 1)
        print('----- During ' + str((FLAGS.loop_count/10)*i) + ' to ' + str((FLAGS.loop_count/10)*(i+1)) + ' steps -----')
        print(" Average time: %s secs" % avg_time)
        print(' GFlop/sec   : {}'.format(flops / avg_time / 1e9))


if __name__ == "__main__":
    main()
