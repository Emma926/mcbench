import time
import argparse
import caffe2

from caffe2.python import (
    dyndep,
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    workspace,
)

from caffe2.python.modeling import initializers

import os
import sys
import numpy as np


parser = argparse.ArgumentParser(description='FC Example')
parser.add_argument('--layer', type=int, default=8,
                    help='Number of FC layers')
parser.add_argument('--node', type=int, default=128,
                    help='Number of nodes per layer')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Training Batch Size')
parser.add_argument('--train_steps', type=int, default=100,
                    help='Number of training steps to run')
parser.add_argument('--mode', type=str, default='train',
                    help='Mode: train or inference.')
parser.add_argument('--profile', type=bool, default=False,
                    help='Do op profile or not.')
parser.add_argument('--warmup_steps', type=int, default=1,
                    help='Number of steps to warm up.')
parser.add_argument('--proto_type', type=str, default='',
                    help='')
parser.add_argument('--async_threads', type=int, default=0,
                    help='async_thread_pool_size')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='rms, sgd')
parser.add_argument('--intra_threads', type=int, default=0,
                    help='Number of max intra op parallel threads.')
args, _ = parser.parse_known_args()

layer = args.layer
node = args.node
batch_size = args.batch_size
input_size = node
output_size = node
train_steps = args.train_steps
mode = args.mode
warmup_steps = args.warmup_steps

os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

def GetInput():
    input_data = np.random.rand(batch_size, input_size).astype(np.float32)
    output_data = np.random.randint(output_size, size=(batch_size,)).astype(np.int32)
    return input_data, output_data


def GetModel(model):
    print('Using brew.fc.')
    sizes = [input_size]
    for _ in range(layer):
        sizes.append(node)
    sizes.append(output_size)

    net = brew.fc(model, "input", 'dense_in', dim_in=sizes[0], dim_out=sizes[1])
    for i in range(1, len(sizes) - 2):
        net = brew.fc(model, net, 'dense_{}'.format(i), dim_in=sizes[i], dim_out=sizes[i + 1])
    net = brew.fc(model, net, 'dense_out', dim_in=sizes[-2], dim_out=sizes[-1])
    softmax = brew.softmax(model, net, 'softmax')
    return softmax


def main():

    workspace.ResetWorkspace()

    if args.profile:
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=2',
        '--caffe2_net_async_names_to_trace=benchmark',
        '--caffe2_net_async_tracing_dumping_nth=2',
        '--caffe2_net_async_tracing_nth=2'])
    else:
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=2',
            #'--caffe2_cpu_numa_enabled=1',
            '--caffe2_net_async_thread_pool_size=' + str(args.async_threads)])

    model = model_helper.ModelHelper(name="FC")

    input_data, output_data = GetInput()
    workspace.FeedBlob("input", input_data)
    workspace.FeedBlob("input_T", input_data.T)
    workspace.FeedBlob("output", output_data)

    out = GetModel(model)

    if mode == 'train':
        xent = model.LabelCrossEntropy([out, "output"], 'xent')
        loss = model.AveragedLoss(xent, "loss")
        model.AddGradientOperators([loss])
        if args.optimizer == 'rms':
            optimizer.build_rms_prop(
                model,
                base_learning_rate=0.1,
                max_gradient_norm=None,
                allow_lr_injection=False
            )
        elif args.optimizer == 'sgd':
            optimizer.build_sgd(
                model,
                base_learning_rate=0.1,
                policy="step",
                stepsize=1,
                gamma=0.999,
            )

    #CAFFE2_NET_TYPE = types.ENUM(
    #            "simple", "dag", "async_dag", "singlethread_async", "async_scheduling"
    #            )
    if args.profile:
        model.Proto().type = 'prof_dag'
    if args.proto_type != '':
        model.net.Proto().type = args.proto_type

    model.net.Proto().num_workers = args.intra_threads


    #warmup_runs = iterations
    #main_runs = iterations
    #run_individual = True
    #stats = workspace.BenchmarkNet(model.name, warmup_runs, main_runs, run_individual)
    #print(stats)

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net, overwrite=True)

    workspace.RunNet(model.net.Proto().name, num_iter=warmup_steps)
    t1 = time.time()
    workspace.RunNet(model.net.Proto().name, num_iter=train_steps)
    t2 = time.time()
    total_time = t2 - t1
    example_per_sec = batch_size * train_steps / total_time
    global_step_per_sec = train_steps / total_time
    print("--------------------CAFFE2-------------------------")
    for arg in vars(args):
        print("***%s: %s" % (arg, getattr(args, arg)))
    if 'OMP_NUM_THREADS' in os.environ:
        print("***OMP_NUM_THREADS: ", os.environ['OMP_NUM_THREADS'])
    if 'MKL_NUM_THREADS' in os.environ:
        print("***MKL_NUM_THREADS: ", os.environ['MKL_NUM_THREADS'])
    print("***Total time: %s" % total_time)
    print("***Average time: %s" % (total_time/train_steps/(layer-1)))
    flops = batch_size * (node * node * (layer - 1) + node * input_size + node * output_size)
    if args.mode == 'train':
        # FWD 2x and BWD 4x
        flops *= 6 * train_steps
    else:
        flops *= 2 * train_steps
    print('***TFLOPS: {}'.format(flops / total_time / 1e12))
    print("---------------------------------------------")
    print("---------------------------------------------")


if __name__ == "__main__":
    main()
