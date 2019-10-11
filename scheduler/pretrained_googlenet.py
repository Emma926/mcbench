from caffe2.python import (
    core,
    model_helper,
    optimizer,
    workspace,
)
from caffe2.python.models import bvlc_googlenet as mynet
import numpy as np
import argparse
import time
import os

parser = argparse.ArgumentParser(description='Pre-trained model')
parser.add_argument('--proto_type', type=str, default='',
                    help='empty or async_scheduling')
parser.add_argument('--async_threads', type=int, default=0,
                    help='async_thread_pool_size')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch Size')
parser.add_argument('--steps', type=int, default=10,
                    help='Number of steps to measure.')
args, _ = parser.parse_known_args()

workspace.ResetWorkspace()
workspace.GlobalInit(['caffe2', '--caffe2_log_level=2',
            '--caffe2_net_async_thread_pool_size=' + str(args.async_threads)])

init_net = mynet.init_net
predict_net = mynet.predict_net
# you must name it something
predict_net.name = "googlenet_predict"

if args.proto_type != '':
    predict_net.type = 'async_scheduling'
    print('Using async scheduling.')
#predict_net.type = 'prof_dag'

img=np.ones((args.batch_size, 3, 224, 224)).astype(np.float32)
workspace.FeedBlob("data", img)

workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)

p = workspace.Predictor(init_net.SerializeToString(), predict_net.SerializeToString())
results = p.run({'data': img})


print(" =============== CAFFE2 ==================")
for arg in vars(args):
        print("***%s: %s" % (arg, getattr(args, arg)))
if 'OMP_NUM_THREADS' in os.environ:
        print("***OMP_NUM_THREADS: ", os.environ['OMP_NUM_THREADS'])
if 'MKL_NUM_THREADS' in os.environ:
        print("***MKL_NUM_THREADS: ", os.environ['MKL_NUM_THREADS'])

# run the net and return prediction
for step in range(args.steps):
    start = time.time()
    results = p.run({'data': img})
    total = time.time() - start
    print('  Average time: %s secs' % total)

# turn it into something we can play with and examine which is in a multi-dimensional array
results = np.asarray(results)
print("results shape: ", results.shape)

