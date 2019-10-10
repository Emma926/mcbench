import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
import time
import numpy as np
import os

tf.flags.DEFINE_integer("loop_count", 300, "Total number of steps.")
tf.flags.DEFINE_integer("batch_size", 128, "")
tf.flags.DEFINE_integer("node", 128, "")
tf.flags.DEFINE_integer("intra_threads", 1, "For best performance, set it to be the number of physical cores per socket")
tf.flags.DEFINE_integer("inter_threads", 1, "For best performance, set to be the number of sockets.")
tf.flags.DEFINE_integer("device_count", 1, "For best performance, set it to be the number of cores per socket.")

FLAGS = tf.flags.FLAGS

layer = 8
# for best CPU performance
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

def model_fn(features):
  net = features

  for i in range(layer):
    net = tf.layers.dense(
      inputs=net,
      units=FLAGS.node,
      name='fc_' + str(i),
      activation=tf.nn.relu)

  net = tf.layers.dense(
    inputs=net,
    units=FLAGS.node,
    name='fc_' + str(layer),
    activation=None)

  return net

def main(unused_argv):
  del unused_argv

  tf.logging.set_verbosity(tf.logging.INFO)
  print('Tensorflow version: ' + str(tf.__version__))
  if FLAGS.loop_count < 10:
      print('loop_count needs to be over 10 and a multiply of 10.')
      exit()

  config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.intra_threads, \
              inter_op_parallelism_threads=FLAGS.inter_threads, \
              log_device_placement=False, allow_soft_placement=True, \
              device_count = {'CPU': FLAGS.device_count})

  X = tf.placeholder("float", [FLAGS.batch_size, FLAGS.node])
  logits = model_fn(X)
  init = tf.global_variables_initializer()
  init_l = tf.local_variables_initializer()

  sess = tf.Session(config=config)
  sess.run(init)
  sess.run(init_l)
  batch_x = np.random.random_sample((FLAGS.batch_size, FLAGS.node)).astype(np.float32)
  res = sess.run([logits], feed_dict={X: batch_x})    

  print(" =============== TensorFlow ==================")
  for k,v in iter(tf.app.flags.FLAGS.flag_values_dict().items()):
    print("***%s: %s" % (k, v))
  if 'OMP_NUM_THREADS' in os.environ:
    print("***OMP_NUM_THREADS: ", os.environ['OMP_NUM_THREADS'])
  if 'MKL_NUM_THREADS' in os.environ:
    print("***MKL_NUM_THREADS: ", os.environ['MKL_NUM_THREADS'])
  if 'KMP_BLOCKTIME' in os.environ:
    print("***KMP_BLOCKTIME: ", os.environ['KMP_BLOCKTIME'])

  flops = 2 * FLAGS.batch_size * FLAGS.node * FLAGS.node
  start = time.time()
  for cnt in range(1, FLAGS.loop_count+1):
      res = sess.run([logits], feed_dict={X: batch_x})    
      if cnt % (FLAGS.loop_count/10) == 0:
          total_time = time.time() - start
          avg_time = total_time / (FLAGS.loop_count/10) / (layer + 1)
          print('----- During ' + str(cnt - (FLAGS.loop_count/10) + 1) + ' to ' + str(cnt) + ' steps -----')
          print('  Average time: %s secs' % avg_time)
          print('  GFlop/sec   : {}'.format(flops / avg_time / 1e9))
          start = time.time()


if __name__ == "__main__":
  tf.app.run()
