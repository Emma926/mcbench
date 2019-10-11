# assume conda is installed
conda create --name caffe2
conda activate caffe2
conda install pytorch-nightly-cpu -c pytorch
conda install protobuf
conda install future
python -m caffe2.python.models.download -i inception_v1
python -m caffe2.python.models.download -i inception_v2
python -m caffe2.python.models.download -i mobilenet_v2
python -m caffe2.python.models.download -i squeezenet
python -m caffe2.python.models.download -i bvlc_reference_rcnn_ilsvrc13
python -m caffe2.python.models.download -i bvlc_googlenet
python -m caffe2.python.models.download -i bvlc_reference_caffenet
