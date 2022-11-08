set -ex
pip3 install -r requirements.txt
USE_CUDA=0 USE_CUDNN=0 BUILD_TEST=0 USE_NNPACK=0 USE_QNNPACK=0 BUILD_CAFFE2_OPS=0 BUILD_CAFFE2=0 python3 setup.py "${@:-develop}"
