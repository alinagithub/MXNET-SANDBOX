#testsoftmaxgluon.py

from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

data_ctx = mx.cpu()
model_ctx = mx.cpu()
# model_ctx = mx.gpu()

