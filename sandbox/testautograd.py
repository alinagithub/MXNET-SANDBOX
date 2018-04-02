#testautograd.py
import mxnet as mx
from mxnet import nd, autograd
mx.random.seed(1)


x = nd.array([[1, 2], [3, 4]])

x.attach_grad()

with autograd.record():
    y = x * 2
    z = y * x

z.backward()

print(x.grad)

