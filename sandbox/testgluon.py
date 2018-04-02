from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon

data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise

batch_size = 16
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)

net = gluon.nn.Dense(1, in_units=2)

print(net.weight)
print(net.bias)

net.collect_params()

type(net.collect_params())

net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

example_data = nd.array([[4,7]])
net(example_data)

print(net.weight.data())
print(net.bias.data())

net = gluon.nn.Dense(1)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

square_loss = gluon.loss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0001})

epochs = 20
loss_sequence = []
num_batches = num_examples / batch_size

for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
    loss_sequence.append(cumulative_loss)


# plot the convergence of the estimated loss function
#%matplotlib inline
#import IPython
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt

plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)

# Adding some bells and whistles to the plot
plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)

plt.show()

params = net.collect_params() # this returns a ParameterDict

print('The type of "params" is a ',type(params))

# A ParameterDict is a dictionary of Parameter class objects
# therefore, here is how we can read off the parameters from it.

for param in params.values():
    print(param.name,param.data())

