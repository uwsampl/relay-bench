"""Forward propagation of MobileNet on GPU. To get the best performance, run the file with:
    MXNET_ENGINE_TYPE=NaiveEngine MXNET_CPU_NNPACK_NTHREADS=4 python mxnet_mobilenet_forward.py"""
import time
import logging
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
from collections import namedtuple

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

f32 = 'float32'
no_bias = False

batch_size = 1
num_classes = 1000
image_shape = (3,224,224)

eps = 1e-10 + 1e-5
fix_gamma = False

def conv_block(data, name, num_filter, kernel=(3,3), stride=(1,1), pad=(1,1)):
    # convolution + bn + relu
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
        pad=pad, no_bias=no_bias, layout='NCHW', name=name + '_conv')
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=fix_gamma, eps=eps, name=name + '_bn')
    act = mx.sym.Activation(data=bn, act_type='relu', name=name + '_relu')
    return act

def separable_conv_block(data, name, num_depthwise_filter, num_pointwise_filter, multiplier=1, kernel=(3,3), downsample=False, pad=(1,1)):
    if downsample:
        stride = (2,2)
    else:
        stride = (1,1)
    # depthwise convolution + bn + relu
    conv1 = mx.sym.Convolution(data=data, num_filter=num_depthwise_filter, num_group=multiplier, kernel=kernel, stride=stride,
        pad=pad, no_bias=no_bias, layout='NCHW', name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=fix_gamma, eps=eps, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    # pointwise convolution + bn + relu
    conv2 = mx.sym.Convolution(data=act1, num_filter=num_pointwise_filter, kernel=(1,1), stride=(1,1),
        pad=(0,0), no_bias=no_bias, layout='NCHW', name=name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=fix_gamma, eps=eps, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    return act2

def mobile_net(num_classes=1000, alpha=1.0, is_shallow=False):
    data = mx.sym.Variable("data")
    body = conv_block(data, 'conv_block_1', int(32*alpha), stride=(2,2))
    body = separable_conv_block(body, 'separable_conv_block_1', int(32*alpha), int(64*alpha))
    body = separable_conv_block(body, 'separable_conv_block_2', int(64*alpha), int(128*alpha), downsample=True)
    body = separable_conv_block(body, 'separable_conv_block_3', int(128*alpha), int(128*alpha))
    body = separable_conv_block(body, 'separable_conv_block_4', int(128*alpha), int(256*alpha), downsample=True)
    body = separable_conv_block(body, 'separable_conv_block_5', int(256*alpha), int(256*alpha))
    body = separable_conv_block(body, 'separable_conv_block_6', int(256*alpha), int(512*alpha), downsample=True)
    if is_shallow:
        body = separable_conv_block(body, 'separable_conv_block_7', int(512*alpha), int(1024*alpha), downsample=True)
        body = separable_conv_block(body, 'separable_conv_block_8', int(1024*alpha), int(1024*alpha))
    else:
        for i in range(7, 12):
            body = separable_conv_block(body, 'separable_conv_block_%d' % i, int(512*alpha), int(512*alpha))
        body = separable_conv_block(body, 'separable_conv_block_12', int(512*alpha), int(1024*alpha), downsample=True)
        body = separable_conv_block(body, 'separable_conv_block_13', int(1024*alpha), int(1024*alpha))
    pool = mx.symbol.Pooling(data=body, name='pool', kernel=(7,7), pool_type='avg', global_pool=True)
    flatten = mx.symbol.Flatten(data=pool, name='flatten')
    fc = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes, no_bias=no_bias)
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax


sym = mobile_net(num_classes=num_classes, alpha=1.0, is_shallow=False)
data_shape = (batch_size,) + image_shape
label_shape = (batch_size,)
ishape = {'data': data_shape, 'softmax_label': label_shape}

ctx = mx.cpu()
# oshape = sym.infer_shape(**ishape)
npa = np.random.uniform(size=ishape['data'])
npb = np.random.uniform(size=ishape['softmax_label'])
nda = mx.nd.array(npa, ctx)
ndb = mx.nd.array(npb, ctx)

mod = mx.mod.Module(sym)
mod.bind(data_shapes=[('data', ishape['data'])],
         label_shapes=[('softmax_label', ishape['softmax_label'])],
         for_training=False)
mod.init_params()

Batch = namedtuple('Batch', ['data', 'label'])
batch = Batch([nda], [ndb])

import time
mx.nd.waitall()

for i in range(10):
    mod.forward(batch)
mx.nd.waitall()

num = 50
t0 = time.time()
for i in range(num):
    mod.forward(batch)
mx.nd.waitall()
t1 = time.time()
print('time: {} ms/iter'.format(round((t1 - t0) * 1000 / num, 3)))
