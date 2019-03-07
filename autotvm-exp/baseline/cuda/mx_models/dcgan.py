"""
Reference:

Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional generative adversarial networks."
arXiv preprint arXiv:1511.06434 (2015).
"""

import mxnet as mx
BatchNorm = mx.sym.BatchNorm
eps = 1e-5 + 1e-12

def deconv2d(data, ishape, oshape, kshape, name, stride=(2, 2)):
    """a deconv layer that enlarges the feature map"""
    target_shape = (oshape[-2], oshape[-1])

    pad_y = (kshape[0] - 1) // 2
    pad_x = (kshape[1] - 1) // 2
    adj_y = (target_shape[0] + 2 * pad_y - kshape[0]) % stride[0]
    adj_x = (target_shape[1] + 2 * pad_x - kshape[1]) % stride[1]

    net = mx.sym.Deconvolution(data,
                               kernel=kshape,
                               stride=stride,
                               #target_shape=target_shape,
                               pad=(pad_y, pad_x),
                               adj=(adj_y, adj_x),
                               num_filter=oshape[0],
                               name=name)
    return net

def deconv2d_bn_relu(data, prefix , **kwargs):
    net = deconv2d(data, name="%s_deconv" % prefix, **kwargs)
    net = BatchNorm(net, fix_gamma=True, eps=eps, name="%s_bn" % prefix)
    net = mx.sym.Activation(net, name="%s_act" % prefix, act_type='relu')
    return net

def get_symbol(oshape, ngf=128, code=None):
    assert oshape[-1] == 32
    assert oshape[-2] == 32

    code = mx.sym.Variable("data") if code is None else code
    net = mx.sym.FullyConnected(code, name="g1", num_hidden=4*4*ngf*4)
    net = mx.sym.Activation(net, name="gact1", act_type="relu")
    # 4 x 4
    net = mx.sym.Reshape(net, shape=(-1, ngf * 4, 4, 4))
    # 8 x 8
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 4, 4, 4), oshape=(ngf * 2, 8, 8), kshape=(4, 4), prefix="g2")
    # 16x16
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 2, 8, 8), oshape=(ngf, 16, 16), kshape=(4, 4), prefix="g3")
    # 32x32
    net = deconv2d(
        net, ishape=(ngf, 16, 16), oshape=oshape[-3:], kshape=(4, 4), name="g4_deconv")
    net = mx.sym.Activation(net, act_type='tanh', name='final_act')
    return net

