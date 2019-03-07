'''
MobileNetV2, implemented in Gluon.

Reference:
Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation
https://arxiv.org/abs/1801.04381
'''
import mxnet as mx
from mxnet.gluon.model_zoo.vision.mobilenet import MobileNetV2

def get_symbol(num_classes=1000, multiplier=1.0, ctx=mx.cpu(), **kwargs):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for  Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    num_classes : int, default 1000
        Number of classes for the output layer.
    multiplier : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    ctx : Context, default CPU
        The context in which to initialize the model weights.
    """
    net = MobileNetV2(multiplier=multiplier, classes=num_classes)
    net.initialize(ctx=ctx, init=mx.init.Xavier())
    net.hybridize()

    data = mx.sym.var('data')
    out = net(data)
    sym = mx.sym.SoftmaxOutput(out, name='softmax')
    return sym
