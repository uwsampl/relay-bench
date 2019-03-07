import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model

def get_symbol(num_classes=1000, num_layers=121, ctx=mx.cpu(), **kwargs):
    """

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
    net = get_model('densenet%d' % num_layers, pretrained=True)
    net.initialize(ctx=ctx, init=mx.init.Xavier())
    net.hybridize()

    data = mx.sym.var('data')
    out = net(data)
    sym = mx.sym.SoftmaxOutput(out, name='softmax')
    return sym

