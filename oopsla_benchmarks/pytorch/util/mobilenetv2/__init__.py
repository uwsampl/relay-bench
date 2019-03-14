import os
from . import MobileNetV2

dirname = os.path.dirname(__file__)
PRETRAIN_PARAMS = os.path.join(dirname, 'mobilenet_v2.pth.tar')
