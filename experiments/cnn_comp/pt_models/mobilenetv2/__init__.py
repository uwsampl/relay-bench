import os
from . import MobileNetV2

mobilenet_dir = os.path.dirname(__file__)
MOBILENET_PARAMS = os.path.join(mobilenet_dir, 'mobilenet_v2.pth.tar')
