import os
from . import dcgan

dcgan_dir = os.path.dirname(__file__)
DCGAN_PARAMS = os.path.join(dcgan_dir, 'weights/netG_epoch_199.pth')
