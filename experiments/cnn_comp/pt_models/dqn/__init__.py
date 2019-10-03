import os
from . import dqn

dqn_dir = os.path.dirname(__file__)
DQN_PARAMS = os.path.join(dqn_dir, 'asteroids.pth')
