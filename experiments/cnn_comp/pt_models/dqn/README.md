Modified from DQN in https://github.com/Kaixhin/Rainbow (model.py)

Values for parameters in util.py taken from this excerpt in main.py in Rainbow:

```
parser = argparse.ArgumentParser(description='Rainbow')
# ...
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
# ...
```