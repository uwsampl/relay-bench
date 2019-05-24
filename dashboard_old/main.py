import copy
import platform
import pprint
import sqlite3

import dank_net

experiments = [dank_net]

# NOTE: We should reuse as much of Steven's infra as possible.
# Maybe start by refactoring it so his script takes a dictionary of config,
# rather than directly pulling from `argparse`

# TODO: Use Facebook's AX for experiment tuning? https://www.ax.dev/
# TODO: Use vegalite for viz? https://vega.github.io/vega-lite/
# TODO: And this wrapper: https://altair-viz.github.io/
# TODO: Use python boomslang for graph generation

# TODO: Get an experiment that returns a constant result and displays it on a
# minimal dashboard.

# TODO: Zach's proposal:
#   - write cronjob on pipsqueak
#   - pull repo
#   - run steven's eval scripts



def run_experiment(experiment, config):
    config = copy.deepcopy(config)
    results = experiment.run(config)
    results['config'] = config
    return results


if __name__ == '__main__':
    machine_info = {
        'platform': platform.uname()._asdict()
    }
    # TODO(weberlo): Need per-experiment config
    config = {
        'frameworks': ['mxnet', 'darknet'],
        'num_layers': 10,
        'num_trials': 100,
    }
    results = {}
    for experiment in experiments:
        exp_name = experiment.get_name()
        assert exp_name not in results
        results[exp_name] = run_experiment(experiment, config)
    print(results)
    conn = sqlite3.connect('dashboard.db')

# purchases = [('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
#              ('2006-04-05', 'BUY', 'MSFT', 1000, 72.00),
#              ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
#             ]
# conn.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', purchases)
