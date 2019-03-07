import argparse

from nips_common import *

task_names = [
    'resnet.C7.B1.i',
    'resnet.C7.B1',
    'other.DEN1',
    'resnet.C7.B1.cd',
]

methods = [
    'small_new#xgb-rank-noir',
    'small_new#xgb-rank-curve',

    'small_new#xgb-rank-knob-tl',
    'small_new#xgb-rank-tl',
    'small_new#xgb-rank-curve-tl',

    # matmul
    'vanilla#xgb-rank-tl',
    'vanilla#xgb-rank-knob-tl',
    'vanilla#xgb-rank-curve-tl',

    # cross device
    'spatial_pack#xgb-reg',
    'spatial_pack#xgb-reg-curve-tl',

    # no transfer
    'small_new#xgb-rank',
    'small_new#xgb-rank.i',
    'vanilla#xgb-rank',
]

def method2color(method):
    trans_table = {
        # transfer
        'small_new#xgb-rank':          'C5',
        'small_new#xgb-rank.i':        'C1',
        'small_new#xgb-rank-tl':       'C1',

        'small_new#xgb-rank-curve-tl': 'C2',
        'small_new#xgb-rank-curve':    'C2',

        'small_new#xgb-rank-noir':     'C0',
        'small_new#xgb-rank-knob-tl':  'C0',

        # matmul
        'vanilla#xgb-rank-knob-tl':    'C0',
        'vanilla#xgb-rank-tl':         'C1',
        'vanilla#xgb-rank-curve-tl':   'C2',

        # cross device
        'spatial_pack#xgb-reg':           'C5',
        'spatial_pack#xgb-reg-curve-tl':  'C2',

        # no transfer
        'vanilla#xgb-rank':            'C5',
    }

    return trans_table[method]

ct = -1
def show_name(name):
    trans_table = {
        'small_new#xgb-rank-tl': 'GBT on Flatten Loop Context $x$',
        'small_new#xgb-rank-curve-tl': 'GBT on Context Relation $R$',
        'small_new#xgb-rank-knob-tl': 'GBT on Configuration $S$',
        'small_new#xgb-rank': 'GBT No Transfer',

        'small_new#xgb-rank-noir': 'GBT on Configuration $S$',

        'small_new#treernn-rank-tl': 'TreeGRU Transfer',
        'small_new#treernn-rank': 'TreeGRU No Transfer',

        'DEN1': "TITANX\nC1$-$C6 -> Matmul-1024",
    }

    global ct 

    if name == 'C7':
        ct += 1
        if ct == 0:
            return 'TITANX\nC7 in domain'
        elif ct == 1:
            return 'TITANX\nC1$-$C6 -> C7'
        elif ct == 2:
            return 'Mali GPU C1$-$C6\n -> A53 CPU C7'

    return trans_table.get(name, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    output = '../figures/invariant_feature.pdf'

    x_max = 150
    draw(task_names, methods, output, show_name, args,
         x_max=x_max, col=len(task_names), yerr_max=0.1, method2color=method2color, offset=(1.2, 1.98),
         add_cap=True)

