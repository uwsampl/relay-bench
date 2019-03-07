import argparse

from nips_common import *

task_names = [
    'resnet.C7.B1', 'resnet.C8.B1', 'resnet.C9.B1', 'other.DEN1'
    #'resnet.C7.B1', 'resnet.C8.B1', 'other.DEN1', 'resnet.C7.B1.cd'
]

methods = [
    'small_new#xgb-rank-tl',
    'small_new#treernn-rank-tl',

    # matmul
    'vanilla#xgb-rank-curve-tl',
    'vanilla#treernn-reg-tl',

    # cross device
    'spatial_pack#xgb-reg-curve-tl',
    'spatial_pack#treernn-reg-tl',
    'spatial_pack#xgb-reg',
    'spatial_pack#treernn-reg',

    # no transfer
    'small_new#xgb-rank',
    'small_new#treernn-rank',
    'vanilla#xgb-rank',
    'vanilla#treernn-rank',
]

ct = 0
def show_name(name):
    trans_table = {
        'small_new#xgb-rank-tl': 'GBT-Transfer',
        'small_new#xgb-rank-knob-tl': 'GBT-Transfer',
        'small_new#treernn-rank-tl': 'TreeGRU-Transfer',
        'small_new#treernn-reg-tl': 'TreeGRU-Transfer',
        'small_new#xgb-rank': 'GBT',
        'small_new#treernn-rank': 'TreeGRU',
        'DEN1': "C1$-$C6 -> Matmul-1024",
    }

    global ct
    if name == 'C7':
        ct += 1
        if ct == 1:
            return 'C1$-$C6 -> C7'
        elif ct == 2:
            return 'Mali GPU C1$-$C6 \n-> A53 CPU C7'

    if name == 'C8':
        return 'C1$-$C6 -> C8'

    if name == 'C9':
        return 'C1$-$C6 -> C9'

    return trans_table.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    output = '../figures/transfer.pdf'

    x_max = 150
    draw(task_names, methods, output, show_name, args,
         x_max=x_max, col=4, yerr_max=0.1)

