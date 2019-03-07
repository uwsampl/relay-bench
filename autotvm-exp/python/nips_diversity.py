import argparse

from nips_common import *

methods = [
    'small_new#xgb-rank-d4', 'small_new#xgb-rank-d2', 'small_new#xgb-rank',
]

def show_name(name):
    trans_table = {
        'spatial_pack#xgb-rank': '$\lambda$=1',
        'spatial_pack#xgb-rank-d2': '$\lambda$=2',
        'spatial_pack#xgb-rank-d4': '$\lambda$=4',
        'small_new#xgb-rank': '$\lambda$=1',
        'small_new#xgb-rank-d2': '$\lambda$=2',
        'small_new#xgb-rank-d4': '$\lambda$=4',
    }

    return trans_table.get(name, name)

task_names = [
    'resnet.C1.B1.d', 'resnet.C2.B1.d', 'resnet.C3.B1.d',
    'resnet.C4.B1.d', 'resnet.C5.B1.d', 'resnet.C6.B1.d',
    'resnet.C7.B1', 'resnet.C8.B1', 'resnet.C9.B1',
    'resnet.C10.B1', 'resnet.C11.B1', 'resnet.C12.B1',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    if args.full:
        output = '../figures/diversity_full.pdf'
    else:
        output = '../figures/diversity.pdf'
        task_names = [task_names[0], task_names[1], task_names[2], task_names[11]]

    draw(task_names, methods, output, show_name, args)

