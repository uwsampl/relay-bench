import argparse

from nips_common import *

methods = [
    'small_new#xgb-rank', 'small_new#treernn-rank',
    'small_new#xgb-reg', 'small_new#treernn-reg',
]

def show_name(name):
    trans_table = {
        'small_new#xgb-rank': 'GBT Rank',
        'small_new#xgb-reg': 'GBT Regression',
        'small_new#treernn-rank': 'TreeGRU Rank',
        'small_new#treernn-reg': 'TreeGRU Regression',
    }

    return trans_table.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')

    args = parser.parse_args()

    if args.full:
        output = '../figures/rank_reg_full.pdf'
    else:
        output = '../figures/rank_reg.pdf'
        task_names = task_names[:4]

    draw(task_names, methods, output, show_name, args)

