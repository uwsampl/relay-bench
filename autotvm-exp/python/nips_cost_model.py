import argparse

from nips_common import *

methods = [
    'small_new#xgb-rank', 'small_new#treernn-rank',
    'small_new#ga', 'small_new#ga*3',
    'small_new#random', 'small_new#random*3',
]

def show_name(name):
    trans_table = {
        'small_new#xgb-rank': 'GBT',
        'small_new#xgb-rank': 'GBT',
        'small_new#treernn-rank': 'TreeGRU',
        'small_new#random': 'Random',
        'small_new#random*3': 'Random X 3',
        'small_new#ga': 'GA',
        'small_new#ga*3': 'GA X 3',
    }

    return trans_table.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    if args.full:
        output = '../figures/cost_model_full.pdf'
    else:
        output = '../figures/cost_model.pdf'
        task_names = task_names[:4]

    draw(task_names, methods, output, show_name, args, col=4)

