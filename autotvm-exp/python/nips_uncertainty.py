import argparse

from nips_common import *

methods = [
    'small_new#xgb-reg-ei', 'small_new#xgb-reg-ucb', 'small_new#xgb-reg-mean'
]

def show_name(name):
    trans_table = {
        'small_new#xgb-reg-ei': 'Expected Improvement',
        'small_new#xgb-reg-ucb': 'Upper Confidence Bound',
        'small_new#xgb-reg-mean': 'Mean',
    }

    return trans_table.get(name, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    args = parser.parse_args()

    if args.full:
        output = '../figures/uncertainty_full.pdf'
    else:
        output = '../figures/uncertainty.pdf'
        task_names = task_names[:4]

    draw(task_names, methods, output, show_name, args)

