from validate_config import validate
from common import (invoke_main, write_status, write_summary,
                    sort_data, render_exception)


def render_summary(devs, models, datasets, data):
    ret = 'Format: (avg. epoch time, final accuracy)\n'
    for dev in devs:
        ret += '_Metrics on {}_\n'.format(dev.upper())
        for listing, metrics in data[dev].items():
            for model in models:
                for dataset in datasets:
                    ret += '{} {} on {}: '.format(listing, model, dataset)
                    avg_time = metrics[dataset][model]['time']['mean']
                    avg_final_acc = metrics[dataset][model]['acc']['mean']
                    ret += '({:.3f}, {:.3f})'.format(avg_time*1e3, avg_final_acc)
                    ret += '\n'
    return ret


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    try:
        all_data = sort_data(data_dir)
        most_recent = all_data[-1]
        title = config['title']
        devs = config['devices']
        datasets = config['datasets']
        models = config['models']
        summary = render_summary(devs, models, datasets, most_recent)
        write_summary(output_dir, title, summary)
        write_status(output_dir, True, 'success')
    except Exception as e:
        write_status(output_dir, False,
                     'Exception encountered:\n' + render_exception(e))
        return 1


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
