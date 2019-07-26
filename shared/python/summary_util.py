"""
Code for writing simple summaries of analyzed data.
"""
from common import (write_status, write_summary, sort_data, render_exception)

def summary_by_dev_and_network(data, devs, networks):
    if not devs:
        return ''
    ret = 'Format: ({})\n'.format(', '.join(networks))
    for dev in devs:
        ret += '_Times on {}_\n'.format(dev.upper())
        for (setting, times) in data[dev].items():
            ret += '{}: '.format(setting)
            ret += ', '.join(['{:.3f}'.format(time*1e3)
                              for (_, time) in times.items()])
            ret += '\n'
    return ret

def summary_by_dev(data, devs):
    if not devs:
        return ''
    data_by_dev = {dev: data[dev] for dev in devs}
    ret = ''
    for dev in devs:
        ret += '_Times on {}_\n'.format(dev.upper())
        for (setting, time) in data_by_dev[dev].items():
            ret += '{}: {:.3f}\n'.format(setting, time*1e3)
    return ret

def write_generic_summary(data_dir, output_dir, title, devices, networks=None, use_networks=False):
    """
    Given a data directory and output directory, this function writes
    a generic summary assuming that the data has a field keyed by device
    (cpu/gpu) and optionally by network. It writes a summary and status to the output dir.
    """
    try:
        all_data = sort_data(data_dir)
        most_recent = all_data[-1]

        summary = None
        if use_networks:
            summary = summary_by_dev_and_network(most_recent, devices, networks)
        else:
            summary = summary_by_dev(most_recent, devices)
        write_summary(output_dir, title, summary)
        write_status(output_dir, True, 'success')

        # TODO do something about comparisons to previous days
    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
