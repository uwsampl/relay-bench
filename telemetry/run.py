import subprocess
import threading
import argparse
import datetime
import sys
import os
from common import idemp_mkdir, invoke_main


def parse_cpu_stat(info:list) -> dict:
    '''
    Takes data returned by sensors, which contains temperature / voltage data.
    Returns a dict, whose keys are (group) name of the sensors and values are pairs of (label x data).
    '''
    current_group = 'ungrouped'
    result = dict({ current_group : [] })
    for line in info:
        # omit empty lines and `N/A` data
        if line and not 'N/A' in line:
            # Adapter names does not contain spaces
            if ' ' not in line:
                current_group = line
                result.update({ line : [] })
            else:
                label, data = ' '.join(line.split()).split()[:2]
                result[current_group].append((label.replace(':', ''), ''.join(data)))
    # return with empty entries removed
    return dict(filter(lambda x: len(x[1]), result.items()))

def parse_gpu_stat(info:list) -> list:
    '''
    Takes data returned by `nvidia-smi`, which contains a string with two lines, returns
    the last line, which is the data we need to monitor during an experiment.
    '''
    filtered = list(filter(lambda x: x, info))
    # ensure the list is not empty
    assert filtered
    return filtered[-1].split(', ')

def start_job(fp_dir, nvidia_fields, time_span, time_run) -> None:
    '''
    A chornological job, runs every `time_span` seconds.
    Fetches data from `nvidia-smi` and `sensors`, then parse it
    and write to corresponding files. Data will not be processed
    during monitoring in order to minimize the influence to the running experiments.

    Note: The process will be halted by `dashboard.py` when an experiment ends. 
    '''
    threading.Timer(time_span, start_job, args=[fp_dir, nvidia_fields, time_span, time_run + 1]).start()
    nvidia_smi = subprocess.run(['nvidia-smi', '--format=csv', '--query-gpu={}'.format(','.join(nvidia_fields))],
                            stdout=subprocess.PIPE, timeout=10)
    parsed_data = parse_gpu_stat(nvidia_smi.stdout.decode().split('\n'))
    # The length of lists of data should be the same
    # since we are using `nvidia_fields` to call the command
    assert len(parsed_data) == len(nvidia_fields)
    # timestamp = parsed_data[0]
    time_after = time_span * time_run
    for filename, data in zip(nvidia_fields[1:], parsed_data[1:]):
        with open(os.path.join(fp_dir, 'gpu', filename), 'a+') as fp:
            # fp.write(f'{timestamp[:-4]} {data}\n')
            fp.write(f'{time_after} {data}\n')

    sensors = subprocess.run(['sensors'], stdout=subprocess.PIPE)
    cpu_stat = parse_cpu_stat(sensors.stdout.decode().split('\n'))
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for (fname, entries) in cpu_stat.items():
        if entries[1:]:
            with open(os.path.join(fp_dir, 'cpu', fname), 'a+') as fp:
                for (label, data) in entries[1:]:
                    # fp.write(f'{timestamp} {label} {data}\n')
                    fp.write(f'{time_after} {label} {data}\n')

def main(interval, output_dir, exp_name):
    '''
        # directory structure:
        # ./output_dir
        #       -> telemtry
        #           -> char_rnn
        #           -> treelstm ...
    '''
    out_dir = os.path.join(output_dir, 'telemetry')
    log_dir = os.path.join(out_dir, exp_name)
    idemp_mkdir(os.path.join(log_dir, 'cpu'))
    idemp_mkdir(os.path.join(log_dir, 'gpu'))
    nvidia_fields = 'timestamp,clocks.gr,clocks.current.memory,utilization.gpu,utilization.memory,memory.used,pstate,power.limit,temperature.gpu,fan.speed'.split(',')
    start_job(log_dir, nvidia_fields, int(interval), 0)

if __name__ == '__main__':
    # main(sys.argv)
    invoke_main(main, 'interval', 'output_dir', 'exp_name')