import subprocess
import os
from common import idemp_mkdir, write_json

def start_telemetry(script_dir, exp_name, output_dir, interval=15):
    '''
    Start recording the telemetry info.
    :returns: subprocess instance of the telemetry recorder
    '''
    if interval > 0:
        return subprocess.Popen(['python3', f'{script_dir}/run.py', 
                                f'--exp_name={exp_name}',
                                f'--interval={interval}',
                                f'--output_dir={output_dir}'], stdout=subprocess.PIPE)

def parse_cpu_stat(stat_dir, time_str):
    '''
    An Ad hoc data parsing for CPU telemetry data
    '''
    cpu_stat = { 'timestamp' : time_str }
    for (_, _, files) in os.walk(stat_dir):
        for fp in files:
            with open(os.path.join(stat_dir, fp), 'r') as file:
                cpu_stat.update({ fp : {} })
                update = lambda label: cpu_stat[fp][label].append
                for lst in filter(lambda x: x, map(lambda x: x.split(), file.read().split('\n'))):
                    ts, label, data = lst
                    if label not in cpu_stat[fp].keys():
                        cpu_stat[fp].update({ label : [] })
                    update(label)((ts, data))
    return cpu_stat

def parse_gpu_stat(stat_dir, time_str):
    '''
    An Ad hoc data parsing for GPU telemetry data
    '''
    gpu_stat = { 'timestamp' : time_str }
    for (_, _, files) in os.walk(stat_dir):
        for fp in files:
            with open(os.path.join(stat_dir, fp), 'r') as file:
                gpu_stat.update({ fp : {'data' : []} })
                update = gpu_stat[fp]['data'].append
                for line in filter(lambda x: x, map(lambda x: x.split(), file.read().split('\n'))):
                    # Some data do not have a unit(e.g. pstate, 
                    # which indicates current performance of GPU).
                    if len(line) == 3:
                        ts, data, unit = line
                    else:
                        ts, data, unit = line + [None]
                    if 'unit' not in gpu_stat[fp].keys():
                        gpu_stat[fp]['unit'] = unit
                    update((ts, data))
    return gpu_stat

def process_telemetry_statistics(info, exp_name, output_dir, time_str, cpu_stat_parser=parse_cpu_stat, gpu_stat_parser=parse_gpu_stat):
    '''
    Collect data of telemetry statistics and write to results directory
    Note: The "parsing" logic procedure written in this file is specialized to deal with
          telemetry collected at pipsqueak. They are not guaranteed to work on other platforms.
    '''
    telemetry_output_dir = info.subsys_telemetry_dir(exp_name)
    if not os.path.exists(telemetry_output_dir):
        idemp_mkdir(telemetry_output_dir)
    data_dir = os.path.join(output_dir, f'telemetry/{exp_name}')
    cpu_telemetry_dir = os.path.join(data_dir, 'cpu')
    gpu_telemetry_dir = os.path.join(data_dir, 'gpu')
    write_json(os.path.join(telemetry_output_dir, 'gpu'), f'gpu-{time_str}.json', gpu_stat_parser(gpu_telemetry_dir, time_str))
    write_json(os.path.join(telemetry_output_dir, 'cpu'), f'cpu-{time_str}.json', cpu_stat_parser(cpu_telemetry_dir, time_str))