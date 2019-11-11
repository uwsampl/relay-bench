import subprocess
import threading
import argparse
import datetime
import sys
import os


def parse_cpu_stat(info:list) -> list:
    # TODO: use sensors-detect to initialize the
    #       configuration for `sensors` command
    #       Note: need sudo
    pass

def parse_gpu_stat(info:list) -> list:
    try:
        assert info
    except:
        return []
    return info[-1].decode().split(', ')

def start_job(fp_dir, nvidia_fields, time_span) -> None:
    threading.Timer(time_span, start_job, args=[fp_dir, nvidia_fields, time_span]).start()
    nvidia_smi = subprocess.Popen(['nvidia-smi', '--format=csv', '--query-gpu={}'.format(','.join(nvidia_fields))], stdout=subprocess.PIPE)
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    parsed_data = parse_gpu_stat(nvidia_smi.stdout.readlines())
    try:
        assert len(parsed_data) == len(nvidia_fields)
        timestamp = parsed_data[0]
        for filename, data in zip(nvidia_fields[1:], parsed_data[1:]):
            with open(os.path.join(fp_dir, 'gpu', filename), 'a+') as fp:
                fp.write(f'{timestamp} {data}\n')
    except:
        print('Inconsistent length, passing')
        pass
    # cpu_info = subprocess.Popen(['sensors'], stdout=subprocess.PIPE)

def main(args):
    parser = argparse.ArgumentParser(description='Telemtry Process of Dashboard')
    parser.add_argument('--interval', type=int, nargs=1, required=False)
    parser.add_argument('--output_dir', type=str, nargs=1, required=True)
    parser.add_argument('--exp_name', type=str, nargs=1, required=True)
    arguments = parser.parse_args(args[1:])
    nvidia_fields = 'timestamp,clocks.gr,clocks.current.memory,utilization.gpu,utilization.memory,memory.used,pstate,power.limit,temperature.gpu,fan.speed'.split(',')
    out_dir = arguments.output_dir[0]
    # directory structure:
    # ./output_dir
    #       -> telemtry
    #           -> char_rnn
    #           -> treelstm ...
    # Possible locations: ~/dashboard-conf/dashboard/results/experiments
    out_dir = os.path.join(os.path.expanduser(out_dir), 'telemetry')
    log_dir = os.path.join(out_dir, arguments.exp_name[0])
    if os.path.exists(log_dir):
        os.system('rm -r {}'.format(log_dir))
    os.makedirs(os.path.join(log_dir, 'gpu'))
    os.makedirs(os.path.join(log_dir, 'cpu'))
    start_job(log_dir, nvidia_fields, arguments.interval[0] if arguments.interval else 30)

if __name__ == '__main__':
    main(sys.argv)