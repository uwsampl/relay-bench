import os
import math
import datetime
from common import invoke_main, sort_data, idemp_mkdir, write_status, process_gpu_telemetry, process_cpu_telemetry, validate_json
from check_prerequisites import check_prerequisites
from dashboard_info import DashboardInfo
from plot_util import PlotBuilder, UnitType, PlotType


def extract_float(s:str) -> float:
    if s.lower() in ('alarm', 'n/a'):
        return math.nan
    data = ''.join(list(filter(lambda x: x.isdigit() or x in ('+', '-', '.', 'e'), s)))
    return float(data)

def generate_graph(timestamp:str, title:str, cate_name:str, \
                   dataset:list, save_path, y_label='', copy_to=[], unit=None):
    plt_builder = PlotBuilder()
    prepared_data = {
        'raw' : {'x' : list(map(lambda x: int(x[0]), dataset)),
                        # list(map(lambda x: datetime.datetime \
                        #                    .strptime(x[0], '%H:%M:%S')
                        #                   .strftime('%M:%S'), dataset)), 
                 'y' : list(filter(lambda x: x is not math.nan, 
                               map(lambda x: extract_float(x[1]), dataset)))},
        'meta' : ['Time', y_label]
    }
    if prepared_data['raw']['y'] and prepared_data['raw']['x']:
        plt_builder.set_title(title)\
                    .set_x_label(prepared_data['meta'][0])\
                    .set_y_label(prepared_data['meta'][1])\
                    .set_unit_type(unit if unit else UnitType.COMPARATIVE)\
                    .make(PlotType.LONGITUDINAL, prepared_data)\
                    .save(save_path, f'{cate_name}-{timestamp}.png')
        for each in copy_to:
            img_path = os.path.join(save_path, f'{cate_name}-{timestamp}.png')
            # overwrite image of last run
            os.system(f'cp {img_path} {each}/{cate_name}.png')

def visualize(device, data, exp_graph_dir, website_copy_dir, msg='',
                get_title=lambda *arg: '-'.join(arg)):
    ts, *data = data
    current_ts_dir = os.path.join(exp_graph_dir, ts)
    graph_dir = os.path.join(current_ts_dir, device)
    idemp_mkdir(graph_dir)
    idemp_mkdir(website_copy_dir)
    print(msg)
    for adapter, title, unit, data in data:
        generate_graph(ts, get_title(adapter, title, unit, data), title, data, graph_dir, 
                        y_label=unit if unit else '', copy_to=[website_copy_dir])


def main(config_dir, home_dir, output_dir):
    info = DashboardInfo(home_dir)
    idemp_mkdir(output_dir)
    for exp_name in info.all_present_experiments():
        exp_status = info.exp_status_dir(exp_name)
        run_status = validate_json(exp_status, 'run_cpu_telemetry', 'run_gpu_telemetry', filename='run.json')
        if check_prerequisites(info, { exp_name : {}}) == (True, 'success') and run_status.get('success', False):
            telemetry_folder = info.subsys_telemetry_dir(exp_name)
            if os.path.exists(telemetry_folder):
                exp_graph_folder = os.path.join(telemetry_folder, 'graph')
                cpu_stat = info.exp_cpu_telemetry(exp_name)
                gpu_stat = info.exp_gpu_telemetry(exp_name)
                cpu_data = sort_data(cpu_stat)
                gpu_data = sort_data(gpu_stat)
                graph_folder = info.exp_graph_dir(exp_name)
                website_include_dir = os.path.join(graph_folder)
                try:
                    if cpu_data and run_status.get('run_cpu_telemetry', False):
                        visualize('cpu', process_cpu_telemetry(cpu_data[-1]), 
                                  exp_graph_folder, 
                                  os.path.join(website_include_dir, 'cpu_telemetry'),
                                  f'Visualizing CPU telemetry for {exp_name}',
                                  lambda adapter, title, *rest: f'{adapter}-{title}')
                    
                    if gpu_data and run_status.get('run_gpu_telemetry', False):
                        visualize('gpu', process_gpu_telemetry(gpu_data[-1]), 
                                  exp_graph_folder, 
                                  os.path.join(website_include_dir, 'gpu_telemetry'),
                                  f'Visualizing GPU telemetry for {exp_name}',
                                  lambda _, title, *rest: title)
                except Exception as e:
                    write_status(output_dir, False, f'Encountered err while generating graphs: {e}')
                    return
                write_status(output_dir, True, 'success')
            else:
                write_status(output_dir, False, 'No telemetry data found')
                return

if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')