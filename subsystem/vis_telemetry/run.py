import os
import math
import datetime
from common import invoke_main, sort_data, idemp_mkdir, write_status, process_gpu_telemetry, process_cpu_telemetry
from dashboard_info import DashboardInfo
from plot_util import PlotBuilder, UnitType, PlotType


def extract_float(s:str) -> float:
    s = ''.join(list(filter(lambda x: x.isdigit() or x in ('+', '-', '.', 'e'), s)))
    try:
        return float(s)
    except:
        return math.nan

def generate_graph(timestamp:str, title:str, cate_name:str,
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

def main(config_dir, home_dir, output_dir):
    info = DashboardInfo(home_dir)
    idemp_mkdir(output_dir)
    for exp_name in info.all_present_experiments():
        exp_conf = info.read_exp_config(exp_name)
        if exp_conf['active']:
            telemetry_folder = info.exp_telemetry_dir(exp_name)
            if os.path.exists(telemetry_folder):
                exp_graph_folder = os.path.join(telemetry_folder, 'graph')
                cpu_stat = os.path.join(telemetry_folder, 'cpu')
                gpu_stat = os.path.join(telemetry_folder, 'gpu')
                cpu_data = sort_data(cpu_stat)
                gpu_data = sort_data(gpu_stat)
                graph_folder = info.exp_graph_dir(exp_name)
                website_include_dir = os.path.join(graph_folder)
                try:
                    if cpu_data:
                        latest = process_cpu_telemetry(cpu_data[-1])
                        ts, *data = latest
                        current_ts_dir = os.path.join(exp_graph_folder, ts)
                        cpu_graph_dir = os.path.join(current_ts_dir, 'cpu')
                        copy_to = os.path.join(website_include_dir, 'cpu_telemetry')
                        idemp_mkdir(cpu_graph_dir)
                        idemp_mkdir(copy_to)
                        print(f'Visualizing CPU telemetry for {exp_name}')
                        for adapter, title, unit, data in data:
                            generate_graph(ts, f'{adapter}-{title}', title, data, cpu_graph_dir, copy_to=[copy_to])
                    
                    if gpu_data:
                        latest = process_gpu_telemetry(gpu_data[-1])
                        ts, *unpack = latest
                        current_ts_dir = os.path.join(exp_graph_folder, ts)
                        gpu_graph_dir = os.path.join(current_ts_dir, 'gpu')
                        copy_to = os.path.join(website_include_dir, 'gpu_telemetry')
                        idemp_mkdir(gpu_graph_dir)
                        idemp_mkdir(copy_to)
                        print(f'Visualizing GPU telemetry for {exp_name}')
                        for _, title, unit, data in unpack:
                            generate_graph(ts, title, title, data, gpu_graph_dir, y_label=unit if unit else '', copy_to=[copy_to])
                except Exception as e:
                    write_status(output_dir, False, f'Encountered err while generating graphs: {e}')
                    return
                write_status(output_dir, True, 'success')
            else:
                write_status(output_dir, False, 'No telemetry data found')

if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')