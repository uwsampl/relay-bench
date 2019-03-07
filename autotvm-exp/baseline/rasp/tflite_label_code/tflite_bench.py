import subprocess
import time
import numpy as np

import os
import sys
import csv

#currently doing 20 * 20 runs (EXPERIMENTS * ITER)
EXPERIMENTS=10
SLEEP_TEMP=48
SLEEP=5
COUNT='100'

def gen_command(model_path, img_path, count):
    cmd = ['/home/pi/tensorflow/tensorflow/contrib/lite/examples/label_image/label_image',
    '--count', count, '--tflite_model', model_path]
    return cmd

def get_models(path):
    files = list()
    for dirpath, _, filenames in os.walk(path):
        dirpath = os.path.abspath(dirpath)
        for filename in filenames:
            if 'tflite' in os.path.splitext(filename)[1]:
                files.append((dirpath, filename))
        break
    return files

def gen_commands(files):
    cmds = list()
    for f in files:
        out_path = os.path.join(f[0], f[1] + '.tflite')
        if 'layer' in f[1]:
            count = COUNT
        elif 'model' in f[1]:
            count = COUNT
        cmd = gen_command(os.path.join(f[0], f[1]), out_path, count)
        cmds.append(cmd)
    return cmds

def main():
    conv = list()
    depthwise = list()
    end2end = list()

    path = sys.argv[1]
    models = get_models(path)
    cmds = gen_commands(models)
    print(cmds)
    for cmd, model in zip(cmds, models):
        ts = list()
        e = 0
        while e < EXPERIMENTS:
            temp = float(subprocess.check_output(['/opt/vc/bin/vcgencmd', 'measure_temp']).decode().split('=')[1].split("'")[0])
            while temp > SLEEP_TEMP:
                print("temp: {0:f}, sleeping...".format(temp, SLEEP))
                time.sleep(SLEEP)
                temp = float(subprocess.check_output(['/opt/vc/bin/vcgencmd', 'measure_temp']).decode().split('=')[1].split("'")[0])

            output = subprocess.check_output(cmd).decode()
            t = output.split('average time:')[1].split('ms')[0].strip()
            print(output)
            print(float(t))
            if float(t) < 0:
                print("got negative, retry...")
                continue
            ts.append(t)
            e += 1

        if 'layer' in model[1]:
            shape = os.path.splitext(model[1].split('layer')[1])[0].split('_')
            shape = [int(dim) for dim in shape]
            shape.append('tflite')
            shape += ts

            if 'mobilenet' in model[1]:
                depthwise.append(shape)
            if 'resnet' in model[1]:
                conv.append(shape)
                
        elif 'model' in model[1]:
            end2end.append([model[1]] + ts)

    depthwise = sorted(depthwise, reverse=True)
    conv = sorted(conv, reverse=True)
    print(depthwise)
    print(conv)
    with open('tflite_depthwise.csv', 'w') as depthwise_file:
        writer = csv.writer(depthwise_file)
        for row in depthwise:
            writer.writerow(row)
    with open('tflite_conv.csv', 'w') as conv_file:
        writer = csv.writer(conv_file)
        for row in conv:
            writer.writerow(row)
    with open('tflite_end2end.csv', 'w') as end2end_file:
        writer = csv.writer(end2end_file)
        for row in end2end:
            writer.writerow(row)

if __name__ == '__main__':
    main()
