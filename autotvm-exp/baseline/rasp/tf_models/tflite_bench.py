import subprocess

import os
import sys
import csv

def gen_command(model_path, img_path):
    cmd = ['/home/pi/tensorflow/tensorflow/contrib/lite/examples/label_image/label_image',
    '--count', '20', '--tflite_model', model_path, '--image',
    '/home/pi/tensorflow/tensorflow/contrib/lite/examples/label_image/image.bmp']
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
        cmd = gen_command(os.path.join(f[0], f[1]), out_path)
        cmds.append(cmd)
    return cmds

def main():
    conv = list()
    depthwise = list()

    path = sys.argv[1]
    models = get_models(path)
    cmds = gen_commands(models)
    print(cmds)
    for cmd, model in zip(cmds, models):
        if 'layer' in model[1]:
            shape = os.path.splitext(model[1].split('layer')[1])[0].split('_')
            shape = [int(dim) for dim in shape]
            output = str(subprocess.check_output(cmd))
                
            time = output.split('average time:')[1].split('ms')[0].strip()
            shape.append('tflite')
            shape.append(time)
                
            if 'mobilenet' in model[1]:
                depthwise.append(shape)
            if 'resnet' in model[1]:
                conv.append(shape)
       
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

if __name__ == '__main__':
    main()
