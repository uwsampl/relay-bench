import time
import json
import argparse
import subprocess

import os
import sys
import csv

def gen_commands(models):
    cmds = []
    for m in models:
        cmd = [
            './label_image',
            '-c', str(args.n_times),
            '-m', m,
            '-i', 'image.bmp',
        ]
        if args.n_threads != 0:
            cmd += ['-t', str(args.n_threads)]
        cmds.append(cmd)
    return cmds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-times", type=int, default=10)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--target", type=str, default='rpi3b-cpu')
    args = parser.parse_args()

    log_file = open("tmp.tsv", "w")

    models = list(filter(lambda x: ".tflite" in x and x.index('.tflite') > 0, os.listdir(".")))
    cmds = gen_commands(models)
    for cmd, model in zip(cmds, models):
        costs = []
        for i in range(3):
            output = str(subprocess.check_output(cmd, stderr=subprocess.STDOUT))
            cost = float(output.split('average time:')[1].split('ms')[0].strip()) / 1000.0
            costs.append(cost)
            time.sleep(10)

        log_line = "\t".join([str(x) for x in [
            args.target, 'eigen', 'network', model.split('.')[0], 'tflite', 'default',
            json.dumps({"cost": costs}), time.time()
        ]])
        print(log_line)
        log_file.write(log_line + "\n") 

