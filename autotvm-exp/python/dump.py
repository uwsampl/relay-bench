import json

import redis

from util import log_value_old

out_file = "../data/autotuning/tuning_curve.log"

r = redis.StrictRedis('fleet', port=6379, db=1)

WANTED_JOB = 'resnet-18.C3.B1'
WANTED_TUNER = 'xgb'
WANTED_DEVICE = 'titanx'

def filter_func(data):
    job_name = data.get('job_name', "")
    return job_name == WANTED_JOB and data['tuning_options']['tuner'] == WANTED_TUNER and get_device_name(data) == WANTED_DEVICE

def trans_device_name(x):
    trans_table = {
        'titanx': 'GeForce GTX TITAN X',
    }
    return trans_table.get(x, x)

def get_device_name(x):
    if 'key' in data:
        return data['key']
    else:
        return data['device']

curves = []
for key in r.scan_iter():
    data = json.loads(r.get(key).decode())

    if filter_func(data):
        seconds = [x[0] for x in data['results']]
        seconds = [float("inf") if x < 0 else x for x in seconds]
        keep = float("inf")
        for i in range(len(seconds)):
            keep = min(keep, seconds[i])
            seconds[i] = keep

        curves.append(seconds)


log_value_old('cuda', trans_device_name(WANTED_DEVICE), WANTED_JOB.replace('resnet-18', 'resnet'), 'small_new#' + WANTED_TUNER, str(curves).replace('inf', 'float("inf")'), out_file)

