from collections import namedtuple
import multiprocessing
import os
import argparse
import random

from util import devices

def run_cmd(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    devices = list(filter(lambda x: 'cuda' in x.backends, devices))

    # NOTE(lmzheng): assume they are in a same network file system

    cmds = []
    for device in devices:
        filename = device.device_name + '.tmp.tsv'
        tmp_filename = "tmp_%0x.tsv" % random.getrandbits(32)

        run_command = ""
        run_command += "cd ~/autotvm-exp/baseline/cuda; "
        run_command += "rm -rf %s; " % filename

        ## core command ##
        run_command += ("CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/tvm/python:~/incubator-mxnet/python "
                       "python3 end2end_mx.py --out-file %s" % tmp_filename)

        cmd = 'ssh -t %s "%s"; ' % (device.ssh_address, run_command)

        # copy and delete tmp file
        cmd += "scp %s:autotvm-exp/baseline/cuda/%s cuda/%s; " % (device.ssh_address, tmp_filename, filename)
        cmd += 'ssh %s "rm autotvm-exp/baseline/cuda/%s" ;' % (device.ssh_address, tmp_filename)

        cmds.append(cmd)

    pool = multiprocessing.Pool(len(cmds))
    pool.map(run_cmd, cmds)

