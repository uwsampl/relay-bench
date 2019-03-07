from collections import namedtuple
import multiprocessing
import os
import argparse

from util import devices

def run_cmd(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action='store_true')
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--backend", type=str, default="all")
    parser.add_argument("--n-times", type=int, default=10)
    args = parser.parse_args()

    devices = list(filter(lambda x: args.target in x.device_name, devices))

    if args.build:
        for device in devices:
            build_command = ""
            build_command += "cd ~/autotvm-exp; git reset --hard; git checkout lmzheng; "
            build_command += "git reset --hard HEAD~10; git pull; "
            build_command += "cd ~/autotvm-exp/baseline/tflite; "
            build_command += "cp label_image.cc ~/tensorflow/tensorflow/contrib/lite/examples/label_image/; "
            build_command += "cd ~/tensorflow/tensorflow/contrib/lite/examples/label_image/; "
            build_command += "g++ -std=c++11 -O3 label_image.cc bitmap_helpers.cc -o label_image -I../../../../../ -I../../downloads/flatbuffers/include -Wl,--no-as-needed -Wl,--no-as-needed -ldl -pthread ../../gen/lib/rpi_armv7/libtensorflow-lite.a; "
            build_command += "cp label_image ~/autotvm-exp/baseline/tflite; "

            cmd = "ssh -t " + device.ssh_address + ' "' + build_command + '"' 

        run_cmd(cmd)

    cmds = []
    for device in devices:
        run_command = ""
        run_command += "cd ~/autotvm-exp/baseline/tflite; "
        run_command += "python3 bench.py --target " + device.device_name + "-cpu " + \
                        " --n-times " + str(args.n_times) + " --n-threads " + \
                        str(device.num_threads) + " ; "

        cmd = "ssh -t " + device.ssh_address + ' "' + run_command + '"; ' 
        cmd += "scp " + device.ssh_address + ":autotvm-exp/baseline/tflite/tmp.tsv " + device.device_name + ".tflite.tsv"

        cmds.append(cmd)

    pool = multiprocessing.Pool(len(cmds))
    pool.map(run_cmd, cmds)

