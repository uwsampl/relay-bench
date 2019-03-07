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
            build_command += "cd ~/ComputeLibrary; cp ~/autotvm-exp/baseline/acl/* .; "

            if device.opencl:
                build_command += "make model USE_OPENCL=1; "
            else:
                build_command += "make model; "

            cmd = "ssh -t " + device.ssh_address + ' "' + build_command + '"' 

        run_cmd(cmd)

    
    cmds = []
    for device in devices:
        run_command = ""
        run_command += "cd ~/ComputeLibrary; rm -rf tmp.tsv; "
        run_command += "export LD_LIBRARY_PATH=~/ComputeLibrary/build; "
        run_command += "./model " + device.device_name + " " + args.backend + " " + str(args.n_times) + " " + str(device.num_threads) + "; "

        cmd = "ssh -t " + device.ssh_address + ' "' + run_command + '"; ' 
        cmd += "scp " + device.ssh_address + ":ComputeLibrary/tmp.tsv " + device.device_name + ".acl.tsv"
        cmds.append(cmd)

    pool = multiprocessing.Pool(len(cmds))
    pool.map(run_cmd, cmds)

