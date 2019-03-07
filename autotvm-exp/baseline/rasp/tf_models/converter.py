import subprocess

import os
import sys

def gen_command(path, out_path):
    cmd = ['bazel', 'run', '--config=opt',
    '//tensorflow/contrib/lite/toco:toco',  '--',
    '--savedmodel_directory={0:s}'.format(path),
    '--output_file={0:s}'.format(out_path)]
    return cmd

def get_model_dirs(path):
    dirs = list()
    for dirpath, dirnames, filenames in os.walk(path):
        dirpath = os.path.abspath(dirpath)
        for dirname in dirnames:
            if 'layer' in dirname or 'model' in dirname:
                dirs.append((dirpath, dirname))
        break
    return dirs

def gen_commands(model_dirs):
    cmds = list()
    for model_dir in model_dirs:
        out_path = os.path.join(model_dir[0], model_dir[1] + '.tflite')
        cmd = gen_command(os.path.join(model_dir[0], model_dir[1]), out_path)
        cmds.append(cmd)
    return cmds

def main():
    path = sys.argv[1]
    workspace_path = sys.argv[2]
    cmds = gen_commands(get_model_dirs(path))
    print(cmds)
    os.chdir(workspace_path)
    for cmd in cmds:
        subprocess.call(cmd) 

if __name__ == '__main__':
    main()
