import os
import subprocess

filename="rnn-data.csv"
if os.path.exists(filename):
    os.remove(filename)

with open(filename, "w") as f:
    f.write(f'MODEL,TARGET,HIDDEN_SIZE,TIME\n')

for x in [128, 256, 512, 1024, 2048]:
    subprocess.check_call(["python", "bench_rnn_one.py", str(x)])
