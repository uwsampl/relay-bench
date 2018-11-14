import subprocess
import shlex
import itertools
# from visualize import visualize

MODELS = [
#    "mlp",
#    "dqn",
     "dcgan",
#    "resnet-18",
    # "densenet",
#    "mobilenet",
    ]

IRS = [
#    "relay",
    "nnvm",
    # "tf",
    ]

TARGETS = [
    ("arm_cpu", "rpi3b"),
    # "x86_cpu",
#    ("gpu", "1080ti"),
    # "fpga",
]

REPEAT = 100

OUT_FILE = "graph-data.csv"

def main():
    with open(OUT_FILE, "w") as outf:
        # print("ir, target, model, avg time (ms), std dev (ms)", file=outf)
        print("IR,Target,Model,Time", file=outf)

 #   server = subprocess.Popen(shlex.split("python3 -m tvm.exec.rpc_server --tracker 0.0.0.0:4242 --key foo --no-fork"))

#    tracker = subprocess.Popen(shlex.split("python3 -m tvm.exec.rpc_tracker --port 4242 --no-fork"))

    for ir, (target, key), model in itertools.product(IRS, TARGETS, MODELS):
        subprocess.run(["python3.6", "tvm_benchmark/benchmark.py",
                        "--ir", ir,
                        "--network", model,
                        "--target", target,
                        "--rpc-key", key,
                        "--repeat", str(REPEAT),
                        "--output", "file",
                        "--outfile", OUT_FILE,
			"--host", "fleet",
                        "--port", "9190",
                        ])

   # tracker.kill()
   # server.kill()

    # visualize("../pl4ml/pldi19/fig/graph")

if __name__ == "__main__":
    main()
