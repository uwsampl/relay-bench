import subprocess
import shlex
import itertools

MODELS = [
    # "mlp",
    # "dqn",
    # "dcgan",
    "resnet-18",
    # "densenet",
    ]

IRS = [
    "relay",
    "nnvm",
    # "tf",
    ]

TARGETS = [
    # "arm_cpu",
    "x86_cpu",
    # "gpu",
    # "fpga",
]

REPEAT = 10

OUT_FILE = "graph-data.csv"

def main():
    with open(OUT_FILE, "w") as outf:
        print("ir, target, model, avg time (ms), std dev (ms)", file=outf)

    server = subprocess.Popen(shlex.split("python3 -m tvm.exec.rpc_server --tracker 0.0.0.0:9190 --key foo"))

    tracker = subprocess.Popen(shlex.split("python3 -m tvm.exec.rpc_tracker --port 9190"))

    for ir, target, model in itertools.product(IRS, TARGETS, MODELS):
        subprocess.run(["python3", "tvm_benchmark/benchmark.py",
                        "--ir", ir,
                        "--network", model,
                        "--target", target,
                        "--rpc-key", "foo",
                        "--repeat", str(REPEAT),
                        "--output", "file",
                        "--outfile", OUT_FILE])

    tracker.kill()
    server.kill()

if __name__ == "__main__":
    main()
