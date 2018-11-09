import subprocess
import shlex
import itertools

MODELS = [
    "mlp",
    "dqn",
    # "resnet-18",
    ]

IRS = ["relay", "nnvm"]

REPEAT = 100

OUT_FILE = "graph-data.csv"

def main():
    with open(OUT_FILE, "w") as outf:
        print("ir, model, avg time (ms), std dev (ms)", file=outf)

    for ir, model in itertools.product(IRS, MODELS):
        subprocess.call(["python3", "tvm_benchmark/x86_cpu_imagenet_bench.py",
                        "--ir", ir,
                        "--network", model,
                        "--rpc-key", "foo",
                        "--repeat", str(REPEAT),
                        "--output", "file",
                        "--outfile", OUT_FILE])

if __name__ == "__main__":
    main()
