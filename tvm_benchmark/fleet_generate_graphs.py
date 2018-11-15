import subprocess
import shlex
import itertools
# from visualize import visualize

MODELS = [
#    "mlp",
    "dqn",
   # "dcgan",
    "resnet-18",
    # "densenet",
    "mobilenet",
    ]

IRS = [
    "relay",
    "nnvm",
    # "tf",
    ]

TARGETS = [
    ("arm_cpu", "rpi3b"),
    # "x86_cpu",
    ("gpu", "titanx"),
    # "fpga",
]

REPEAT = 10

OUT_FILE = "graph-data.csv"

def main():
    with open(OUT_FILE, "w") as outf:
        print("IR,Target,Model,Time,NNVM", file=outf)

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

    with open("tvm_benchmark/tf_results.csv", "r") as tf_res, open(OUT_FILE, "a") as outf:
        for line in tf_res:
            outf.write(line)

if __name__ == "__main__":
    main()
