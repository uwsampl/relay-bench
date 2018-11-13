import cairosvg
import subprocess
import shlex
import argparse
import os
from shutil import copyfile

def visualize(outdir="graph"):
    # prepare outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # vega-lite -> svg
    subprocess.run(shlex.split(f"vl2svg tvm_benchmark/graph.json {outdir}/graph.svg"))

    # svg -> pdf
    cairosvg.svg2pdf(url=f"{outdir}/graph.svg", write_to=f"{outdir}/graph.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="graph",
                        help="The output directory of the pdf.")
    args = parser.parse_args()

    visualize(args.outdir)
