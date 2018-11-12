import cairosvg
import subprocess
import shlex
import argparse

def visualize(outdir):
    # vega-lite -> svg
    subprocess.run(shlex.split("vl2svg tvm_benchmark/graph.json graph.svg"))

    # svg -> pdf
    cairosvg.svg2pdf(url="graph.svg", write_to=outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, default="graph.pdf",
                        help="The output directory of the pdf.")
    args = parser.parse_args()

    visualize(args.o)
