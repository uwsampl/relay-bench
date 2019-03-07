import os
import time

t = time.localtime()
outfile = "baseline-%d-%d-%d-%02d:%02d.tsv" % (
        t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)

with open(outfile, "w") as fout:
    fout.write("\t".join(["target", "backend", "workload_type", "workload",
                          "tuner", "template", "value", "time", "stamp",]) + "\n")

for dirpath, dirnames, filenames in os.walk("."):
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.tsv' and 'baseline' not in filename:
            cmd = "cat %s >> %s " % (os.path.join(dirpath, filename), outfile)
            print(cmd)
            os.system(cmd)

