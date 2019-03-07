import shutil

fo = open("arxiv/TVMTechReport.tex", "w")
for line in open("TVMTechReport.tex"):
    pos = line.find("{figures/")
    if pos != -1:
        pos2 = line[pos:].find("}")
        assert pos2 != -1
        fig_name = line[pos:pos + pos2].split("/")[1]
        shutil.copy("figures/%s.pdf" % fig_name, "arxiv")
        line = line[:pos] + "{" + fig_name + line[pos + pos2:]
    fo.write(line)
fo.close()

shutil.copy("TensorOpt.bib", "arxiv")

print("Finish getting arxiv")

