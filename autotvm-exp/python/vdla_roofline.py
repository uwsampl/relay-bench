import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

df = pd.read_csv('../data/fpga/vdla_latency_hiding.csv', index_col=0)

colors = ['white', 'black', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

fig, ax = plt.subplots()

plt.ticklabel_format(style='plain', axis='boths', scilimits=(0,0))


df_mt = df[(df.threaded == True) & (df.layer > 0)]
for name, group in df_mt.groupby('layer'):
  if name == 1:
    label = 'optimized'
  else:
    label = '_nolegend_'
  ax.loglog(group["arith-intensity"], group["skip-alu-gops"], marker='^', linestyle='', color=colors[name%len(colors)], label=label, basex=10)
  for x, y in zip(group["arith-intensity"], group["skip-alu-gops"]):
    ax.annotate("{0:.0f}".format(y), xy=(x/1.05, y*1.05), fontsize=8)

df_st = df[(df.threaded == False) & (df.layer > 0)]
for name, group in df_st.groupby('layer'):
  if name == 1:
    label = 'baseline'
  else:
    label = '_nolegend_'
  ax.loglog(group["arith-intensity"], group["skip-alu-gops"], marker='_', linestyle='', color=colors[name%len(colors)], label=label, basex=10)
  for x, y in zip(group["arith-intensity"], group["skip-alu-gops"]):
    ax.annotate("{0:.0f}".format(y), xy=(x/1.05, y/1.08), fontsize=8)

for name, group in df[df.layer > 0].groupby('layer'):
  ax.loglog(group["arith-intensity"], group["skip-alu-gops"], marker='', linestyle='--', color=colors[name%len(colors)], label="C{}".format(name), basex=10)

# Derive max throughput
num_pes = 256
clk_freq_MHz = 100.0
xput_Gops = num_pes * clk_freq_MHz * 2 / 1000
# Derive max bandwidth
acp_width_bits = 64
bw_GBps = acp_width_bits * clk_freq_MHz / 1000 / 8
# Derive roofline knee
knee = xput_Gops / bw_GBps 

plt.loglog([0, knee], [0, xput_Gops], 'k-', lw=2, color='tab:blue', basex=10)
plt.loglog([knee, 500], [xput_Gops, xput_Gops], 'k-', lw=2, color='tab:blue', basex=10)

ax.set_xlim(7, 500)
ax.set_ylim(6, 60)
ax.grid()

ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Arithmetic Intensity: Ops per Byte of Data (log scale)")
ax.set_ylabel("Giga Ops per Sec (log scale)")
ax.legend(prop={'size': 8})
ax.set_aspect(0.9)

ax.grid()

# plt.show()
fig.savefig('../figures/vdla.pdf', bbox_inches='tight')


