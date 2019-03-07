# draw figures with TVM insterad of anonymous name

python3 cuda_end2end_plot_nips.py --s --tvm
python3 mali_end2end_plot_nips.py --s --tvm
python3 rasp_end2end_plot_nips.py --s --tvm

python3 cuda_op_plot_nips.py      --s --tvm
python3 mali_op_plot_nips.py      --s --tvm
python3 rasp_op_plot_nips.py      --s --tvm

python3 nips_wall_clock.py        --s --tvm

