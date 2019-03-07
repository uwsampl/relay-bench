# redraw all figures in NIPS paper

# component evaluation
python3 nips_cost_model.py --full --s
python3 nips_diversity.py --full --s
python3 nips_uncertainty.py --full --s
python3 nips_rank_reg.py --full --s

python3 nips_cost_model.py --s
python3 nips_diversity.py --s
python3 nips_uncertainty.py --s
python3 nips_rank_reg.py --s

# transfer
python3 nips_cross_shape.py --s
python3 nips_cross_operator.py --s 
python3 nips_wall_clock.py --s

# end2end
#python3 cuda_op_plot.py --s
#python3 cuda_end2end_plot_nips.py --s

