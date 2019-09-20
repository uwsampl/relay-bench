from validate_config import validate
from plot_util import PlotBuilder, PlotScale, PlotType, generate_longitudinal_comparisons
from exp_templates import visualize_template, generate_graphs_by_dev

MODEL_TO_TEXT = {
    'nature-dqn': 'Nature DQN',
    'vgg-16': 'VGG-16',
    'resnet-18': 'ResNet-18',
    'mobilenet': 'MobileNet'
}

def visualize(dev, raw_data, output_dir):
    # empty data: nothing to do
    if not raw_data.items():
        return

    # make model names presentable
    for (framework, models) in raw_data.items():
        # NOTE: need to convert the keys to a list, since we're mutating them
        # during traversal.
        for model in list(models.keys()):
            val = models[model]
            del models[model]
            models[MODEL_TO_TEXT[model]] = val

    data = {
        'raw': raw_data,
        'meta': ['Framework', 'Network', 'Mean Inference Time (ms)']
    }

    PlotBuilder().set_title('CNN Comparison on {}'.format(dev.upper())) \
                 .set_y_label(data['meta'][2]) \
                 .set_y_scale(PlotScale.LOG) \
                 .set_figure_height(3.0) \
                 .set_aspect_ratio(3.3) \
                 .set_sig_figs(3) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, 'cnns-{}.png'.format(dev))


if __name__ == '__main__':
    visualize_template(validate, generate_graphs_by_dev(visualize))
