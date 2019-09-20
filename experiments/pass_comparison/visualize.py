from collections import OrderedDict
from validate_config import validate
from plot_util import (PlotBuilder, PlotScale, PlotType,
                       UnitType, generate_longitudinal_comparisons)
from exp_templates import visualize_template, generate_graphs_by_dev

MODEL_TO_TEXT = {
    'nature-dqn': 'Nature DQN',
    'vgg-16': 'VGG-16',
    'resnet-18': 'ResNet-18',
    'mobilenet': 'MobileNet'
}

def visualize(dev, raw_data, output_dir):
    if not raw_data.items():
        return

    # make model names presentable
    for (pass_name, models) in raw_data.items():
        # NOTE: need to convert the keys to a list, since we're mutating them
        # during traversal.
        for model in list(models.keys()):
            val = models[model]
            del models[model]
            models[MODEL_TO_TEXT[model]] = val

    data = {
        'raw': OrderedDict(sorted(raw_data.items())),
        'meta': ['Pass Combo', 'Network', 'Mean Inference Time (ms)']
    }

    PlotBuilder().set_title('Individual Passes Applied on {}'.format(dev.upper())) \
                 .set_y_label(data['meta'][2]) \
                 .set_y_scale(PlotScale.LOG) \
                 .set_figure_height(3) \
                 .set_aspect_ratio(3.3) \
                 .set_unit_type(UnitType.SECONDS) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, 'pass-comparison-{}.png'.format(dev))


if __name__ == '__main__':
    visualize_template(validate, generate_graphs_by_dev(visualize))
