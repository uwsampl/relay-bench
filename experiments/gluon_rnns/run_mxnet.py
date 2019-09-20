from validate_config import validate
from exp_templates import (common_trial_params, common_early_exit,
                           run_template)

import numpy as np
import mxnet as mx

from mxnet_util import import_gluon_rnn


def rnn_setup(dev, network):
    use_gpu = (dev == 'gpu')
    context = mx.gpu(0) if use_gpu else mx.cpu(0)

    net, num_states, shapes = import_gluon_rnn(network)
    net.initialize(ctx=context)
    net.hybridize()

    shape_list = [shapes['data']] + [shapes['state%s' % i] for i in range(num_states)]
    mx_inputs = [mx.nd.array(np.random.rand(*shape).astype('float32'), ctx=context)
                 for shape in shape_list]
    return [lambda: net(*mx_inputs)[0].asnumpy()]


def rnn_trial(thunk):
    thunk()


def rnn_teardown(thunk):
    pass


if __name__ == '__main__':
    run_template(validate_config=validate,
                 check_early_exit=common_early_exit({'frameworks': 'mxnet'}),
                 gen_trial_params=common_trial_params(
                     'mxnet', 'gluon_rnns',
                     rnn_trial, rnn_setup, rnn_teardown,
                     ['device', 'network'],
                     ['devices', 'networks']))
