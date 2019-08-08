from mxnet import gluon

def import_gluon_rnn(name):
    if name not in ['rnn', 'gru', 'lstm']:
        raise Exception('Unsupported network: %s' % name)
    data_shape = (5, 1, 128)
    state_shape = (1, 128)
    num_states = 2 if name == 'lstm' else 1

    data_names = ['data%s' % i for i in range(num_states + 1)]
    shapes = {'data' : data_shape}
    for i in range(num_states):
        shapes['state%s' % i] = state_shape

    fname = '%s_i128_h128' % name
    net = gluon.nn.SymbolBlock.imports(
        "model_zoo_data/%s-symbol.json.data" % (fname),
        data_names,
        "model_zoo_data/%s-0001.params.data" % (fname))
    return net, num_states, shapes
