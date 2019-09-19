from validate_config import validate
from common import invoke_main, write_status
from trial_util import run_trials, configure_seed

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


def training_setup(device, batch_size, num_classes, epochs):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    thunk = lambda: model.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_test, y_test))
    return [thunk]


def training_trial(thunk):
    return thunk()


def training_teardown(thunk):
    pass


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    if 'keras' not in config['frameworks']:
        write_status(output_dir, True, 'Keras not run')
        return 0

    configure_seed(config)

    success, msg = run_trials(
        'keras', 'training_loop',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        training_trial, training_setup, training_teardown,
        ['device', 'batch_size', 'language', 'input'],
        [config['devices'], config['batch_sizes'],
         config['num_classes'], config['epochs']],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)
    if not success:
        return 1

if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'output_dir')
