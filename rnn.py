import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, Permute, Reshape, AveragePooling1D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adagrad, Adam
from keras.regularizers import l2
import tensorflow as tf
from train_and_eval import train_model
import utils as u

np.random.seed(0)


def make_rnn_model(n_dim, n_labels):
    # data dimension parameters
    frames = n_dim[0]
    bands = n_dim[1]

    num_channels = n_dim[2] if len(n_dim) > 2 else 1

    # params
    model_name = 'rnn'
    training = u.TrainingParams(n_epoch=50, batch_size=32, early_stop_patience=8)

    model = Sequential()
    if num_channels == 1:
        # model.add(Permute((2,1), input_shape=(bands, frames)))
        pass
    else:
        with tf.variable_scope("Reshape"):
            # model.add(Permute((2,1,3), input_shape=(bands, frames, num_channels)))
            model.add(Reshape((frames, num_channels*bands), input_shape=(frames, bands, num_channels)))

    with tf.variable_scope("LSTM1"):
        model.add(LSTM(25, input_dim=num_channels*bands, input_length=frames, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(AveragePooling1D())

    with tf.variable_scope("LSTM2"):
        model.add(LSTM(25, input_dim=num_channels * bands, input_length=frames, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(AveragePooling1D())
        # model.add(AveragePooling1D())

    with tf.variable_scope("flatten"):
        # flatten to single dimension
        model.add(Flatten())
    # with tf.variable_scope("LSTM2"):
    #     model.add(LSTM(48, input_dim=num_channels*bands, input_length=frames, return_sequences=True))
    #     model.add(Dropout(0.5))
    #
    # with tf.variable_scope("LSTM3"):
    #     model.add(LSTM(32, input_dim=num_channels*bands, input_length=frames, return_sequences=True))
    #     model.add(Dropout(0.5))
    #
    # with tf.variable_scope("LSTM4"):
    #     model.add(LSTM(16))
    #     model.add(Dropout(0.5))

    with tf.variable_scope("Dense1"):
        model.add(Dense(n_labels, W_regularizer=l2(0.001)))
        model.add(Activation('softmax'))


    # optimizer
    # sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    # optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=0.001)
    # adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model_name, model, training


if __name__ == "__main__":
    train_model(make_rnn_model, u.FEATURE_SET_MFCCS)
