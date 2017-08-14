import tensorflow as tf
from scipy.stats import randint as randint
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.regularizers import l2

import utils as u
from train_and_eval import get_optimizers, train_model, grid_search, random_search


def make_cnn_model(n_dim, n_labels, pool_size=(4,2), learning_rate=0.001, f_size=3, optimizer='adamax', n1=24, n2=48, n3=48):

    # hyperparams (if not using grid or random search)
    model_name = '5_layer_cnn'
    training = u.TrainingParams(n_epoch=40, batch_size=30, early_stop_patience=8) # hardcoded params


    # 5 layer CNN described in https://arxiv.org/pdf/1608.04363.pdf
    # input data frames => (128,128,1) tensors

    # data dimension parameters
    frames = n_dim[1]
    bands = n_dim[0]
    num_channels = n_dim[2]
    optimizer = get_optimizers(learning_rate, optimizer)

    model = Sequential()

    with tf.variable_scope("layer1"):
        # W shape (24,1,f,f),(4,2) max-pooling, ReLU activation
        model.add(Convolution2D(n1, f_size, f_size, border_mode='valid', input_shape=(bands, frames, num_channels)))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Activation('relu'))

    with tf.variable_scope("layer2"):
        # W has the shape (48,24,f,f), (4,2) max-pooling, ReLU activation
        model.add(Convolution2D(n2, f_size, f_size, border_mode='valid'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Activation('relu'))

    with tf.variable_scope("layer3"):
        # W shape (48, 48, f, f), no pooling, ReLU activation
        model.add(Convolution2D(n3, f_size, f_size, border_mode='valid'))
        model.add(Activation('relu'))

    with tf.variable_scope("flatten"):
        # flatten to single dimension
        model.add(Flatten())

    with tf.variable_scope("conn1"):
        # fully connected, 64 hidden units, L2 penalty
        model.add(Dense(64, W_regularizer=l2(0.001)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    with tf.variable_scope("conn2"):
        # output layer,one output unit per class, L2 penalty, softmax activation
        model.add(Dense(n_labels, W_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model_name, model, training


def get_cnn_model(n_dim, n_labels, pool_size, learning_rate, f_size, optimizer, n1, n2, n3):
    # helper for grid search
    model_name, model, training = make_cnn_model(**locals())  # pass all input vars to make_model
    return model


if __name__ == "__main__":
    feature_set = u.FEATURE_SET_SPECS
    # TODO make command line args for these

    # grid saerch
    param_grid = {
        'nb_epoch': [30],  # 20, 25, 30, 40
        'batch_size': [30],  # 20 30, 50, 100, 80, 40,
        'learning_rate': [0.001], # 0.01, 0.001,  0.005
        'pool_size': [(4, 2)],  # (2, 4), (2, 2), (4, 4)
        'f_size': [3], # 5
        'optimizer': ['adamax'],
        'n1': [24],
        'n2': [48],
        'n3': [48]
    }
    # grid_search(get_cnn_model, param_grid, feature_set)
    '''
    {'pool_size': (4, 2), 'optimizer': 'adamax', 'n3': 48, 'n1': 24, 'learning_rate': 0.001, 'batch_size': 30, 'n2': 48, 'f_size': 3, 'nb_epoch': 30}
    loss :  0.976803440588, acc :  0.731228070259
    '''

    # random search
    n_iterations = 10
    param_dist = {
        'nb_epoch': randint(25, 45),
        'batch_size': randint(15, 40),
    }
    # random_search(get_cnn_model, param_dist, n_iterations, feature_set)

    train_model(make_cnn_model, feature_set)
