from scipy.stats import randint as randint
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.regularizers import l2

import utils as u
from train_and_eval import get_optimizers, train_model, grid_search, random_search
'''
Tried to make deeper cnn, but it seems like it doesnt learn very well and overfits compared to cnn.py
'''

def make_cnn_model(n_dim, n_labels, pool_size=(4, 2), learning_rate=0.0005, f_size=5, optimizer='adamax', n1=32):
    # hyperparams (if not using grid or random search)
    model_name = 'cnn2'
    training = u.TrainingParams(n_epoch=18, batch_size=15, early_stop_patience=8)  # hardcoded params

    # data dimension parameters
    frames = n_dim[1]
    bands = n_dim[0]
    num_channels = n_dim[2]
    optimizer = get_optimizers(learning_rate, optimizer)

    model = Sequential()

    model.add(Convolution2D(n1, f_size, f_size, border_mode='valid', input_shape=(bands, frames, num_channels)))
    model.add(Activation('relu'))
    model.add(Convolution2D(n1, f_size, f_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Convolution2D(n1, f_size, f_size, border_mode='same', input_shape=(bands, frames, num_channels)))
    model.add(Activation('relu'))
    model.add(Convolution2D(n1, f_size, f_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_labels, W_regularizer=l2(0.001)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model_name, model, training


def get_cnn_model(n_dim, n_labels, pool_size, learning_rate, f_size, optimizer, n1):
    # helper for grid search
    model_name, model, training = make_cnn_model(**locals())  # pass all input vars to make_model
    return model


if __name__ == "__main__":
    feature_set = u.FEATURE_SET_SPECS
    # TODO make command line args for these

    # grid saerch
    param_grid = {
        'nb_epoch': [15, 20],
        'batch_size': [30],
        'learning_rate': [0.0005],  #
        'pool_size': [(4, 2)],  # (2, 4), (2, 2), (4, 4)
        'f_size': [5],  # 5
        'optimizer': ['adamax'],
        'n1': [32],
    }

    # grid_search(get_cnn_model, param_grid, feature_set)
    '''
    {'f_size': 5, 'learning_rate': 0.0005, 'pool_size': (4, 2), 'optimizer': 'adamax',
    n1': 32, 'batch_size': 30, 'nb_epoch': 15}
    loss :  1.23839746878
    '''

    # random search
    n_iterations = 10
    param_dist = {
        'nb_epoch': randint(25, 45),
        'batch_size': randint(15, 40),
    }
    # random_search(get_cnn_model, param_dist, n_iterations, feature_set)

    train_model(make_cnn_model, feature_set)
