from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape

import utils as u
from train_and_eval import get_optimizers, train_model, grid_search, random_search



def make_ffn_model(n_dim, n_labels, nodes=512, learning_rate=0.001, optimizer='adamax'):
    # hyperparams (if not using grid or random search)
    model_name = 'ffn2'
    training = u.TrainingParams(n_epoch=20, batch_size=128, early_stop_patience=8)  # hardcoded

    # Simple nn
    frames = n_dim[1]
    bands = n_dim[0]
    num_channels = n_dim[2]

    # simple 3 layer model
    optimizer = get_optimizers(learning_rate, optimizer)
    model = Sequential()
    model.add(Reshape(input_shape=(bands, frames, num_channels), target_shape=(bands*frames,)))
    model.add(Dense(nodes, input_shape=(n_dim[0],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nodes))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_labels))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model_name, model, training


def get_ffn_model(n_dim, n_labels, nodes,learning_rate, optimizer):
    # helper for grid search
    model_name, model, training = make_ffn_model(**locals())  # pass all input vars to make_model
    return model  # must only return model for grid search wrapper


if __name__ == "__main__":
    feature_set = u.FEATURE_SET_SPECS_NORM

    param_grid = {
        'nb_epoch': [20],
        'batch_size': [32],
        'learning_rate': [0.0001],
        'optimizer': ['adamax'],
        'nodes': [512]
    }
    # grid_search(get_ffn_model, param_grid, feature_set)

    '''
    best params:
    {'batch_size': 64, 'learning_rate': 0.0005, 'nb_epoch': 30, 'nodes': 500, 'optimizer': 'adamax'}
    1425/1425 [==============================] - 0s
    loss :  1.5037392541
    acc :  0.490526316124
    '''
    train_model(make_ffn_model, feature_set)
