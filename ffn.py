from keras.layers import Dense, Dropout
from keras.models import Sequential

import utils as u
from train_and_eval import get_optimizers, train_model, grid_search, random_search


def make_ffn_model(n_dim, n_labels, n1=100, n2=200, n3=200, learning_rate=0.0005, optimizer='adamax'):
    # hyperparams (if not using grid or random search)
    model_name = '3_layer_ffn'
    training = u.TrainingParams(n_epoch=40, batch_size=100, early_stop_patience=8)  # hardcoded

    # simple 3 layer model
    optimizer = get_optimizers(learning_rate, optimizer)
    model = Sequential()
    model.add(Dense(n1, input_dim=n_dim[0], init='normal', activation='relu'))
    model.add(Dense(n2, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n3, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_labels, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model_name, model, training


def get_ffn_model(n_dim, n_labels, n1, n2, n3, learning_rate, optimizer):
    # helper for grid search
    model_name, model, training = make_ffn_model(**locals())  # pass all input vars to make_model
    return model  # must only return model for grid search wrapper


if __name__ == "__main__":
    feature_set = u.FEATURE_SET_MEANS

    param_grid = {
        'nb_epoch': [40],
        'batch_size': [100],
        'learning_rate': [0.0005],
        'optimizer': ['adamax'],
        'n1': [100],
        'n2': [200],
        'n3': [200]
    }
    #grid_search(get_ffn_model, param_grid, feature_set)

    '''
    best params:
    {'n1': 100, 'batch_size': 100, 'nb_epoch': 40, 'n3': 200, 'optimizer': 'adamax', 'learning_rate': 0.0005, 'n2': 200}
    loss :  1.38495757374
    acc :  0.582577132883
    '''
    train_model(make_ffn_model, feature_set)
