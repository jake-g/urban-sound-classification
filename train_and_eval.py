import datetime
import time

import numpy as np
import os
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adagrad, Adam, SGD, Adamax, RMSprop
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

import utils as u

np.random.seed(0)  # for reproducibility


def train_model(model_generator, feature_set):

    # TODO get data function that returns dict (options for one hot or not, val or not)
    # load dataset
    paths = u.load_paths('PATHS.yaml')  # get paths from file
    train_x, val_x, test_x = u.load_data(paths['extracted_data'] + 'features_%s.p' % feature_set)
    train_y, val_y, test_y = u.load_data(paths['extracted_data'] + 'labels_%s.p' % feature_set)

    model_name, model, training = model_generator(n_dim=train_x.shape[1:], n_labels=test_y.shape[1])
    run_id = '%s_%s' % (model_name, datetime.datetime.now().isoformat())
    print('\nTrain and Evaluate: %s' % model_name)

    # callbacks
    earlystop = EarlyStopping(monitor='val_loss', patience=training.early_stop_patience, verbose=1, mode='auto')
    log_dir=os.path.join(paths['tensorboard_logs'], run_id)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=3, write_graph=True)
    t0 = time.time()
    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), callbacks=[earlystop, tensorboard], nb_epoch=training.n_epoch,
                        batch_size=training.batch_size)
    training_time = time.time() - t0

    # test
    y_prob = model.predict_proba(test_x, verbose=0)
    y_pred = np_utils.probas_to_classes(y_prob)
    y_true = np.argmax(test_y, 1)

    # evaluate the model's accuracy
    t0 = time.time()
    score, accuracy = model.evaluate(test_x, test_y, batch_size=training.batch_size)
    testing_time = time.time() - t0
    cm = confusion_matrix(y_true, y_pred, labels=None)
    # p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    # roc = roc_auc_score(test_y, y_prob)
    # print("F-Score:", round(f, 2))  # similar value to the accuracy score, but useful for cross-checking
    # print("ROC:", round(roc, 3))

    # print results
    print("\nclassifier: %s" % model_name)
    print("training time: %0.4fs" % training_time)
    print("testing time: %0.4fs" % testing_time)
    print("accuracy: %0.4f" % accuracy)
    print("confusion matrix:\n%s" % cm)
    # print model.summary()


    # plot and save results
    fname = paths['model_save'] + model_name + '_accuracy_%0.2f' % accuracy
    u.plot_keras_loss(fname, history)  # saves plot png
    model.save(fname + '.h5')
    cm_path = './confusion_plots/%s' % model_name
    cm_title = '%s (Accuracy: %0.2f)' % (model_name, accuracy)
    u.plot_confusion_matrix(cm, cm_path, title=cm_title)


def grid_search(model_generator, param_grid, feature_set):
    print('\nGrid Search')
    # load data
    print('loading', feature_set)
    paths = u.load_paths('PATHS.yaml')  # get paths from file
    train_x, val_x, test_x = u.load_data(paths['extracted_data'] + 'features_%s.p' % feature_set)
    train_y, val_y, test_y = u.load_data(paths['extracted_data'] + 'labels_%s.p' % feature_set)
    test_x = np.vstack((val_x, test_x))  # dont use validation set for grid search, add to test data
    test_y = np.vstack((val_y, test_y))

    # train
    # grid search train with 3 fold validation
    classifier = KerasClassifier(model_generator, n_dim=train_x.shape[1:], n_labels=test_y.shape[1])
    validator = GridSearchCV(classifier, param_grid=param_grid, scoring='neg_log_loss', verbose=3, n_jobs=1)
    validator.fit(train_x, train_y)

    # results
    search_results(validator, test_x, test_y)


def random_search(model_generator, param_dist, n_iter, feature_set):
    print('\nRandomized {[Search witn %d iterations\n' % n_iter)
    # load data
    print('loading', feature_set)
    paths = u.load_paths('PATHS.yaml')  # get paths from file
    train_x, val_x, test_x = u.load_data(paths['extracted_data'] + 'features_%s.p' % feature_set)
    train_y, val_y, test_y = u.load_data(paths['extracted_data'] + 'labels_%s.p' % feature_set)
    test_x = np.vstack((val_x, test_x))  # dont use validation set for grid search, add to test data
    test_y = np.vstack((val_y, test_y))

    # train
    # grid search train with 3 fold validation
    classifier = KerasClassifier(model_generator, n_dim=train_x.shape[1:], n_labels=test_y.shape[1])
    validator = RandomizedSearchCV(classifier, param_distributions=param_dist, n_iter=n_iter, verbose=3)
    validator.fit(train_x, train_y)

    # results
    search_results(validator, test_x, test_y)



def search_results(validator, test_x, test_y):
    # results
    # validator.best_estimator_ returns sklearn-wrapped version of best model.
    # validator.best_estimator_.model returns the (unwrapped) keras model
    print('The parameters of the best model are: ')
    print(validator.best_params_)
    best_model = validator.best_estimator_.model
    metric_names = best_model.metrics_names
    metric_values = best_model.evaluate(test_x, test_y)
    for metric, value in zip(metric_names, metric_values):
        print(metric, ': ', value)


def get_optimizers(learning_rate, optimizer):
    # helper for models takes optimizer key string and learning rate
    optimizers = {
        'sgd': SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False),
        'adagrad': Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0),
        'adam': Adam(lr=learning_rate),
        'adamax': Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        'rmsprop': RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    }
    return optimizers[optimizer]
