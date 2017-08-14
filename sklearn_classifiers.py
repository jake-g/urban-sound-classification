import pickle
import time

import numpy as np
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import utils as u


def get_classifiers():
    return [
        # ('Logistic Regression (C=1)', LogisticRegression(C=1)),
        # ('SVM, linear', SVC(kernel="linear", C=0.015, cache_size=200)),
        # ('k nn', KNeighborsClassifier(3)),
        # ('Decision Tree', DecisionTreeClassifier(max_depth=15)),
        ('Random Forest', RandomForestClassifier(n_estimators=500, n_jobs=10)),
        # ('AdaBoost', AdaBoostClassifier()),
        # ('Naive Bayes', GaussianNB()),
        # ('LDA', LinearDiscriminantAnalysis()),
        # ('QDA', QuadraticDiscriminantAnalysis()),
        # ('Gradient Boosting', GradientBoostingClassifier(learning_rate=0.5, n_estimators=50))
    ]


def train_and_evaluate_all(feature_set):
    data = get_data(feature_set)
    classifiers = get_classifiers()

    # one loop to fit them all
    classifier_data = {}
    best = []
    for clf_name, clf in classifiers:
        print("\n'%s' classifier..." % clf_name)
        t0 = time.time()
        clf.fit(data['train']['X'], data['train']['y'])
        t1 = time.time()
        an_data = get_results(clf, data, t1 - t0, feature_set, clf_name=clf_name, save_confusion=True)
        classifier_data[clf_name] = {'training_time': t1 - t0,
                                     'testing_time': an_data['testing_time'],
                                     'accuracy': an_data['accuracy']}
        best.append((clf_name, an_data['accuracy']))

    best = sorted(best, key=lambda x: x[1])  # sort by accuracy
    print_top_n(best, 10)
    pickle.dump(best, open('./sk_classifier_results_%s.p' % feature_set, "wb"))


def get_results(clf, data, fit_time, feature_set, clf_name='', save_confusion=False):
    results = {}

    t0 = time.time()
    predicted = np.array([])
    for i in range(0, len(data['test']['X']), 128):  # go in chunks of size 128
        predicted_single = clf.predict(data['test']['X'][i:(i + 128)])
        predicted = np.append(predicted, predicted_single)
    t1 = time.time()
    cm = metrics.confusion_matrix(data['test']['y'], predicted)
    results['testing_time'] = t1 - t0
    results['accuracy'] = metrics.accuracy_score(data['test']['y'], predicted)

    print("classifier: %s" % clf_name)
    print("training time: %0.4fs" % fit_time)
    print("testing time: %0.4fs" % results['testing_time'])
    print("accuracy: %0.4f" % results['accuracy'])
    print("confusion matrix:\n%s" % cm)
    if save_confusion:
        path = './confusion_plots/%s_%s' % (clf_name, feature_set)
        title = '%s (accuracy: %0.2f)' % (clf_name, results['accuracy'])
        u.plot_confusion_matrix(cm, path, title=title)
    return results


def get_data(feature_set):
    # load dataset
    paths = u.load_paths('PATHS.yaml')  # get paths from file
    train_x, val_x, test_x = u.load_data(paths['extracted_data'] + 'features_%s.p' % feature_set)
    train_y, val_y, test_y = u.load_data(paths['extracted_data'] + 'labels_%s.p' % feature_set)
    test_x = np.vstack((val_x, test_x))  # dont use validation set for grid search, add to test data
    test_y = np.vstack((val_y, test_y))
    train_y = u.inv_one_hot_encode(train_y)  # remove one hot encoding
    test_y = u.inv_one_hot_encode(test_y)

    if feature_set == u.FEATURE_SET_SPECS_NORM:
        # flatten 128 x 128 image
        length = train_x.shape[1] * train_x.shape[2]
        train_x = train_x.reshape(train_x.shape[0], length)
        test_x = test_x.reshape(test_x.shape[0], length)

    data = {'train': {'X': train_x, 'y': train_y},
            'test': {'X': test_x, 'y': test_y},
            'n_classes': len(np.unique(train_y))}

    print("dataset has %i training samples and %i test samples." %
          (len(data['train']['X']), len(data['test']['X'])))

    return data


def print_top_n(l, n=10):
    print('\nTop %d' % n)
    for i in range(0, n):
        name, acc = l.pop()
        print('%d : %s (%0.2f)' % (i + 1, name, acc))


if __name__ == '__main__':
    feature_set = u.FEATURE_SET_MEANS
    # feature_set = u.FEATURE_SET_SPECS_NORM
    results = train_and_evaluate_all(feature_set)
