from keras_classifiers import ffn1, ffn2, cnn1, cnn2, train_and_evaluate
from sklearn_classifiers import train_and_evaluate_all
import utils as u


print('\n\nEvaluating DNN Classifiers')
# train_and_evaluate(cnn1, u.FEATURE_SET_SPECS)
# train_and_evaluate(cnn2, u.FEATURE_SET_SPECS)
# train_and_evaluate(ffn1, u.FEATURE_SET_MEANS)
# train_and_evaluate(ffn2, u.FEATURE_SET_SPECS_NORM)

print('\n\nEvaluating Classifiers')
train_and_evaluate_all(u.FEATURE_SET_MEANS)
# train_and_evaluate_all(u.FEATURE_SET_SPECS_NORM)