# File name: Ada_Boost.py
# https://en.wikipedia.org/wiki/AdaBoost
# https://machinelearningmastery.com/adaboost-ensemble-in-python/
# AdaBoost Algorithm

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


import warnings
warnings.filterwarnings("ignore")


def confusion_matrix_accuracy(cm):
    """
    confusion_matrix_accuracy method
    :param cm: {array-like}
    :return: {float}
    """
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    return diagonal_sum / sum_of_all_elements


def adaboost_classifier(x_cls, y_cls, n_estimators, max_depth, learning_rate):
    X_train, X_test, y_train, y_test = train_test_split(x_cls, y_cls
                                                        , test_size=0.2
                                                        , random_state=5)
    base_estimator = DecisionTreeClassifier(max_depth=max_depth)
    # AdaBoost Classifier
    ab = AdaBoostClassifier(n_estimators=n_estimators
                            , base_estimator=base_estimator
                            , learning_rate=learning_rate)
    """ 
        base_estimator: 
            LogisticRegression()
        """
    """
        When using machine learning algorithms that have a stochastic learning algorithm, it is good practice 
        to evaluate them by averaging their performance across multiple runs or repeats of cross-validation. 
        When fitting a final model it may be desirable to either increase the number of trees until the variance 
        of the model is reduced across repeated evaluations, or to fit multiple final models and average 
        their predictions.
        """
    # Evaluate the model using Cross Validation
    c_v = RepeatedStratifiedKFold(n_splits=10
                                  , n_repeats=3
                                  , random_state=1)
    n_scores = cross_val_score(ab, X_train, y_train
                               , scoring='accuracy'
                               , cv=c_v
                               , n_jobs=-1
                               , error_score='raise')
    grid_search(ab, X_train, y_train)
    # Print model performance
    print('Model Accuracy: \n', n_scores)

    # Fit
    ab.fit(X_train, y_train)
    # Predict
    y_predicted = ab.predict(X_test)

    print("\nAccuracy Score: : {%.2f%%}" % (accuracy_score(y_test, y_predicted) * 100.0))
    cm = confusion_matrix(y_test, y_predicted)
    print("Confusion Matrix: {%.2f%%}" % (confusion_matrix_accuracy(cm) * 100.0))
    f1_test2 = f1_score(y_test, y_predicted, average='weighted')
    print("F1 SCORE for test set: {%.2f%%}" % (f1_test2 * 100.0))


def get_data():
    _X, _y = None, None

    args = {'n_samples': 1000
            , 'n_features': 20
            , 'n_informative': 15
            , 'random_state': 5}

    flag = int(input("\nChoose: "
                     "\n(1) Iris Dataset "
                     "\n(2) Brest Cancer Dataset "
                     "\n(3) Make a Random Classification Dataset!"
                     "\nExit: press any other key!\n"))
    if flag == 1:
        # Load Iris dataset
        iris = datasets.load_iris()
        _X = iris.data
        _y = iris.target
    elif flag == 2:
        # Load Brest Cancer data
        bc = datasets.load_breast_cancer()
        _X = bc.data
        _y = bc.target
    elif flag == 3:
        # Create a regression dataset
        _X, _y = make_regression(noise=0.1, **args)
    elif flag == 4:
        # Create a classification dataset
        _X, _y = make_classification(n_redundant=5, **args)
    else:
        print('-' * 50)
        print("Exit")
        print('-' * 50)
        exit()

    # X, y shapes
    print(f'\nX shape:{_X.shape} \n y shape: {_y.shape}')

    return _X, _y, flag


def grid_search(model, x_trn, y_trn):
    # define the grid of values to search
    grid = dict()
    grid['n_estimators'] = [10, 50, 100, 500]
    grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10
                                 , n_repeats=3
                                 , random_state=1)
    # define the grid search procedure
    grid_search = GridSearchCV(estimator=model
                               , param_grid=grid
                               , n_jobs=-1
                               , cv=cv
                               , scoring='accuracy')
    # execute the grid search
    grid_result = grid_search.fit(x_trn, y_trn)
    # summarize the best score and configuration
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # summarize all scores that were evaluated
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    while True:
        X, y, _flag = get_data()

        if _flag != 4:
            # AdaBoost Regression
            print('-' * 100)
            print('AdaBoost Regression')
            print('-' * 100)
            adaboost_regressor(X, y)

        if _flag != 3:
            # AdaBoost Classification
            print('-' * 100)
            print('AdaBoost Classification')
            print('-' * 100)
            adaboost_classifier(X, y, n_estimators=50, max_depth=10, learning_rate=1.1)
