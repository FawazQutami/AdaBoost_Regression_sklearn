# File name: AdaBoost_Regression_sklearn.py
# https://en.wikipedia.org/wiki/AdaBoost
# https://machinelearningmastery.com/adaboost-ensemble-in-python/
# AdaBoost Algorithm

import pandas as pd
import numpy as np
from time import perf_counter

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

import warnings

# pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")
marks = '-' * 100


def model_validation(model, x_trn, y_trn):
    # Evaluate the model using Cross Validation
    c_v = RepeatedKFold(n_splits=10
                        , n_repeats=3
                        , random_state=1)
    cv_scores = cross_val_score(model, x_trn, y_trn
                                , scoring='neg_mean_absolute_error'
                                , cv=c_v
                                , n_jobs=-1
                                , error_score='raise')

    # Print model performance
    print('\nCross Validation - Model Accuracy: \n', cv_scores)


def ada_boost_regression(x_reg, y_reg, random_state, base_estimator):
    X_train, X_test, y_train, y_test = train_test_split(x_reg, y_reg
                                                        , test_size=0.2
                                                        , random_state=random_state)

    """
        When using machine learning algorithms that have a stochastic learning algorithm, it is good practice 
        to evaluate them by averaging their performance across multiple runs or repeats of cross-validation. 
        When fitting a final model it may be desirable to either increase the number of trees until the variance 
        of the model is reduced across repeated evaluations, or to fit multiple final models and average 
        their predictions.
        """

    # Evaluate the model using Grid Search
    learning_rate, n_estimators = validation(AdaBoostRegressor()
                                             , X_train
                                             , y_train
                                             , random_state=random_state)
    # Evaluate the model using Cross Validation
    # model_validation(AdaBoostRegressor(), X_train, y_train)

    # Build the AdaBoostRegressor using the best parameters from the grid search
    ab = AdaBoostRegressor(base_estimator=base_estimator
                           , loss='linear'  # {‘linear’, ‘square’, ‘exponential’},
                           , learning_rate=learning_rate
                           , n_estimators=n_estimators
                           , random_state=random_state)
    print("-- Model configuration:\n\t", ab)

    # Fit
    ab.fit(X_train, y_train)
    # Predict
    y_predicted = ab.predict(X_test)

    # Report the model performance
    # Performance report for testing data ---------------------
    test_data_list = performance_report(y_test, y_predicted)
    # Performance report for training data ---------------------
    train_data_list = performance_report(y_train, ab.predict(X_train))

    metrics = ['MSE (Mean Squared Error)'
        , 'RMSE (Root Mean Squared Error)'
        , 'MAE (Mean Absolute Error)'
        , 'R-squared (coefficient of determination)']
    performance_df = pd.DataFrame({'Training performance': train_data_list
                                      , 'Testing performance': test_data_list}
                                  , index=metrics)
    print('\n-- Performance report:\n', performance_df)

    # Get the feature importance ------------
    feature_imp = pd.DataFrame({'importance': ab.feature_importances_}
                               , index=list(X_train))
    print('\n-- Feature Importance:\n', feature_imp.sort_values(by=['importance']
                                                                , ascending=False))

    return feature_imp.nlargest(10, 'importance').sort_values(by=['importance']
                                                              , ascending=False)


def performance_report(set1, set2):
    mse = '{:.3f}'.format(mean_squared_error(set1, set2))
    rmse = '{:.3f}'.format(np.sqrt(r2_score(set1, set2)) * 100.0)
    mae = '{:.3f}'.format(mean_absolute_error(set1, set2))
    r2 = '{:.3f}'.format(r2_score(set1, set2) * 100.0)

    performance_list = [mse, rmse, mae, r2]

    return performance_list


def validation(model, x_trn, y_trn, random_state):
    # Grid parameters
    grid = dict()
    grid['n_estimators'] = [10, 50, 100, 500]
    grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1]

    # Evaluation procedure
    c_v = RepeatedStratifiedKFold(n_splits=10
                                  , n_repeats=10
                                  , random_state=random_state)
    # Grid search procedure
    grid_search = GridSearchCV(estimator=model
                               , param_grid=grid
                               , n_jobs=-1  # -1 means using all processors
                               , cv=c_v
                               , scoring='neg_mean_squared_error'
                               # 'neg_mean_absolute_error'
                               # 'neg_root_mean_squared_error'
                               # 'neg_mean_squared_log_error'
                               # 'explained_variance'
                               , error_score='raise'
                               )
    # Fit the grid search
    """scaler = StandardScaler().fit(x_trn)
    features = scaler.transform(x_trn)"""
    results = grid_search.fit(x_trn, y_trn)
    # Best score - Mean cross-validated score of the best_estimator
    print("-- GridSearchCV best score result:\n\t %f" % results.best_score_)
    # Best estimator - Estimator that was chosen by the search
    print("-- GridSearchCV best estimator result:\n\t %s" % results.best_estimator_)

    return results.best_params_['learning_rate'], results.best_params_['n_estimators']


def get_data():
    features, target, data_name = None, None, None

    flag = int(input("\nChoose a Dataset: *****************************"
                     "\n(1) Iris Dataset "
                     "\n(2) Brest cancer Dataset "
                     "\n(3) Boston house-prices Dataset "
                     "\nExit: press any other key!\n"))
    if flag == 1:
        # Load Iris dataset
        iris = datasets.load_iris()
        data = pd.DataFrame(data=np.c_[iris['data'], iris['target']]
                            , columns=np.append(iris['feature_names'], ['target']))
        features, target = data.drop(columns=['target'], axis=1), data['target']
        data_name = 'Iris dataset'

    elif flag == 2:
        # Load Brest Cancer dataset
        bc = datasets.load_breast_cancer()
        data = pd.DataFrame(data=np.c_[bc['data'], bc['target']]
                            , columns=np.append(bc['feature_names'], ['target_names']))
        features, target = data.drop(columns=['target_names'], axis=1), data['target_names']
        data_name = 'Brest Cancer dataset'

    elif flag == 3:
        # Load Boston house-prices dataset
        bh = datasets.load_boston()
        # features, target =bh.data, bh.target

        data = pd.DataFrame(data=np.c_[bh['data'], bh['target']]
                            , columns=np.append(bh['feature_names'], ['MEDV']))
        features, target = data.drop(columns=['MEDV'], axis=1), data['MEDV']
        # Convert target from continuous to categorical using LabelEncoder
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target)
        data_name = 'Boston house-prices dataset'

    else:
        print(marks)
        print("Exit")
        print(marks)
        exit()

    return features, target, data_name  # , data


def plot_feature_importance(fi, nam):
    x_labels = list(fi.index)
    y_labels = list(fi['importance'])
    x = np.arange(1, len(x_labels) + 1, 1)

    plt.style.use('ggplot')
    # Format the plot
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.5

    ax.bar(x, fi['importance'], width)
    plt.axis([0, max(x) + width, 0, max(y_labels) + 0.1])
    ax.axhline(y=0, color='g')
    ax.axvline(x=0, color='g')
    ax.set_xlabel('Features', fontdict=font)
    ax.set_ylabel('Importance', fontdict=font)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    color = ['c', 'g', 'b', 'm']
    for i in x:
        ax.hlines(y=y_labels[i - 1]
                  , xmin=min(x) - 1
                  , xmax=i
                  , colors='b'
                  , linestyles='dashed')
        ax.text(x=i
                , y=y_labels[i - 1]
                , s='{:.4f}'.format(y_labels[i - 1])
                , ha='center'
                , va='bottom'
                , color='darkred'
                , fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Feature importance for ' + nam, fontdict=font)
    plt.tight_layout(True)
    plt.show()


if __name__ == '__main__':
    while True:

        X, y, name = get_data()

        # AdaBoost Regression
        print(marks)
        print('AdaBoost Regression for ' + name + f'\n\tX.shape: {X.shape}, y.shape: {y.shape}')
        print(marks)

        start = perf_counter()
        feature_importance = ada_boost_regression(X, y, random_state=5, base_estimator=None)
        end = perf_counter()
        print(f"\n-- Execution time:\n\t {(end - start): .2f} seconds --> {(end - start) / 60: f} minutes")

        print(marks)
        plot_feature_importance(feature_importance, name)
