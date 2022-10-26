from data_preprocess import get_processed_train_test
from random_predictor import compute_accuracy_margin_random, compute_MSE_random

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from eval_metrics import get_margin_accuracy
import numpy as np

def random_search_decision_tree(X_train, Y_train):

    regressor = DecisionTreeRegressor(random_state=42)

    criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    splitter = ['best', 'random']
    max_depth = [k for k in range(5,21)]
    max_depth.append(None)
    min_samples_split = [k for k in range(2,11)]
    min_samples_leaf = [1,2,4,8]
    max_features = ['auto', 'sqrt', 'log2']

    random_grid = {
        'criterion': criterion,
        'splitter': splitter,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features
    }

    random_search = RandomizedSearchCV(estimator=regressor, param_distributions=random_grid, random_state=42)
    search = random_search.fit(X_train, Y_train)

    return search

# Results RandomizedSearch DecisionTree = 
# {'splitter': 'random', 'min_samples_split': 8, 
# 'min_samples_leaf': 8, 'max_features': 'auto', 'max_depth': 13, 'criterion': 'friedman_mse'}

# Eval Results DecisionTree =
# MSE = 1603.03 / sqrt = 40
# accuracy 10 = 38.3% / accuracy 20 = 65.4%

def random_search_forest(X_train, Y_train):

    n_estimators = [int(x) for x in np.linspace(200, 2000, num=5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 100, num=4)]
    max_depth.append(None)
    min_samples_split = [2,5,10]
    min_samples_leaf = [1,2,4]
    bootstrap = [True, False]

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    regressor = RandomForestRegressor(random_state=42)

    regressor_random = RandomizedSearchCV(
        estimator=regressor, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42,n_jobs=-1
    )
    regressor_random.fit(X_train,Y_train)

    return regressor_random.best_params_

# results RandomizedSearch = 
    # {'n_estimators': 1550, 'min_samples_split': 10, 'min_samples_leaf': 1, 
    # 'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': True}

# Eval results =
# MSE 1332.74, sqrt(MSE) = 36.5
# accuracy 10 = 38% / accuracy 20 = 66%


if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test, columns = get_processed_train_test()

    bagging = BaggingRegressor(base_estimator=LinearRegression(),random_state=42)

    n_estimators = [int(x) for x in np.linspace(10, 100, num=10)]
    max_samples = [0.2,0.4,0.6,0.8,1.0]
    max_features = [0.2,0.4,0.6,0.8,1.0]
    bootstrap_bis = [True, False]
    bootstrap_features = [True, False]

    param_grid = {
        'n_estimators': n_estimators,
        'max_samples': max_samples,
        'max_features': max_features,
        'bootstrap': bootstrap_bis,
        'bootstrap_features': bootstrap_features
    }

    bagging_randomsearch = RandomizedSearchCV(estimator=bagging,param_distributions=param_grid, random_state=42)

    search = bagging_randomsearch.fit(X_train, Y_train)

    print(search.best_params_)
    best_model = search.best_estimator_

    best_model.fit(X_train, Y_train)

    predictions = best_model.predict(X_test)

    print(mean_squared_error(Y_test, predictions))
    print(get_margin_accuracy(Y_test, predictions, 10))
    