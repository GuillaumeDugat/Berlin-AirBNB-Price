from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

from data_preprocess import get_processed_train_test
from random_predictor import compute_accuracy_margin_random

RANDOM_STATE = 42
CROSS_VALIDATION = 5
VERBOSE = 3

def ada_boost(X_train, X_test, Y_train, Y_test):
    print('Training model...')
    ADBregr = AdaBoostRegressor(random_state = RANDOM_STATE, n_estimators = 4)
    ADBregr.fit(X_train, Y_train)
    
    print('Predictions:')
    predictions = ADBregr.predict(X_test)

    mse = mean_squared_error(Y_test, predictions)
    print('MSE:', mse)

    mrg = compute_accuracy_margin_random(Y_test, predictions, 20)
    print('Margin accuracy', mrg)


def gradient_boosted_tree(X_train, X_test, Y_train, Y_test):
    print('Training model...')
    GDBreg  = GradientBoostingRegressor(n_estimators = 80, learning_rate=0.1,
        max_depth = 1, random_state = RANDOM_STATE, loss = 'squared_error')
    GDBreg .fit(X_train, Y_train)
    
    print('Predictions:')
    predictions = GDBreg .predict(X_test)

    mse = mean_squared_error(Y_test, predictions)
    print('MSE:', mse)

    mrg = compute_accuracy_margin_random(Y_test, predictions, 20)
    print('Margin accuracy', mrg)


def xg_boost(X_train, X_test, Y_train, Y_test):
    print('Training model...')
    ADBregr = XGBRegressor(n_estimators = 40, learning_rate = 0.1, max_depth = 3, random_state=RANDOM_STATE)
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    ADBregr.fit(X_train, Y_train)
    
    print('Predictions:')
    predictions = ADBregr.predict(X_test)

    mse = mean_squared_error(Y_test, predictions)
    print('MSE:', mse)

    mrg = compute_accuracy_margin_random(Y_test, predictions, 20)
    print('Margin accuracy', mrg)


def ada_boost_gridsearch(X_train, X_test, Y_train, Y_test):
    print('Training model...')
    ADBregr = AdaBoostRegressor(random_state = RANDOM_STATE)
    pgrid = {
        'base_estimator': [DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=3), DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5)],
        'n_estimators': [4,8,15,30,50],
        'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        'loss': ['linear'], #['linear', 'square', 'exponential'],
    }
    grid_search = GridSearchCV(ADBregr, param_grid=pgrid, scoring='neg_mean_squared_error', cv=CROSS_VALIDATION, verbose=VERBOSE)
    grid_search.fit(X_train, Y_train)

    print('Best parameters:', grid_search.best_params_)
    print('MSE on test set:', -grid_search.score(X_test, Y_test))
    
    # print('Predictions:')
    # predictions = grid_search.best_estimator_.predict(X_test)

    # mse = mean_squared_error(Y_test, predictions)
    # print('MSE:', mse)

    # mrg = compute_accuracy_margin_random(Y_test, predictions, 20)
    # print('Margin accuracy', mrg)

    # Best parameters: {'learning_rate': 0.01, 'loss': 'linear', 'n_estimators': 8}
    # MSE on test set: 1519.6215054528334

    # Best parameters: {'learning_rate': 0.001, 'loss': 'linear', 'n_estimators': 4}
    # MSE on test set: 1652.7434469200057

    # Best parameters: {'base_estimator': DecisionTreeRegressor(random_state=42), 'learning_rate': 0.001, 'loss': 'exponential', 'n_estimators': 30}
    # MSE on test set: 1472.121773334426

    # Best parameters: {'base_estimator': DecisionTreeRegressor(max_depth=5, random_state=42), 'learning_rate': 0.001, 'loss': 'linear', 'n_estimators': 30}
    # MSE on test set: 1405.8201892482223


def gradient_boosted_tree_gridsearch(X_train, X_test, Y_train, Y_test):
    print('Training model...')
    GDBreg  = GradientBoostingRegressor(random_state = RANDOM_STATE)
    pgrid = {
        'n_estimators': [4,8,15,30,50, 100],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 1],
        'max_depth': [1, 3, 5],
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    }
    grid_search = GridSearchCV(GDBreg, param_grid=pgrid, scoring='neg_mean_squared_error', cv=CROSS_VALIDATION, verbose=VERBOSE)
    grid_search.fit(X_train, Y_train)

    print('Best parameters:', grid_search.best_params_)
    print('MSE on test set:', -grid_search.score(X_test, Y_test))

    # Best parameters: {'learning_rate': 0.1, 'loss': 'squared_error', 'max_depth': 1, 'n_estimators': 100}
    # MSE on test set: 1356.829711237964


def xg_boost_gridsearch(X_train, X_test, Y_train, Y_test):
    print('Training model...')
    ADBregr = XGBRegressor(n_estimators = 40, learning_rate = 0.1, max_depth = 3, random_state=RANDOM_STATE)
    ADBregr  = XGBRegressor(random_state = RANDOM_STATE)
    pgrid = {
        'n_estimators': [4,8,15,30,50, 80],
        'learning_rate': [0.01, 0.05, 0.1, 0.5],
        'max_depth': [2, 3, 5],
    }
    grid_search = GridSearchCV(ADBregr, param_grid=pgrid, scoring='neg_mean_squared_error', cv=CROSS_VALIDATION, verbose=VERBOSE)
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    grid_search.fit(X_train, Y_train)

    print('Best parameters:', grid_search.best_params_)
    print('MSE on test set:', -grid_search.score(X_test, Y_test))

    # Best parameters: {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 80}
    # MSE on test set: 1571.7219762337752


if __name__ == '__main__':
    print('Preprocessing...')
    X_train, X_test, Y_train, Y_test, _ = get_processed_train_test(path_to_folder='data', add_processing=False)

    # ada_boost(X_train, X_test, Y_train, Y_test)
    # ada_boost_gridsearch(X_train, X_test, Y_train, Y_test)
    # gradient_boosted_tree(X_train, X_test, Y_train, Y_test)
    # gradient_boosted_tree_gridsearch(X_train, X_test, Y_train, Y_test)
    # xg_boost(X_train, X_test, Y_train, Y_test)
    xg_boost_gridsearch(X_train, X_test, Y_train, Y_test)
