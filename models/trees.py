import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

BEST_PARAMS_TREE = {
    'max_depth': 5,
    'max_features': 'auto',
    'min_samples_leaf': 4,
    'min_samples_split': 9,
    'splitter': 'best'
}
#MSE = 1700

def create_decision_tree_model(best_model: bool=False):

    if best_model:
        return DecisionTreeRegressor(random_state=42, **BEST_PARAMS_TREE)

    regressor = DecisionTreeRegressor(random_state=42)

    pgrid = {
        #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], #best criterion = squared error
        'splitter': ['best', 'random'],
        'max_depth': [int(x) for x in np.linspace(1,100,num=21)]+[None],
        'min_samples_split': [k for k in range(2,11)],
        'min_samples_leaf': [1,2,4,8],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    return GridSearchCV(regressor, param_grid=pgrid, scoring='neg_mean_squared_error', cv=3, verbose=2)

BEST_PARAMS_FOREST = {
    'n_estimators': 200,
    'min_samples_split': 10,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': 70,
    'bootstrap': True
}
#MSE = 1600

def create_random_forest_model(best_model: bool=False):

    if best_model:
        return RandomForestRegressor(random_state=42, **BEST_PARAMS_FOREST)

    regressor = RandomForestRegressor(random_state=42)

    pgrid = {
        'n_estimators': [int(x) for x in np.linspace(20, 200, num=10)],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(10, 100, num=4)]+[None],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1,2,4],
        'bootstrap': [True, False]
    }

    return RandomizedSearchCV(regressor, param_distributions=pgrid, n_iter=30, scoring='neg_mean_squared_error', cv=3, verbose=3)