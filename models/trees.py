import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def create_decision_tree_model():

    regressor = DecisionTreeRegressor(random_state=42)

    parameters_grid = {
        #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], #best criterion = squared error
        'splitter': ['best', 'random'],
        'max_depth': [int(x) for x in np.linspace(1,100,num=21)]+[None],
        'min_samples_split': [k for k in range(2,11)],
        'min_samples_leaf': [1,2,4,8],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        estimator=regressor, param_grid=parameters_grid, scoring='neg_mean_squared_error', cv=3, verbose=2
    )

    return grid_search


def create_random_forest_model():

    regressor = RandomForestRegressor(random_state=42)

    random_grid = {
        'n_estimators': [int(x) for x in np.linspace(100, 2100, num=21)],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(10, 100, num=4)]+[None],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1,2,4],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=regressor, param_distributions=random_grid, n_iter=30, scoring='neg_mean_squared_error', cv=3, verbose=2
    )

    return random_search