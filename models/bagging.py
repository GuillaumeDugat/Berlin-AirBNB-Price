from matplotlib.streamplot import Grid
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def create_bagging_model():

    regressor = BaggingRegressor(random_state=42)
    base_estimators = [LinearRegression(), DecisionTreeRegressor(random_state=42)]

    param_grid = {
        'base_estimator': base_estimators,
        'n_estimators': [int(x) for x in np.linspace(10,100, num=10)],
        'max_samples': [0.2,0.4,0.6,0.8,1.0],
        'max_features': [0.2,0.4,0.6,0.8,1.0],
        'bootstrap_bis': [True, False],
        'boostrap_features': [True, False]
    }

    grid_search = GridSearchCV(
        estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2
    )

    return grid_search