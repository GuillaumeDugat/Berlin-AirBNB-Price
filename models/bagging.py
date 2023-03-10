from matplotlib.streamplot import Grid
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

BEST_PARAMS_BAGGING = {
    'best_estimator': LinearRegression(),
    'max_samples': 0.6,
    'n_estimators': 90,
}
#MSE = 1595

N_ITER=30
CROSS_VALIDATION = 5
VERBOSE = 3

def create_bagging_model(best_model: bool=False):

    if best_model:
        return BaggingRegressor(random_state=42, **BEST_PARAMS_BAGGING)

    regressor = BaggingRegressor(random_state=42)
    base_estimators = [LinearRegression(), DecisionTreeRegressor(random_state=42)]

    pgrid = {
        'base_estimator': base_estimators,
        'n_estimators': [int(x) for x in np.linspace(10,100, num=10)],
        'max_samples': [0.2,0.4,0.6,0.8,1.0],
        'max_features': [0.2,0.4,0.6,0.8,1.0],
        'bootstrap_features': [True, False]
    }

    #return GridSearchCV(regressor, param_grid=pgrid, scoring='neg_mean_squared_error', cv=3, verbose=2)
    return RandomizedSearchCV(regressor, param_distributions=pgrid, scoring='neg_mean_squared_error', n_iter=N_ITER, cv=CROSS_VALIDATION, verbose=VERBOSE)