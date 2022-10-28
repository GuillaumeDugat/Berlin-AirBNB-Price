import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR

BEST_PARAMS_SVR = {}
#MSE = 

N_ITER=30
CROSS_VALIDATION = 5
VERBOSE = 3

def create_SVR_model(best_model: bool=False):

    if best_model:
        return SVR(**BEST_PARAMS_SVR)

    regressor = SVR()

    pgrid = {
        'C': [10**k for k in range(-3, 3)],
        'tol': [10**(-4), 3*10**(-4), 10**(-3), 3*10**(-3), 10**(-2)],
        'epsilon': [0.01, 0.03, 0.1, 0.3, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'shrinking': [True,False]
    }
    
    #return GridSearchCV(regressor, param_grid=pgrid, scoring='neg_mean_squared_error', cv=3, verbose=2)
    return RandomizedSearchCV(regressor, param_distributions=pgrid, scoring='neg_mean_squared_error', n_iter=N_ITER, cv=CROSS_VALIDATION, verbose=VERBOSE)