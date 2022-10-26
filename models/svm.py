import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR

def create_SVR_model():

    regressor = SVR()

    parameters = {
        'C': [10**k for k in range(-3, 3)],
        'tol': [10**(-4), 3*10**(-4), 10**(-3), 3*10**(-3), 10**(-2)],
        'epsilon': [0.01, 0.03, 0.1, 0.3, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'shrinking': [True,False]
    }
    
    grid_search = GridSearchCV(
        estimator=regressor, param_grid=parameters, scoring='neg_mean_squared_error', cv=3, verbose=2
    )

    return grid_search