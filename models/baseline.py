"""Implements simple models to compare our approaches"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


class MeanEstimator(BaseEstimator):
    """Use the mean of y for predictions"""
    def __init__(self):
        super().__init__()
        self.mean = None

    def fit(self, X, y):
        self.mean = np.mean(y)

        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean)

class KNNMeanEstimator(KNeighborsRegressor):
    """Use the mean of y among the closest neighbors for predictions"""
    def __init__(self, columns, **kwargs):
        super().__init__(**kwargs)
        self.columns = "" # To prevent bug in __str__
        self.selection_col = np.isin(columns[:-1], ["Latitude", "Longitude"]) # Select only latitude and longitude

    def fit(self, X, y):
        return super().fit(X[:, self.selection_col], y)

    def predict(self, X):
        return super().predict(X[:, self.selection_col])


def create_mean_reg():
    return MeanEstimator()

def create_KNN_mean_reg(columns):
    return KNNMeanEstimator(columns, n_neighbors=100, weights='uniform')

def create_linear_reg():
    return LinearRegression()
