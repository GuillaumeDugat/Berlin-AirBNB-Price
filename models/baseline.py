"""Implements simple models to compare our approaches"""

import random

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


def get_price_repartition(arr: np.array) -> dict:

    nb_rows = arr.shape[0]
    repartition = {}
    price_list = [0,20,40,60,80,100,125,150,175,200,250]

    for k in range(1,len(price_list)):
        nb_values = arr[arr <= price_list[k]].shape[0]
        proportion = int(1000*nb_values/nb_rows)/1000
        repartition[proportion] = (price_list[k-1]+1, price_list[k])
    
    return repartition

def get_random_price(repartition: dict) -> float:

    p = random.random()

    for proba in repartition:
        if p <= proba:
            start, end = repartition[proba]
            return float(random.randint(start, end))


class RandomPredictor(BaseEstimator):
    
    def __init__(self):
        super().__init__()
        self.repartition = {}
    
    def fit(self, X, y):
        self.repartition = get_price_repartition(y)

        return self
    
    def predict(self, X):
        predictions = np.array([get_random_price(self.repartition) for k in range(X.shape[0])])
        return predictions



def create_mean_reg():
    return MeanEstimator()

def create_KNN_mean_reg(columns):
    return KNNMeanEstimator(columns, n_neighbors=100, weights='uniform')

def create_linear_reg():
    return LinearRegression()

def create_random_predictor():
    return RandomPredictor()
