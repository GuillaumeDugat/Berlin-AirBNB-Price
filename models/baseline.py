"""Implements simple models to compare our approaches"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from data_preprocess import get_processed_train_test
from random_predictor import compute_accuracy_margin_random


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


def baseline(model, X_train, X_test, Y_train, Y_test):
    print('Model :', model)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    print('Root Mean Square Error:', rmse)

    mae = mean_absolute_error(Y_test, predictions)
    print('Mean Absolute Error:', mae)

    mrg = compute_accuracy_margin_random(Y_test, predictions, 20)
    print('Margin accuracy', mrg)

    print()



if __name__ == '__main__':
    print('Preprocessing...\n')
    X_train, X_test, Y_train, Y_test, columns = get_processed_train_test(path_to_folder='data', add_processing=False)

    baseline(MeanEstimator(), X_train, X_test, Y_train, Y_test)
    
    selection_col = np.isin(columns[:-1], ["Latitude", "Longitude"]) # Select only latitude and longitude
    baseline(KNeighborsRegressor(n_neighbors=100, weights='uniform'), X_train[:, selection_col], X_test[:, selection_col], Y_train, Y_test)

    baseline(LinearRegression(), X_train, X_test, Y_train, Y_test)

