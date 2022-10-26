import numpy as np
from eval_metrics import get_margin_accuracy
import random

from sklearn.metrics import mean_squared_error

def get_repartition(arr: np.array) -> dict:

    nb_rows = arr.shape[0]
    repartition = {}
    price_list = [0,20,40,60,80,100,125,150,175,200,250,300,400,500,900]

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


def get_predictions(Y_train: np.array, Y_test: np.array) -> np.array:

    repartition = get_repartition(Y_train)
    predictions = np.array([get_random_price(repartition) for k in range(Y_test.shape[0])])

    return predictions


def compute_MSE_random(Y_train: np.array, Y_test: np.array) -> float:

    predictions = get_predictions(Y_train, Y_test)

    return mean_squared_error(Y_test, predictions)

def compute_accuracy_margin_random(Y_train: np.array, Y_test: np.array, margin: int) -> float:

    predictions = get_predictions(Y_train, Y_test)
    
    return get_margin_accuracy(Y_test, predictions, margin)
