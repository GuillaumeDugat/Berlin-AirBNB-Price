import numpy as np
from datetime import datetime

from sklearn.metrics import mean_squared_error,mean_absolute_error

def convert_date(date_str: str) -> int:
 
    date_compare = datetime(2022, 10, 13)
    date_to_compare = datetime.strptime(date_str, '%Y-%m-%d')
    comparison = (date_compare - date_to_compare).days

    return comparison


def get_quantiles_error(goals: np.array, predictions: np.array) -> dict:

    absolute_errors = np.abs(goals-predictions)
    absolute_errors = np.sort(absolute_errors)
    print(absolute_errors)
    quantiles_list = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_errors = {}

    n = len(absolute_errors)
    for quantile in quantiles_list:
        quantile_errors[quantile] = absolute_errors[int(n*quantile)]
    
    return quantile_errors


def full_evaluation(goals: np.array, predictions: np.array) -> dict:

    mse = mean_squared_error(goals, predictions)
    mae = mean_absolute_error(goals, predictions)
    quantiles = get_quantiles_error(goals, predictions)

    all_metrics = {
        'mean_squared_error': mse,
        'root_mean_squared_error': mse**0.5,
        'mean_absolute_error': mae,
        'quantiles': quantiles
    }

    return all_metrics

def print_eval(goals: np.array, predictions: np.array):

    all_metrics = full_evaluation(goals, predictions)

    print('Mean Squared Error : {}'.format(all_metrics['mean_squared_error']))
    print('Root Mean Squared Error : {}'.format(all_metrics['root_mean_squared_error']))
    print('Mean Absolute Error : {}'.format(all_metrics['mean_absolute_error']))

    print('1/10 of predictions are better than : {}'.format(all_metrics['quantiles'][0.9]))
    print('1/4 of predictions are better than : {}'.format(all_metrics['quantiles'][0.25]))
    print('Half of predictions are better than : {}'.format(all_metrics['quantiles'][0.5]))
    print('1/4 of predictions are worse than : {}'.format(all_metrics['quantiles'][0.75]))
    print('1/10 of predictions are worse than : {}'.format(all_metrics['quantiles'][0.9]))
