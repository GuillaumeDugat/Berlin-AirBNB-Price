from sklearn.metrics import mean_squared_error,mean_absolute_error

import numpy as np

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