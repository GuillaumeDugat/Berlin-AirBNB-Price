import numpy as np

def get_margin_accuracy(goals: np.array, predictions: np.array, margin: int) -> float:

    count, total = 0, goals.shape[0]
    for k in range(total):
        if abs(goals[k] - predictions[k]) <= margin:
            count += 1
    
    return count/total