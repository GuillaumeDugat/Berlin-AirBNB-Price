from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

from data_preprocess import get_processed_train_test
from random_predictor import compute_accuracy_margin_random

if __name__ == '__main__':
    print('Preprocessing...')
    X_train, X_test, Y_train, Y_test, _ = get_processed_train_test(path_to_folder='data', add_processing=True)

    print('Training model...')
    ADBregr = AdaBoostRegressor(random_state = 42, n_estimators = 4)
    ADBregr.fit(X_train, Y_train)
    
    print('Predictions:')
    predictions = ADBregr.predict(X_test)

    mse = mean_squared_error(Y_test, predictions)
    print('MSE:', mse)

    mrg = compute_accuracy_margin_random(Y_test, predictions, 20)
    print('Margin accuracy', mrg)
