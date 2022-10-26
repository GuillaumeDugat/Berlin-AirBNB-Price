import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocessing.baseline_preprocessing import get_processed_train_test
from models.baseline import create_mean_reg, create_KNN_mean_reg, create_linear_reg


def baseline(model, X_train, X_test, Y_train, Y_test):
    print('Model :', model)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    print('Root Mean Square Error:', rmse)

    mae = mean_absolute_error(Y_test, predictions)
    print('Mean Absolute Error:', mae)

    print()



if __name__ == '__main__':
    print('Preprocessing...\n')
    X_train, X_test, Y_train, Y_test, columns = get_processed_train_test(path_to_folder='data', add_processing=False)

    # model = create_mean_reg()
    # model = create_KNN_mean_reg(columns)
    model = create_linear_reg()
    baseline(model, X_train, X_test, Y_train, Y_test)

