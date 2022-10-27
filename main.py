import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from preprocessing.baseline_preprocessing import get_baseline_preprocessed_train_test
from models.baseline import create_mean_reg, create_KNN_mean_reg, create_linear_reg, create_random_predictor
from models.boosting import create_ada_boost, create_gradient_boosted_tree, create_xg_boost
from utils import print_eval


def preprocess(
    path_to_folder: str = 'data',
    remove_outliers: bool=True,
    imputing_missing_values: bool=False,
    rescaling: bool = False
):
    """Perform all the preprocessing tasks (baseline preprocessing, rescaling, pca or subset selection)"""
    # Baseline preprocessing
    X_train, X_test, Y_train, Y_test, columns = get_baseline_preprocessed_train_test(
        path_to_folder=path_to_folder,
        remove_outliers=remove_outliers,
        imputing_missing_values=imputing_missing_values,
    )

    # Rescaling
    if rescaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test, columns


def main():
    print('Preprocessing...\n')
    X_train, X_test, Y_train, Y_test, columns = preprocess(
        path_to_folder='data',
        remove_outliers=True,
        imputing_missing_values=False,
        rescaling=False
    )

    # model = create_mean_reg()
    # model = create_KNN_mean_reg(columns)
    model = create_linear_reg()
    # model = create_random_predictor()
    # model = create_ada_boost()
    # model = create_gradient_boosted_tree()
    # model = create_xg_boost()

    print('Training model...\n')
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    print_eval(Y_test, predictions)


if __name__ == '__main__':
    main()
