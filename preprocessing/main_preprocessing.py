from sklearn.preprocessing import StandardScaler

from preprocessing.baseline_preprocessing import get_baseline_preprocessed_train_test
from preprocessing.pca import pcaSelection, plsSelection


def preprocess(
    path_to_folder: str = 'data',
    remove_outliers: bool=True,
    imputing_missing_values: bool=False,
    rescaling: bool = False,
    pca: bool = False,
    pls: bool = False,
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
    
    # Principal Component Analysis (with the best num_features, see preprocessing/pca.py)
    if pca:
        pca_transformer = pcaSelection()
        X_train = pca_transformer.fit_transform(X_train)
        X_test = pca_transformer.transform(X_test)

    # Partial Least Squares (with the best n_components, see preprocessing/pca.py)
    elif pls:
        pls_transformer = plsSelection()
        X_train, _ = pls_transformer.fit_transform(X_train, Y_train.ravel()) # TODO why does it return a tuple?
        X_test = pls_transformer.transform(X_test)

    return X_train, X_test, Y_train, Y_test, columns
