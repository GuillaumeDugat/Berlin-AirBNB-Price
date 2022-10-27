import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


def pcaSelection(X_train: pd.DataFrame, numFeatures: int):
    """
    numFeatures is the number of components desired
    Returns the PCA model with the number of features desired
    """
    pca_model = PCA(n_components=numFeatures)
    X_new = pca_model.fit_transform(X_train)
    return X_new


def plsSelection(X_train: pd.DataFrame, y_train:pd.Series, n_components: int) :
    """
    Returns the PLS Modified X_transform with the number of desired components
    """
    pls = PLSRegression(n_components)
    X_transform = pls.fit_transform(X_train, y_train.ravel())
    return X_transform
