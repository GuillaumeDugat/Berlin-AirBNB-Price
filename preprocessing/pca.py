from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

PCA_BEST_NUM_FEATURES = ...
PLS_BEST_N_COMPONENTS = ...


def pcaSelection(numFeatures: int = PCA_BEST_NUM_FEATURES):
    """
    numFeatures is the number of components desired
    Returns the PCA model with the number of features desired
    """
    return PCA(n_components=numFeatures)


def plsSelection(n_components: int = PLS_BEST_N_COMPONENTS):
    """
    Returns the PLS Modified X_transform with the number of desired components
    """
    return PLSRegression(n_components)
