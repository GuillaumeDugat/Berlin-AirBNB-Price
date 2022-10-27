from sklearn.preprocessing import StandardScaler

from preprocessing.baseline_preprocessing import get_baseline_preprocessed_train_test
from preprocessing.pca import pcaSelection, plsSelection
from preprocessing.dimensional_reduction import forwardSelection, backwardSelection



path_to_folder = 'data',
remove_outliers =True,
imputing_missing_values =False,
rescaling = False,
pca= True

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
nb_cols = len(X_train[0])
pca_transformer = pcaSelection(nb_cols)
X_train = pca_transformer.fit_transform(X_train)

import matplotlib.pyplot as plt
import numpy as np

# Explained variance plot
plt.bar(range(1,len(pca_transformer.explained_variance_ratio_ )+1),pca_transformer.explained_variance_ratio_ )
plt.ylabel('Percentage of explained variance')
plt.xlabel('Number of components')
plt.plot(range(1,len(pca_transformer.explained_variance_ratio_ )+1),
         np.cumsum(pca_transformer.explained_variance_ratio_ ),
         c='red',
         label="Cumulative Explained Variance")
plt.legend(loc='upper left')
plt.show()