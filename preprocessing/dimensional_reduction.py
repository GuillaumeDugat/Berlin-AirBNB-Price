import time
from typing import List
import math

import pandas as pd
import numpy as np
import itertools

#PCA Selection imports
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from preprocessing.preprocessing_classes import SelectFeatures

FORWARD_SELECTION_BEST_FEATURES = [
    'Accomodates',
    'Room Type_Entire home/apt',
    'Bathrooms',
    'Property Type_Hotel',
    'Bedrooms',
    'Dist_GENDARMENMARKT',
    'Guests Included',
    'Property Type_Loft',
    'Neighborhood Group_NeukÃ¶lln',
    'Is Superhost',
    'Neighborhood Group_Charlottenburg-Wilm.',
    'Property Type_Serviced apartment',
    'Dist_BRANDEBOURG',
    'Dist_ALEXANDERPLATZ',
    'Property Type_Boutique hotel',
    'Neighborhood Group_Mitte',
    'Neighborhood Group_Treptow - KÃ¶penick',
    'Min Nights',
    'Latitude',
    'Property Type_Hostel',
    'Property Type_Apartment',
    'Neighborhood Group_Lichtenberg',
    'Neighborhood Group_Marzahn - Hellersdorf',
    'Room Type_Shared room',
    'Property Type_House',
    'Neighborhood Group_Pankow',
    'Property Type_Condominium',
    'Room Type_Private room'
]

BACKWARD_SELECTION_BEST_FEATURES = [
    'Is Superhost',
    'Latitude',
    'Longitude',
    'Accomodates',
    'Bathrooms',
    'Bedrooms',
    'Guests Included',
    'Min Nights',
    'Neighborhood Group_Charlottenburg-Wilm.',
    'Neighborhood Group_Friedrichshain-Kreuzberg',
    'Neighborhood Group_Mitte',
    'Neighborhood Group_NeukÃ¶lln',
    'Neighborhood Group_Pankow',
    'Neighborhood Group_Reinickendorf',
    'Neighborhood Group_Tempelhof - SchÃ¶neberg',
    'Neighborhood Group_Treptow - KÃ¶penick',
    'Property Type_Apartment',
    'Property Type_Boutique hotel',
    'Property Type_Hostel',
    'Property Type_Hotel',
    'Property Type_Loft',
    'Property Type_Serviced apartment',
    'Room Type_Entire home/apt',
    'Room Type_Private room',
    'Dist_ALEXANDERPLATZ',
    'Dist_BRANDEBOURG',
    'Dist_GENDARMENMARKT',
]

def processSubset(X_train:pd.DataFrame, y_train:pd.Series, feature_set: List[str]):
    """
    Returns a dict with the features applied to a Linear Regression Model with k-folding  
    and the mean of the mean squared error applied to each model.
    """
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    model = LinearRegression()
    score = -cross_val_score(model, X_train[list(feature_set)], y_train, scoring='neg_mean_squared_error', cv=cv)
    med_score = math.sqrt(np.mean(score))

    return {"features": feature_set, "MSE": med_score}

def forward(X_train: pd.DataFrame, y_train:pd.Series, features: List[str] = []): 
    """
    Evaluates the models adding one more feature
    Returns the best model of each iteration
    """
    remaining_features = [d for d in X_train.columns if d not in features]
    #stic = time.time()
    results=[]
    for d in remaining_features:
        results.append(processSubset(X_train, y_train, features+[d]))
    models = pd.DataFrame(results)
    best_model = models.loc[models["MSE"].argmin()]
    #toc = time.time()
    #print("Processed ", models.shape[0], "models on", len(features)+1, "features in", (toc-tic), "seconds.")
    return best_model


def backward(X_train: pd.DataFrame, y_train:pd.Series, features: List[str] = []):
    """
    Evaluates the models removing one more feature
    Returns the best model of each iteration
    """
    #tic = time.time()
    results = []
    for combo in itertools.combinations(features, len(features)-1):
        results.append(processSubset(X_train, y_train, combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['MSE'].argmin()]
    #toc = time.time()
    #print("Processed ", models.shape[0], "models on", len(features)-1, "features in", (toc-tic), "seconds.")
    return best_model


def getModels(X_train: pd.DataFrame, y_train:pd.Series, typeSelection: str = None):
    """
    Defines the best set model according to the number of properties used (in the test set)
    """
    if typeSelection == 'Backward':
        print("Backward Test")
        models_bwd = pd.DataFrame(columns=["MSE", "features"], index = range(1,len(X_train.columns)))
        tic = time.time()
        features = X_train.columns
        while(len(features) > 1):  
            models_bwd.loc[len(features)-1] = backward(X_train, y_train, features)
            features = models_bwd.loc[len(features)-1]["features"]
        toc = time.time()
        print("Total elapsed time:", (toc-tic), "seconds.")
        return models_bwd

    else: 
        print("Forward Test")
        models_fwd = pd.DataFrame(columns=["MSE", "features"])
        tic = time.time()
        features = []
        for i in range(1,len(X_train.columns)+1):
            models_fwd.loc[i] = forward(X_train, y_train, features)
            features = models_fwd.loc[i]["features"]
        toc = time.time()
        print("Total elapsed time:", (toc-tic), "seconds.")
        return models_fwd


def subsetSelection(X_train: pd.DataFrame, y_train:pd.Series, columns, typeSelection: str ='Backward'):
    """
    typeSelection accepts: 'Backward'|'Forward'

    returns the array with the features used in the best model 
    """
    X_train = pd.DataFrame(X_train, columns=[col for col in columns if col != "Price"])
    models = getModels(X_train, y_train, typeSelection)
    features = models.iloc[models['MSE'].astype('float').argmin()]['features']
    return features


def forwardSelection(columns: list, features: list = FORWARD_SELECTION_BEST_FEATURES):
    return SelectFeatures(columns, features)

def backwardSelection(columns: list, features: list = BACKWARD_SELECTION_BEST_FEATURES):
    return SelectFeatures(columns, features)
