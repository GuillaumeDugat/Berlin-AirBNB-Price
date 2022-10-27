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

"""
Returns a dict with the features applied to a Linear Regression Model with k-folding  
and the mean of the mean squared error applied to each model.
"""
def processSubset(X_train:pd.DataFrame, y_train:pd.Series, feature_set: List[str]):
    
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    model = LinearRegression()
    score = -cross_val_score(model, X_train[list(feature_set)], y_train, scoring='neg_mean_squared_error', cv=cv)
    med_score = math.sqrt(np.mean(score))

    return {"features": feature_set, "MSE": med_score}

"""
Evaluates the models adding one more feature
Returns the best model of each iteration
"""
def forward(X_train: pd.DataFrame, y_train:pd.Series, features: List[str] = []): 
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


"""
Evaluates the models removing one more feature
Returns the best model of each iteration
"""
def backward(X_train: pd.DataFrame, y_train:pd.Series, features: List[str] = []):
    
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


"""
Defines the best set model according to the number of properties used (in the test set)
"""
def getModels(X_train: pd.DataFrame, y_train:pd.Series, typeSelection: str = None):
    
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


"""
typeSelection accepts: 'Backward'|'Forward'

returns the array with the features used in the best model 
"""
def subsetSelection(X_train: pd.DataFrame, y_train:pd.Series, typeSelection: str ='Backward'):

    models = getModels(X_train, y_train, typeSelection)
    features = models.iloc[models['MSE'].astype('float').argmin()]['features']
    return features
