import time
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Style options for plots.
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import pandas as pd
import statsmodels.api as sm
import itertools

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import data_preprocess


"""
Returns a dict with the resulting model of applying a OLS (Ordinary Least Square Regression) 
and the RSS (Residual Sum of Squares) of the model.
"""
def processSubset(X:pd.DataFrame, y:pd.Series, feature_set: List[str]):
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)])-y)**2).sum()
    return {"model": regr, "RSS": RSS}

"""
Evaluates the models adding one more feature
Returns the best model of each iteration
"""
def forward(X: pd.DataFrame, y:pd.Series, features: List[str] = []): 
    remaining_features = [d for d in X.columns if d not in features]
    #stic = time.time()
    results=[]
    for d in remaining_features:
        results.append(processSubset(X,y,features+[d]))
    models = pd.DataFrame(results)
    best_model = models.loc[models["RSS"].argmin()]
    #toc = time.time()
    #print("Processed ", models.shape[0], "models on", len(features)+1, "features in", (toc-tic), "seconds.")
    return best_model


"""
Evaluates the models removing one more feature
Returns the best model of each iteration
"""
def backward(X: pd.DataFrame, y:pd.Series, features: List[str] = []):
    
    #tic = time.time()
    results = []
    for combo in itertools.combinations(features, len(features)-1):
        results.append(processSubset(X,y,combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    #toc = time.time()
    #print("Processed ", models.shape[0], "models on", len(features)-1, "features in", (toc-tic), "seconds.")
    return best_model


"""
Defines the best set model according to the number of properties used (in the test set)
"""
def getModels(X: pd.DataFrame, y:pd.Series, typeSelection: str = None):
    
    if typeSelection == 'Backward':
        models_bwd = pd.DataFrame(columns=["RSS", "model"], index = range(1,len(X.columns)))
        tic = time.time()
        features = X.columns
        while(len(features) > 1):  
            models_bwd.loc[len(features)-1] = backward(X, y, features)
            features = models_bwd.loc[len(features)-1]["model"].model.exog_names
        toc = time.time()
        print("Total elapsed time:", (toc-tic), "seconds.")
        return models_bwd

    else: 
        models_fwd = pd.DataFrame(columns=["RSS", "model"])
        tic = time.time()
        features = []
        for i in range(1,len(X.columns)+1):
            models_fwd.loc[i] = forward(X, y, features)
            features = models_fwd.loc[i]["model"].model.exog_names
        toc = time.time()
        print("Total elapsed time:", (toc-tic), "seconds.")
        return models_fwd

    

def plotErrors(rss, rsquared_adj, aic, bic):
    
    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

    # Set up a 2x2 grid so we can look at 4 plots at once
    plt.subplot(2, 2, 1)

    # We will now plot a red dot to indicate the model with the lowest RSS value.
    # The argmax() function can be used to identify the location of the minimum point of a vector.
    plt.plot(rss)
    plt.plot(rss.argmin(), rss.min(), "or")
    plt.xlabel('# Features')
    plt.ylabel('RSS')

    # We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
    # The argmax() function can be used to identify the location of the maximum point of a vector.
    plt.subplot(2, 2, 2)
    plt.plot(rsquared_adj)
    plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "or")
    plt.xlabel('# Features')
    plt.ylabel('adjusted rsquared')

    # We'll do the same for AIC and BIC, this time looking for the models with the SMALLEST statistic
    plt.subplot(2, 2, 3)
    plt.plot(aic)
    plt.plot(aic.argmin(), aic.min(), "or")
    plt.xlabel('# Features')
    plt.ylabel('AIC')

    plt.subplot(2, 2, 4)
    plt.plot(bic)
    plt.plot(bic.argmin(), bic.min(), "or")
    plt.xlabel('# Features')
    plt.ylabel('BIC')


"""
typeSelection accepts: 'Backward'|'Forward'
preprocess in order to normalize and clean the data
plot to show the graphicos of errors

returns the best model for each error 
"""
def dimensionalReduction(typeSelection: str ='Backward', preprocess: bool=True, plot: bool=False):
    
    X_train, X_test, y_train, y_test, column = data_preprocess.get_processed_train_test('./data/', preprocess)
    col = column[column != 'Price']
    X_train = pd.DataFrame(X_train, columns=col)
    y_train = pd.Series(y_train)
    models = getModels(X=X_train, y=y_train, typeSelection=typeSelection)

    #Type of error used
    errorEvaluated = ["RSS", "RsqAdj", "AIC", "BIC"]
    rss = models["RSS"].astype('float')
    rsquared_adj = models.apply(lambda row: row[1].rsquared_adj, axis=1)
    aic = models.apply(lambda row: row[1].aic, axis=1)
    bic = models.apply(lambda row: row[1].bic, axis=1)
    aux = [rss, rsquared_adj, aic, bic]

    if plot:
        plotErrors(rss,rsquared_adj,aic,bic)

    #We chose the model for each optimization
    chosenModels = []
    chosenModels.append(models.loc[rss.argmin()].model)
    chosenModels.append(models.loc[rsquared_adj.argmax()].model)
    chosenModels.append(models.loc[aic.argmin()].model)
    chosenModels.append(models.loc[bic.argmin()].model)

    features = []
    for model in chosenModels:
        features.append(model.model.exog_names)

    bestModels = pd.DataFrame({"typeError": errorEvaluated, "model": chosenModels, "features": features})
    return bestModels

