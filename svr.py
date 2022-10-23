from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from time import time

from data_preprocess import get_processed_train_test



X_train, X_test, Y_train, Y_test, _ = get_processed_train_test(path_to_folder='data', \
                                                               add_processing=True)
# importance de scaler (avec add_processing=True) pour avoir un temps de convergence raisonnable
# (sans ça c'est vraiment très très long)

def SVR_prediction(X_train, X_test, Y_train, Y_test):
    for C in [10**(-3), 10**(-2), 10**(-1), 10**0, 10**1, 10**2]:
        debut=time()
        regressor = SVR(kernel = 'linear',gamma='scale',C=C, max_iter=10**6)
        regressor.fit(X_train, Y_train)
        predictions = regressor.predict(X_test)
        mse = mean_squared_error(Y_test, predictions)
        print('C:',C,', MSE:', mse,', time:',time()-debut)

#SVR_prediction(X_train, X_test, Y_train, Y_test)


def SVR_gridsearch(X_train, X_test, Y_train, Y_test):
    regressor = SVR()
    pgrid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'C': [10**(-3), 10**(-2), 10**(-1), 10**0, 10**1, 10**2, 10**3]
    }
    grid_search = GridSearchCV(regressor, param_grid=pgrid, scoring='neg_mean_squared_error', cv=5, verbose=3)
    grid_search.fit(X_train, Y_train)

    print('Best parameters:', grid_search.best_params_)
    print('MSE on test set:', -grid_search.score(X_test, Y_test))

# SVR_gridsearch(X_train, X_test, Y_train, Y_test)  
# Best parameters: {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}
# MSE on test set: 1428.7437281364225



def SVR_gridsearch2(X_train, X_test, Y_train, Y_test):
    regressor = SVR(C=100, gamma='scale', kernel='rbf')  
    pgrid = {
        'tol': [10**(-4), 3*10**(-4), 10**(-3), 3*10**(-3), 10**(-2)],
        'epsilon':[0.01, 0.03, 0.1, 0.3, 1],
        'shrinking':[True,False]
    }
    grid_search = GridSearchCV(regressor, param_grid=pgrid, scoring='neg_mean_squared_error', cv=5, verbose=3)
    grid_search.fit(X_train, Y_train)

    print('Best parameters:', grid_search.best_params_)
    print('MSE on test set:', -grid_search.score(X_test, Y_test))

SVR_gridsearch2(X_train, X_test, Y_train, Y_test)
# Best parameters: {'epsilon': 1, 'shrinking': True, 'tol': 0.0003}
# MSE on test set: 1427.2103958031676


# Conclusion : meilleurs paramètres pour SVR : 
# C=100, gamma='scale', kernel='rbf', epsilon=1, shrinking=True, tol=0.0003
# qui donnent une MSE sur le test set de 1427.2103958031676