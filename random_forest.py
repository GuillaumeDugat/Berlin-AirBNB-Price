from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from data_preprocess import get_processed_train_test



X_train, X_test, Y_train, Y_test, _ = get_processed_train_test(path_to_folder='data', \
                                                               add_processing=True)

def random_forest_gridsearch(X_train, X_test, Y_train, Y_test):
    regressor = RandomForestRegressor()
    pgrid = {
        'n_estimators': [10, 30, 100, 300, 1000],
        'max_features': ['sqrt', 'log2', 1.0],
        'max_samples': [0.1, 0.5, None],
        'min_samples_leaf': [1, 3, 10, 30],
        'oob_score': [True, False]

    }
    grid_search = GridSearchCV(regressor, param_grid=pgrid, scoring='neg_mean_squared_error', cv=5, verbose=3)
    grid_search.fit(X_train, Y_train)

    print('Best parameters:', grid_search.best_params_)
    print('MSE on test set:', -grid_search.score(X_test, Y_test))

random_forest_gridsearch(X_train, X_test, Y_train, Y_test)
# Best parameters: {'max_features': 1.0, 'max_samples': 0.1, 'min_samples_leaf': 3, 'n_estimators': 300, 'oob_score': True}
# MSE on test set: 1350.9333713323117


