from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
CROSS_VALIDATION = 5
VERBOSE = 3
N_ITER = 30

BEST_PARAMS_ADA_BOOST = {'n_estimators': 30, 'loss': 'square', 'learning_rate': 0.001, 'base_estimator': DecisionTreeRegressor(max_depth=5, random_state=42)}
BEST_PARAMS_GRADIENT_BOOSTED_TREE = {'n_estimators': 50, 'max_depth': 5, 'loss': 'huber', 'learning_rate': 0.05}
BEST_PARAMS_XG_BOOST = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}

class XGBRegressorCustom(XGBRegressor):
    """Need to LabelEncoder on y in fit, to prevent an error"""

    def fit(self, X, y):
        le = LabelEncoder()
        return super().fit(X, le.fit_transform(y))



def create_ada_boost(best_model: bool=False):
    if best_model:
        return AdaBoostRegressor(random_state = RANDOM_STATE, **BEST_PARAMS_ADA_BOOST)
    ADBregr = AdaBoostRegressor(random_state = RANDOM_STATE)
    pgrid = {
        'base_estimator': [
            DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=3),
            DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=4),
            DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5)],
        'n_estimators': [4, 8, 15, 30, 50, 80],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 1],
        'loss': ['linear', 'square', 'exponential'],
    }
    return RandomizedSearchCV(ADBregr, param_distributions=pgrid, scoring='neg_mean_squared_error', n_iter=N_ITER, cv=CROSS_VALIDATION, verbose=VERBOSE)

def create_gradient_boosted_tree(best_model: bool=False):
    if best_model:
        return GradientBoostingRegressor(random_state = RANDOM_STATE, **BEST_PARAMS_GRADIENT_BOOSTED_TREE)
    GDBreg  = GradientBoostingRegressor(random_state = RANDOM_STATE)
    pgrid = {
        'n_estimators': [4, 8, 15, 30, 50, 80],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 1],
        'max_depth': [2, 3, 5],
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    }
    return RandomizedSearchCV(GDBreg, param_distributions=pgrid, scoring='neg_mean_squared_error', n_iter=N_ITER, cv=CROSS_VALIDATION, verbose=VERBOSE)

def create_xg_boost(best_model: bool=False):
    if best_model:
        return XGBRegressorCustom(random_state = RANDOM_STATE, **BEST_PARAMS_XG_BOOST)
    ADBregr  = XGBRegressorCustom(random_state = RANDOM_STATE)
    pgrid = {
        'n_estimators': [4, 8, 15, 30, 50, 80, 100, 250, 500],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 1],
        'max_depth': [2, 3, 5],
    }
    return RandomizedSearchCV(ADBregr, param_distributions=pgrid, scoring='neg_mean_squared_error', n_iter=N_ITER, cv=CROSS_VALIDATION, verbose=VERBOSE)
