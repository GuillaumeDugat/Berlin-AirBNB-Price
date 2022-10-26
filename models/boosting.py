from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
CROSS_VALIDATION = 5
VERBOSE = 3

class XGBRegressorCustom(XGBRegressor):
    """Need to LabelEncoder on y in fit, to prevent an error"""

    def fit(self, X, y):
        le = LabelEncoder()
        return super().fit(X, le.fit_transform(y))



def create_ada_boost():
    ADBregr = AdaBoostRegressor(random_state = RANDOM_STATE)
    pgrid = {
        'base_estimator': [DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=3), DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5)],
        'n_estimators': [4,8,15,30,50],
        'learning_rate': [0.1], #[0.001, 0.005, 0.01, 0.05, 0.1],
        'loss': ['linear'],
    }
    return GridSearchCV(ADBregr, param_grid=pgrid, scoring='neg_mean_squared_error', cv=CROSS_VALIDATION, verbose=VERBOSE)

def create_gradient_boosted_tree():
    GDBreg  = GradientBoostingRegressor(random_state = RANDOM_STATE)
    pgrid = {
        'n_estimators': [4,8,15,30,50, 100],
        'learning_rate': [0.1], #[0.001, 0.01, 0.05, 0.1, 0.5, 1],
        'max_depth': [1, 3, 5],
        'loss': ['squared_error'], #['squared_error', 'absolute_error', 'huber', 'quantile'],
    }
    return GridSearchCV(GDBreg, param_grid=pgrid, scoring='neg_mean_squared_error', cv=CROSS_VALIDATION, verbose=VERBOSE)

def create_xg_boost():
    ADBregr  = XGBRegressorCustom(random_state = RANDOM_STATE)
    pgrid = {
        'n_estimators': [5, 10, 50, 100], #[500, 750, 1000],
        'learning_rate': [0.1], #[0.01, 0.05, 0.1],
        'max_depth': [2, 3],
    }
    return GridSearchCV(ADBregr, param_grid=pgrid, scoring='neg_mean_squared_error', cv=CROSS_VALIDATION, verbose=VERBOSE)
