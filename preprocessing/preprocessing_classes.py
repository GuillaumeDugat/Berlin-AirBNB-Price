from math import sqrt
from cmath import pi
import numpy as np

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from utils import convert_date


class KeepColumns(BaseEstimator, TransformerMixin):

    def __init__(self, keep_columns):
        super().__init__()
        self.keep_columns = keep_columns
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):

        X = X[self.keep_columns]
        X = X.replace('*', np.nan)

        return X

class FeatureImputing(BaseEstimator, TransformerMixin):

    def __init__(self, kwargs_dict):
        super().__init__()
        self.imputer_dict = {col: SimpleImputer(**kwargs_dict[col]) for col in kwargs_dict}
    
    def fit(self, X: pd.DataFrame, y=None):
        for col in self.imputer_dict:
            self.imputer_dict[col].fit(X[[col]])
        return self

    def transform(self, X: pd.DataFrame, y=None):
        for col in self.imputer_dict:
            X[col] = self.imputer_dict[col].transform(X[[col]])
        return X

class DropNan(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.dropna()
        return X

class GetDummies(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_encode):
        super().__init__()
        self.columns_to_encode = columns_to_encode
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None):

        return pd.get_dummies(X, columns=self.columns_to_encode)

class TransformDate(BaseEstimator, TransformerMixin):

    def __init__(self, columns_dates):
        super().__init__()
        self.columns_dates = columns_dates
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None):

        for column in self.columns_dates:
            X.loc[X[column].notnull(), column] = X.loc[X[column].notnull(), column].apply(convert_date)

        return X

class TransformDistance(BaseEstimator, TransformerMixin):

    def __init__(self, location_dict):
        super().__init__()
        self.location_dict = location_dict
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None):

        for location in self.location_dict:
            x_l, y_l = self.location_dict[location]
            distance = lambda x : sqrt((x_l - x['Latitude'])**2 + (y_l - x['Longitude'])**2)
            name_column = 'Dist_'+location
            X[name_column] = X.apply(distance, axis=1) / 360 * 2 * pi * 6371
            
        return X

class TransformBoolean(BaseEstimator, TransformerMixin):

    def __init__(self, columns_boolean):
        super().__init__()
        self.columns_boolean = columns_boolean

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None):

        for column in self.columns_boolean:
            X[column] = X[column].apply( lambda x : 0 if x == 'f' else 1)
        
        return X

class TransformStrings(BaseEstimator, TransformerMixin):

    def __init__(self, columns_float, columns_int):
        super().__init__()
        self.columns_float = columns_float
        self.columns_int = columns_int
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        for column in self.columns_float:
            X[column] = X[column].astype(float)
        
        for column in self.columns_int:
            X[column] = X[column].astype(float).astype(int)
        
        return X

class SelectFeatures(BaseEstimator, TransformerMixin):
    """Once converted to numpy array, it is more difficult to 
    select certain features on X_train and X_test, this class allows to do it easily"""

    def __init__(self, all_columns, columns_to_keep):
        super().__init__()
        self.columns_to_keep = "" # To prevent bug in __str__
        self.all_columns = "" # To prevent bug in __str__
        self.selection_col = np.isin([col for col in all_columns if col != "Price"], columns_to_keep) # Select the specified columns (last one of all_columns is y)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[:, self.selection_col]
