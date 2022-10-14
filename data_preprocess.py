from cmath import pi
from venv import create
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from math import sqrt
import numpy as np

from datetime import datetime


def convert_date(date_str: str) -> int:
 
    date_compare = datetime(2022, 10, 13)
    date_to_compare = datetime.strptime(date_str, '%Y-%m-%d')
    comparison = (date_compare - date_to_compare).days

    return comparison


class DropNan(BaseEstimator, TransformerMixin):

    def __init__(self, keep_columns):
        super().__init__()
        self.keep_columns = keep_columns
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):

        X = X[self.keep_columns]
        X = X.replace('*', np.nan)
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
            X[column] = X[column].apply(convert_date)

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

def create_pipeline():

    location_dict = {
        'ALEXANDERPLATZ' : (52.5229547506655, 13.415715439885059),
        'BRANDEBOURG' : (52.51696974072543, 13.37789650882256),
        'GENDARMENMARKT' : (52.515477165623075, 13.398549665529353)
    }

    columns_to_keep = ['Host Since', 'Is Superhost', 'Neighborhood Group', 'Latitude', 
    'Longitude', 'Property Type', 'Room Type', 'Accomodates', 'Bathrooms', 'Bedrooms', 'Beds',
    'Guests Included', 'Min Nights', 'Reviews', 'Instant Bookable', 'Price']

    columns_to_encode = ['Neighborhood Group', 'Property Type', 'Room Type']

    columns_dates = ['Host Since']

    columns_boolean = ['Is Superhost', 'Instant Bookable']

    pipeline_to_return = Pipeline(steps= [
        ('dropnan', DropNan(columns_to_keep)),
        ('onehot', GetDummies(columns_to_encode)),
        ('dates', TransformDate(columns_dates)),
        ('boolean', TransformBoolean(columns_boolean)),
        ('location', TransformDistance(location_dict))
    ])

    return pipeline_to_return

    
if __name__ == '__main__':

    raw_dataframe = pd.read_csv('train_airbnb_berlin.csv')

    new_pipeline = create_pipeline()

    processed_dataframe = new_pipeline.fit_transform(raw_dataframe)
    for column in processed_dataframe.columns:
        print(column)