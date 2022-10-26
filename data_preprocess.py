import os
from math import sqrt
from cmath import pi

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


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

def create_pipeline():

    location_dict = {
        'ALEXANDERPLATZ' : (52.5229547506655, 13.415715439885059),
        'BRANDEBOURG' : (52.51696974072543, 13.37789650882256),
        'GENDARMENMARKT' : (52.515477165623075, 13.398549665529353)
    }

    columns_to_keep = ['Host Since', 'Is Superhost', 'Neighborhood Group', 'Latitude', 
    'Longitude', 'Property Type', 'Room Type', 'Accomodates', 'Bathrooms', 'Bedrooms', 'Beds',
    'Guests Included', 'Min Nights', 'Reviews', 'Instant Bookable', 'Price']

    imputing_dict = {
        'Host Since': {'strategy': 'mean'},
        'Is Superhost': {'strategy': 'most_frequent'},
        'Property Type': {'strategy': 'most_frequent'},
        'Room Type': {'strategy': 'most_frequent'},
        'Accomodates': {'strategy': 'mean'},
        'Bathrooms': {'strategy': 'mean'},
        'Bedrooms': {'strategy': 'mean'},
        'Beds': {'strategy': 'mean'},
        'Guests Included': {'strategy': 'mean'},
        'Min Nights': {'strategy': 'mean'},
    }

    columns_to_encode = ['Neighborhood Group', 'Property Type', 'Room Type']

    columns_dates = ['Host Since']

    columns_boolean = ['Is Superhost', 'Instant Bookable']

    columns_float = ['Bathrooms']
    columns_int = ['Accomodates', 'Bedrooms', 'Beds', 'Guests Included', 'Min Nights']

    pipeline_to_return = Pipeline(steps= [
        ('keep_columns', KeepColumns(columns_to_keep)),
        ('dates', TransformDate(columns_dates)),
        # ('imputing', FeatureImputing(imputing_dict)),
        ('dropnan', DropNan()),
        ('onehot', GetDummies(columns_to_encode)),
        ('boolean', TransformBoolean(columns_boolean)),
        ('location', TransformDistance(location_dict)),
        ('strings', TransformStrings(columns_float, columns_int))
    ])

    return pipeline_to_return

def create_test_split(
    path_to_csv: str='data/raw/train_airbnb_berlin.csv',
    test_size: float=0.2,
    random_seed: int=42
    ):

    df_raw = pd.read_csv(path_to_csv)

    # Create test sample
    df_sampled = df_raw.sample(frac=test_size, random_state=random_seed)

    # Retrieve train split
    rest_index = [k for k in df_raw.index if k not in df_sampled.index]
    df_rest = df_raw.iloc[rest_index]
    
    # Reset index
    df_sampled = df_sampled.reset_index(drop=True)
    df_rest = df_rest.reset_index(drop=True)

    # Save to csv
    df_sampled.to_csv('data/split_to_use/train.csv')
    df_rest.to_csv('data/split_to_use/test.csv')


def get_processed_train_test(path_to_folder: str='data', add_processing: bool=False, remove_outliers: bool=True) -> list:
    
    path_train = os.path.join(path_to_folder, 'train.csv')
    path_test = os.path.join(path_to_folder, 'test.csv')

    train_df = pd.read_csv(path_train)
    if remove_outliers: #99% of prices are below this point
        train_df = train_df[train_df['Price'] <= 250]
    test_df = pd.read_csv(path_test)

    processing_pipeline = create_pipeline()

    train_processed_df = processing_pipeline.fit_transform(train_df)
    test_processed_df = processing_pipeline.transform(test_df)

    columns = np.array(train_processed_df.columns)

    X_train = train_processed_df.loc[:, train_processed_df.columns != 'Price']
    Y_train = train_processed_df[['Price']].to_numpy().reshape(X_train.shape[0])

    X_test = test_processed_df.loc[:, test_processed_df.columns != 'Price']
    Y_test = test_processed_df[['Price']].to_numpy().reshape(X_test.shape[0])


    if add_processing:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
    
    return [X_train, X_test, Y_train, Y_test, columns]


if __name__ == '__main__':

    create_test_split()

    X_train, X_test, Y_train, Y_test = get_processed_train_test()