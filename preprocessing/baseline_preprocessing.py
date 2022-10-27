import os

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from preprocessing.preprocessing_classes import (
    KeepColumns, FeatureImputing, DropNan, GetDummies, TransformDate,
    TransformDistance, TransformBoolean, TransformStrings
)



def create_pipeline(imputing_missing_values):

    location_dict = {
        'ALEXANDERPLATZ' : (52.5229547506655, 13.415715439885059),
        'BRANDEBOURG' : (52.51696974072543, 13.37789650882256),
        'GENDARMENMARKT' : (52.515477165623075, 13.398549665529353)
    } # Points of interest selected on the map of Berlin

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

    steps= [
        ('keep_columns', KeepColumns(columns_to_keep)),
        ('dates', TransformDate(columns_dates)),
        # ('imputing', FeatureImputing(imputing_dict)),
        ('dropnan', DropNan()),
        ('onehot', GetDummies(columns_to_encode)),
        ('boolean', TransformBoolean(columns_boolean)),
        ('location', TransformDistance(location_dict)),
        ('strings', TransformStrings(columns_float, columns_int))
    ]
    if imputing_missing_values:
        steps.insert(2, ('imputing', FeatureImputing(imputing_dict)))

    pipeline_to_return = Pipeline(steps=steps)

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
    df_rest.to_csv('data/train.csv')
    df_sampled.to_csv('data/test.csv')


def get_baseline_preprocessed_train_test(
    path_to_folder: str='data',
    remove_outliers: bool=True,
    imputing_missing_values: bool=False
) -> list:
    
    path_train = os.path.join(path_to_folder, 'train.csv')
    path_test = os.path.join(path_to_folder, 'test.csv')

    train_df = pd.read_csv(path_train)
    if remove_outliers: #99% of prices are below this point
        train_df = train_df[train_df['Price'] <= 250]
    test_df = pd.read_csv(path_test)

    processing_pipeline = create_pipeline(imputing_missing_values)

    train_processed_df = processing_pipeline.fit_transform(train_df)
    test_processed_df = processing_pipeline.transform(test_df)

    columns = np.array(train_processed_df.columns)

    X_train = train_processed_df.loc[:, train_processed_df.columns != 'Price']
    Y_train = train_processed_df[['Price']].to_numpy().reshape(X_train.shape[0])

    X_test = test_processed_df.loc[:, test_processed_df.columns != 'Price']
    Y_test = test_processed_df[['Price']].to_numpy().reshape(X_test.shape[0])

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    return [X_train, X_test, Y_train, Y_test, columns]


if __name__ == '__main__':
    create_test_split()
