import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
import random

from typing import Optional, Tuple, List, Union

from collections import Counter # remove_bad_cols()

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def encode_predictors_housing_data(X):
    """
    Encode predictor variables in the housing data using one-hot encoding for nominal fields.
    
    Args:
        X (pd.DataFrame): Input DataFrame containing predictor variables.

    Returns:
        pd.DataFrame: DataFrame with encoded predictor variables.
    """
    # get lists of continuous features, nominal features, etc.
    predictor_type_dict = get_feat_types()
    exclude_fields = predictor_type_dict['exclude_fields']
    nominal_fields = predictor_type_dict['nominal_fields']
    ordinal_fields = predictor_type_dict['ordinal_fields']
    dichotomous = predictor_type_dict['dichotomous']
    continuous_fields = predictor_type_dict['continuous_fields']
    
    # init list of features/variables to keep
    keep_cols=[]
 
    # to allow for subsets of the data to be processed with this function, reduce lists to predictors present in X
    exclude_fields = list(set(exclude_fields).intersection(X.columns))
    nominal_fields = list(set(nominal_fields).intersection(X.columns))
    ordinal_fields = list(set(ordinal_fields).intersection(X.columns))
    dichotomous = list(set(dichotomous).intersection(X.columns))
    continuous_fields = list(set(continuous_fields).intersection(X.columns))
    
    # continuous fields can be stored without any changes
    keep_cols.extend(continuous_fields)
    
    # add nominal fields as dummy vars
    X_enc = X.copy()
    X_enc.loc[:, nominal_fields]=X[nominal_fields].astype("category")
    one_hot = pd.get_dummies(X_enc[nominal_fields])
    keep_cols.extend(one_hot.columns)
    X_enc=X_enc.join(one_hot)
    
    # ordinal fields are skipped since they require some additional code to map different strings to different numerical values. we'll leave them out for this workshop
    
    # binary vars can be stored as numeric representations (using factorize function)
    for bin_var in dichotomous:
        if bin_var=='Street':
            new_vals, uniques = X_enc['Street'].factorize(['Grvl','Pave'])
            X_enc['Street'] = new_vals
        elif bin_var=='CentralAir':
            new_vals, uniques = X['CentralAir'].factorize(['N','Y'])
            X_enc['CentralAir'] = new_vals
        else:
            raise ValueError(('A new binary variable needs to be appropriately factorized:', bin_var))
            
    keep_cols.extend(dichotomous)
    
    # keep only these columns (continous features and one-hot encoded features) 
    X_enc=X_enc[keep_cols]
    
    return X_enc

# import pandas as pd
# import numpy as np
# from pandas.core.frame import DataFrame

def remove_bad_cols(X: Union[pd.Series, pd.DataFrame], limited_var_thresh: float) -> pd.DataFrame:
    """
    Remove variables that have NaNs as observations and vars that have a constant value across all observations.

    Args:
        X (Union[pd.Series, pd.DataFrame]): Input data as a Pandas Series or DataFrame.
        limited_var_thresh (float): Threshold for considering a column as having limited variance.

    Returns:
        pd.DataFrame: A DataFrame with bad columns removed.
    """
    if isinstance(X, pd.Series):
        X = pd.DataFrame(X, columns=[X.name])
        
    # Remove NA
    without_na = X.dropna(axis='columns')
    # Find Sparse
    sparse_cols = without_na.apply(lambda col: col.value_counts(normalize=True).max() * 100 > limited_var_thresh)
    # Remove Sparse
    without_sparse = without_na.drop(without_na.columns[sparse_cols], axis='columns')

    # Print dropped cols
    orig_cols = X.columns
    final_cols = without_sparse.columns
    dropped_cols = orig_cols[~orig_cols.isin(final_cols)]
    print(f'{len(dropped_cols)} removed, {len(final_cols)} remaining.')
    if len(dropped_cols) > 0:
        print(f'Columns removed: {list(dropped_cols)}')

    # Return
    return without_sparse


def create_zscore_feature(arr: np.ndarray) -> np.ndarray:
    """center data at 0"""
    normalized_arr = np.zeros_like(arr)
    for dim in range(arr.shape[-1]):
        data = arr[:, dim]
        normalized_arr[:, dim] = (data-np.mean(data))/np.std(data)
    return normalized_arr


# def clean_data(housing):
    
def get_feat_types():
    # Also see here for more thorough documentation regarding the feature set: 
    # https://www.openml.org/d/42165
    
    exclude_fields=['Id','MiscFeature','MiscVal'] # ID is an index variable. Removing MiscFeature and MiscVal for simplicity. These two features are always paired; would take a small amount of code to convert the combination of these vars into a set of one-hot-encoded vars.
    nominal_fields=['MSSubClass','MSZoning','Alley','LandContour',
                   'LotConfig','Neighborhood','Condition1','Condition2',
                   'BldgType','HouseStyle','RoofStyle','RoofMatl',
                   'Exterior1st','Exterior2nd','MasVnrType','Foundation', 
                   'Heating','Electrical','GarageType','MiscFeature',
                   'SaleType','SaleCondition','Utilities']
    ordinal_fields=['LotShape','LandSlope','ExterQual','ExterCond', 
                   'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                   'BsmtFinType2','HeatingQC','KitchenQual','Functional',
                   'FireplaceQu','GarageFinish','GarageQual','GarageCond',
                   'PavedDrive','PoolQC','Fence']
    dichotomous=['Street','CentralAir']
    continuous_fields=['LotFrontage','LotArea','YearBuilt','YearRemodAdd',
                      'OverallQual','OverallCond','MasVnrArea','BsmtFinSF1',  
                      'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',
                      '2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
                      'BsmtHalfBath','FullBath','HalfBath','KitchenAbvGr','BedroomAbvGr',
                      'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars',
                      'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
                      '3SsnPorch','ScreenPorch','PoolArea','YrSold','MoSold']
    
    predictor_type_dict = {}
    predictor_type_dict['exclude_fields']=exclude_fields
    predictor_type_dict['nominal_fields']=nominal_fields
    predictor_type_dict['ordinal_fields']=ordinal_fields
    predictor_type_dict['dichotomous']=dichotomous
    predictor_type_dict['continuous_fields']=continuous_fields

    return predictor_type_dict

def plot_dist_before_after(before: pd.DataFrame, after: pd.DataFrame, column: str) -> None:
    """show a columns distribution before and after z scoring"""
    # choose column
    column = '1stFlrSF'

    # plot
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    plot_configs = zip(
        axs, 
        [before, after], 
        ["blue", "orange"], 
        [f"original {column}", f"z-scored {column}"]
    ) 
    for (ax, df, color, title) in plot_configs:
        ax.hist(df[column], color=color)
        ax.set_title(title) 

    plt.show()


def zscore(df: pd.DataFrame, train_means: pd.Series, train_stds: pd.Series) -> pd.DataFrame:
    """return z-scored dataframe using training data mean and standard deviation"""
    return (df - train_means) / train_stds

def generate_combinations(items: List[str], K: int) -> List[tuple]:
    """
    Generate all combinations of K items from the given list of items.

    Args:
        items (List[str]): List of items to choose combinations from.
        K (int): Number of items in each combination.

    Returns:
        List[tuple]: A list of tuples representing the combinations.
    """
    return list(itertools.combinations(items, K))

def get_predictor_combos(X_train: pd.DataFrame, K: int, n: int) -> List[List[str]]:
    """
    Get sampled predictor variable combinations from the training data.

    Args:
        X_train (pd.DataFrame): Training feature data.
        K (int): Number of columns in each combination.
        n (int): Number of combinations to sample.

    Returns:
        List[List[str]]: A list of lists representing the sampled predictor variable combinations.
    """
    X_train_columns = list(X_train.columns)
    
    all_combinations = generate_combinations(X_train_columns, K)
    sampled_combinations = random.sample(all_combinations, min(n, len(all_combinations)))
    sampled_combinations = [list(combo) for combo in sampled_combinations]  # Convert to list of lists

    return sampled_combinations


def prep_fulldim_zdata(const_thresh: int = 98, test_size: float = 0.33, y_log_scaled: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares the housing dataset for high-dimensional regression analysis.

    Args:
        const_thresh (int, optional): Threshold for removing columns with low variance. Default is 98.
        test_size (float, optional): Percentage of the data to use for testing. Default is 0.33.
        y_log_scaled (bool, optional): Whether the target values are log-scaled or not. Default is True.

    Returns:
        X_train_z (pd.DataFrame): Z-scored training predictor variables.
        X_test_z (pd.DataFrame): Z-scored testing predictor variables.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Testing target values.
        y_raw (pd.Series): Raw target values before any scaling or transformation.
    """
    # Load housing dataset
    housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')
    y_raw = housing['target']
    X = housing['data']
    if y_log_scaled:
        y = np.log(y_raw)
    else:
        y = y_raw
    # Encode categorical data
    X_encoded = encode_predictors_housing_data(X)

    # Remove columns with low variance
    X_encoded_good = remove_bad_cols(X_encoded, const_thresh)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded_good, y, test_size=test_size, random_state=0)

    # Z-score predictors
    X_train_z = zscore(df=X_train, train_means=X_train.mean(), train_stds=X_train.std())
    X_test_z = zscore(df=X_test, train_means=X_train.mean(), train_stds=X_train.std())

    return X_train_z, X_test_z, y_train, y_test, y_raw
