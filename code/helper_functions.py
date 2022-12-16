import numpy as np
import pandas as pd


# def clean_data(housing):
    
def encode_predictors_housing_data(X):
    # get lists of continuous features, nominal features, etc.
    exclude_fields, nominal_fields, ordinal_fields, dichotomous, continuous_fields = get_feat_types()
    
    # init list of features/variables to keep
    keep_cols=[]
 
    # add nominal fields as dummy vars
    X[nominal_fields]=X[nominal_fields].astype("category")
    dummy_vars=pd.get_dummies(X[nominal_fields])
    keep_cols.extend(dummy_vars.columns)
    X=X.join(dummy_vars)

    # continuous fields can be stored without any changes
    keep_cols.extend(continuous_fields)
    
    # ordinal fields are skipped since they require some additional code to map different strings to different numerical values
    X=X
    
    # binary vars can be stored as numeric representations (using factorize function)
    for bin_var in dichotomous:
        if bin_var=='Street':
            new_vals, uniques = X['Street'].factorize(['Grvl','Pave'])
            X['Street'] = new_vals
        elif bin_var=='CentralAir':
            new_vals, uniques = X['CentralAir'].factorize(['N','Y'])
            X['CentralAir'] = new_vals
        else:
            raise ValueError(('A new binary variable needs to be appropriately factorized:', bin_var))
            
    keep_cols.extend(dichotomous)
#     dummy_vars=pd.get_dummies(X[dichotomous])
#     keep_cols.extend(dummy_vars.columns)
#     X=X.join(dummy_vars)
    
    # keep only these columns (continous features and one-hot encoded features) 
    X=X[keep_cols]
    
    return X

def remove_bad_cols(X, limited_var_thresh):
    # Remove variables that have NaNs as observations and vars that have a constant value across all observations
    all_feats=X.columns
    rem_cols=[]
    for feat_index in range(0,len(all_feats)):
        feat_name = X.columns[feat_index]
        this_X = np.array(X.loc[:,feat_name]).reshape(-1, 1) # in general, it is safer (more stable) to index by column names rather than by row/col numbers
        sum_nans = np.sum(np.isnan(this_X)) # sum up nans present in column/feature
        unique_vals = np.unique(this_X) # sum up number of unique possible values for this column/feature
        val_counts = X[feat_name].value_counts(normalize=True) # check for nearly constant columns

        # exclude column if there are any NaNs or if column contains a constant (1 unique value only)
        if sum_nans > 0: 
            rem_cols.append(feat_name)
            print(feat_name + ' removed due to presence of NaNs (sum of nans = ' + str(sum_nans) + ')')
        elif sum(val_counts > limited_var_thresh):
            rem_cols.append(feat_name)
            print(feat_name + ' removed due to lack of variance ( >' + str(limited_var_thresh*100) + '% rows have the same value value)')
            
    print('All columns removed:', rem_cols)
    print('# of columns removed:', len(rem_cols))

    X=X.drop(rem_cols, axis = 1)
    
    return X

def get_feat_types():
    # Also see here for more thorough documentation regarding the feature set: 
    # https://www.openml.org/d/42165

    # OverallQual: Rates the overall material and finish of the house
    # GrLivArea: Above grade (ground) living area square feet
    # GarageCars: Size of garage in car capacity
    # GarageArea: Size of garage in square feet
    # TotalBsmtSF: Total square feet of basement area
    # 1stFlrSF: First Floor square feet
    # FullBath: Full bathrooms above grade (i.e., not in the basement)
    # TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    # YearBuilt: Original construction date
    # YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
    # data['MSZoning'] # nominal variable with 5 different general zoning options: https://www.openml.org/d/42165
    # data['MSSubClass'] # nominal variable with 15 different types of dwellings
    # data['Street'].unique() Grvl or Pave
    # 'LotShape' # ordinal variable; map as
    #     Reg Regular=1
    #     IR1 Slightly irregular=2
    #     IR2 Moderately Irregular=3
    #     IR3 Irregular=4
    # data['LandContour'] # nominal variable with 4 different categories
    # data['Utilities'] # could code as ordinal variable with 4 different levels:
    #     AllPub All public Utilities (E,G,W,& S) - 4
    #     NoSewr Electricity, Gas, and Water (Septic Tank) - 3
    #     NoSeWa Electricity and Gas Only - 2
    #     ELO Electricity only - 1
    # data['LotConfig'] # nominal variable with 5 different categories
    # data['LandSlope'] # ordinal variable with 3 different levels
    #     1=Gtl Gentle slope
    #     2=Mod Moderate Slope
    #     3=Sev Severe Slope
    
    # data['Neighborhood'] # nominal variable with 25 different categories
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
    
    return exclude_fields, nominal_fields, ordinal_fields, dichotomous, continuous_fields


# PCA
def my_PCA(X: pd.DataFrame, variance_thresh: float) -> pd.DataFrame:
    raise NotImplementedError    
    
    
