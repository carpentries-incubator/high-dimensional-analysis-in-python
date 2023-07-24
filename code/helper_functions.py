import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib


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

# regression 
import matplotlib.pyplot as plt
def plot_model_predictions(predictor,
                           x_train, x_test,
                           y_train, y_test,
                           y_pred_train, y_pred_test,logTransformY,
                           err_type=None,train_err=None,test_err=None):
    
    if type(y_train) != 'numpy.ndarray':
        y_train=np.asarray(y_train)
    if type(y_test) != 'numpy.ndarray':
        y_test=np.asarray(y_test)
    if type(y_pred_train) != 'numpy.ndarray':
        y_pred_train=np.asarray(y_pred_train)
    if type(y_pred_test) != 'numpy.ndarray':
        y_pred_test=np.asarray(y_pred_test)
        
    # get min and max y values
    all_y = np.concatenate((y_train, y_test, y_pred_train, y_pred_test), axis=0)
    if logTransformY:
        min_y = 11#np.percentile(all_y, 1)
        max_y = 13.6#np.percentile(all_y, 100)
        tick_dist = .5
    else:
        min_y = 50000#np.percentile(all_y, 2)
        max_y = 350001#np.percentile(all_y, 95)
        tick_dist = 50000
    
    # Fig1. True vs predicted sale price
    fig1, (ax1, ax2) = plt.subplots(1,2)#, sharex=True, sharey=True)
    if logTransformY:
        fig1.suptitle('True vs. Predicted log(Sale_Price)')
    else:
        fig1.suptitle('True vs. Predicted Sale Price')

    #train set
    ax1.scatter(y_train, y_pred_train, alpha=.1) 

    if train_err is not None:
        ax1.title.set_text('Train ' + err_type + ' = ' + str(round(train_err,2)))
    else:
        ax1.title.set_text('Train Data')
    ax1.set_xlabel('True')
    ax1.set_ylabel('Predicted')
    ax1.set_aspect('equal')
    # ax1.axis('equal')
    ax1.set_xlim([min_y, max_y])
    ax1.set_ylim([min_y, max_y])
    ax1.plot([0, 1], [0, 1], transform=ax1.transAxes)
    ax1.xaxis.set_ticks(np.arange(min_y, max_y, tick_dist))
    ax1.yaxis.set_ticks(np.arange(min_y, max_y, tick_dist))
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)

    #test set
    ax2.scatter(y_test, y_pred_test, alpha=.1) 
    ax2.set_aspect('equal')
    ax2.set_xlim([min_y, max_y])
    ax2.set_ylim([min_y, max_y])
    ax2.plot([0, 1], [0, 1], transform=ax2.transAxes)
    ax2.xaxis.set_ticks(np.arange(min_y, max_y, tick_dist))
    ax2.yaxis.set_ticks(np.arange(min_y, max_y, tick_dist))

    ax2.xaxis.set_tick_params(labelbottom=False)
    ax2.yaxis.set_tick_params(labelleft=False)

    if test_err is not None:
        ax2.title.set_text('Test ' + err_type + ' = ' + str(round(test_err,2)))
    else:
        ax2.title.set_text('Test Data')

    # Fig2. Line of best fit 
    fig2 = None
    if x_train.shape[1]==1:
        #train data
        fig2, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
        if logTransformY:
            fig2.suptitle('Line of Best Fit - ' + predictor + ' vs. log(sale_price)')
        else:
            fig2.suptitle('Line of Best Fit - ' + predictor + ' vs. Sale Price')
        ax1.scatter(x_train,y_train,alpha=.1) 
        ax1.plot(x_train,y_pred_train,color='k') 
        ax1.set_ylim([min_y, np.max(all_y)])

        ax1.set_xlabel(predictor)
        if logTransformY:
            ax1.set_ylabel('log(sale_price)')
        else:
            ax1.set_ylabel('Sale Price')
        if train_err is not None:
            ax1.title.set_text('Train ' + err_type + ' = ' + str(round(train_err,2)))
        else:
            ax1.title.set_text('Train Data')
        #test data
        ax2.scatter(x_test,y_test,alpha=.1) 
        ax2.plot(x_test,y_pred_test,color='k') 
        if test_err is not None:
            ax2.title.set_text('Test ' + err_type + ' = ' + str(round(test_err,2)))
        else:
            ax2.title.set_text('Test Data')
        ax2.set_ylim([min_y, np.max(all_y)])

    return (fig1, fig2)
    
# PCA
def my_PCA(X: pd.DataFrame, variance_thresh: float) -> pd.DataFrame:
    raise NotImplementedError    
    
    
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, Tuple

from sklearn.datasets import make_classification


def plot_salesprice(df: pd.DataFrame, ylog: bool = False) -> None:
    # set font size w/ params context manager
    font = {'font.size': 14}
    with mpl.rc_context(font):
        # define values
        SalePrices = df.SalePrice.tolist()
        mean = sum(SalePrices)/len(SalePrices)
        median = sorted(SalePrices)[len(SalePrices)//2]

        # define colors
        my_blue = [30/255, 136/255, 229/255]
        my_yellow = [255/255, 193/255, 7/255]
        my_magenta = [216/255, 27/255, 96/255]

        # plot
        try:
            df[~df['top_10']].SalePrice.hist(bins=20, color=my_blue, label='other prices', figsize=(8, 6))
            df[df['top_10']].SalePrice.hist(bins=20, color='brown', label='Top 10 % of prices', figsize=(8, 6))
        except (KeyError, ValueError):
            df.SalePrice.hist(bins=20, color=my_blue, label='prices', figsize=(8, 6))

        if ylog:
            plt.yscale('log')
            plt.ylim(1, plt.ylim()[1])
        plt.title('SalePrice Distribution: Count vs. SalePrice')
        plt.ylabel('Count, n')
        plt.xlabel("Sales Price, USD")
        plt.xticks(rotation=45, ha='right')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(lambda x, y: "{:,}".format(int(x)))
        plt.axvline(mean, label='mean: {:,.0f}'.format(mean), color=my_magenta)
        plt.axvline(median, label='median: {:,.0f}'.format(median), color=my_yellow)
        plt.legend()
        plt.show()
        

def split_df(df: pd.DataFrame,
             split_col: str,
             bottom_split: Optional[int] = None,
             top_split: Optional[int] = None) -> pd.DataFrame:
    """
    df: df to split (buy rows)
    split_col: column name with values to split on
    bottom_split: 2 digit int - percent from bottom for one split
    top_split: 2 digit int - percent from top for the other split
    returns: original df with top and bottom split bool columns added.
    """
    if bottom_split and top_split:
        if bottom_split + top_split > 100:
            raise ValueError('bottom and top splits must not sum to more than 100')

    # top zsplit
    if top_split:
        top_split_col_name = f'top_{top_split}'
        df_top_split = df[split_col].sort_values(ascending=False)[:int(len(df[split_col])/100*top_split)]

        # create new BOOLEAN column that is True when prices equal to or above top 30 value else False
        df[top_split_col_name] = df[split_col] >= df_top_split.tolist()[-1]

        if not bottom_split:
            df_bottom_split = df[~df[top_split_col_name]]

    # bottom split
    if bottom_split:
        bot_split_col_name = f'bot_{bottom_split}'
        df_bottom_split = df[split_col].sort_values(ascending=True)[:int(len(df[split_col])/100*bottom_split)]

        # create new BOOLEAN column that is True when prices equal to or below bottom 30 value else False
        df[bot_split_col_name] = df[split_col] <= df_bottom_split.tolist()[-1]

        if not top_split:
            df_top_split = df[~df[bot_split_col_name]]

    # sanity warning
    if bottom_split and top_split:
        in_both_df = df[df[top_split_col_name] & df[bot_split_col_name]]
        in_both_df_len = len(in_both_df)
        if in_both_df_len > 0:
            logging.warning(f'{in_both_df_len} rows are in both splits:\n{in_both_df}')

    return df


def plot_corr_matrix_allVars(df: pd.DataFrame) -> pd.DataFrame:
    cols_in_corr_order=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'LotArea', 'BsmtFullBath', 'BsmtUnfSF', 'BedroomAbvGr', 'ScreenPorch', 'PoolArea', 'MoSold', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass', 'EnclosedPorch', 'KitchenAbvGr']
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    corr_mat = (
        df.drop(['top_10', 'hue'], axis=1)
        [cols_in_corr_order]
        .corr()
    )
    upper_triangle_mask = np.triu(corr_mat)
    sns.heatmap(corr_mat, cmap='magma', ax=ax, mask=upper_triangle_mask)
    plt.title('correlation heatmap')
    plt.show()
    return corr_mat

def plot_corr_matrix(corr_matrix: pd.DataFrame) -> plt.figure:
    # Create a heatmap with variable labels
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, cbar=True, linewidths=0.5)

    # Set plot labels
    plt.title("Correlation Matrix")
    plt.xlabel("Predictor Variables")
    plt.ylabel("Predictor Variables")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Show the plot
    plt.tight_layout()
    return fig


def plot_regression_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols_in_corr_order=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'BsmtQual_Ex', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'KitchenQual_Ex', 'Foundation_PConc', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'ExterQual_Gd', 'ExterQual_Ex', 'BsmtFinType1_GLQ', 'HeatingQC_Ex', 'GarageFinish_Fin', 'Neighborhood_NridgHt', 'BsmtFinSF1', 'SaleType_New', 'SaleCondition_Partial', 'LotFrontage', 'FireplaceQu_Gd', 'GarageType_Attchd', 'MasVnrType_Stone', 'Neighborhood_NoRidge', 'WoodDeckSF', 'KitchenQual_Gd', '2ndFlrSF', 'OpenPorchSF', 'BsmtExposure_Gd', 'Exterior2nd_VinylSd', 'Exterior1st_VinylSd', 'HalfBath', 'GarageCond_TA', 'LotArea', 'FireplaceQu_Ex', 'CentralAir_Y', 'GarageQual_TA', 'MSZoning_RL', 'HouseStyle_2Story', 'Electrical_SBrkr', 'RoofStyle_Hip', 'GarageType_BuiltIn', 'BsmtQual_Gd', 'PavedDrive_Y', 'BsmtFullBath', 'LotShape_IR1', 'Neighborhood_StoneBr', 'BsmtUnfSF', 'MasVnrType_BrkFace', 'GarageFinish_RFn', 'RoofMatl_WdShngl', 'BedroomAbvGr', 'FireplaceQu_TA', 'PoolQC_Ex', 'LotConfig_CulDSac', 'Neighborhood_Somerst', 'BldgType_1Fam', 'BsmtExposure_Av', 'Exterior1st_CemntBd', 'Exterior2nd_CmentBd', 'Neighborhood_Timber', 'LotShape_IR2', 'LandContour_HLS', 'BsmtFinType2_Unf', 'Functional_Typ', 'Condition1_Norm', 'ScreenPorch', 'ExterCond_TA', 'BsmtCond_TA', 'Heating_GasA', 'PoolArea', 'MSZoning_FV', 'BsmtCond_Gd', 'Exterior2nd_ImStucc', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Veenker', 'Neighborhood_ClearCr', 'Condition1_PosN', 'Condition2_PosN', 'Condition2_PosA', 'MoSold', 'LandContour_Low', 'Exterior2nd_Other', 'RoofMatl_WdShake', '3SsnPorch', 'BsmtExposure_Mn', 'GarageQual_Gd', 'LandSlope_Mod', 'Condition1_PosA', 'BsmtFinType2_ALQ', 'SaleType_Con', 'Street_Pave', 'LotShape_IR3', 'HouseStyle_2.5Fin', 'Exterior1st_Stone', 'Neighborhood_Gilbert', 'GarageQual_Ex', 'Exterior1st_BrkFace', 'Condition2_Norm', 'LandSlope_Sev', 'Exterior1st_ImStucc', 'Exterior2nd_BrkFace', 'Neighborhood_NWAmes', 'Condition1_RRNn', 'MiscFeature_TenC', 'RoofStyle_Shed', 'RoofMatl_Membran', 'SaleType_CWD', 'Neighborhood_Blmngtn', 'LotConfig_FR3', 'RoofStyle_Flat', 'PoolQC_Fa', 'Neighborhood_SawyerW', 'SaleType_ConLI', 'Utilities_AllPub', 'PoolQC_Gd', 'ExterCond_Ex', 'Condition1_RRAn', 'RoofMatl_Tar&Grv', 'Condition1_RRNe', 'LotConfig_Corner', 'BldgType_TwnhsE', 'Condition2_RRAe', 'Foundation_Wood', 'BsmtFinType2_GLQ', 'RoofMatl_Metal', 'RoofStyle_Mansard', 'GarageCond_Gd', 'MiscFeature_Gar2', 'Fence_GdPrv', 'LotConfig_FR2', 'RoofMatl_ClyTile', 'BsmtFinSF2', 'Foundation_Stone', 'Utilities_NoSeWa', 'RoofMatl_Roll', 'Condition2_RRAn', 'SaleCondition_Alloca', 'Functional_Mod', 'Exterior2nd_Stone', 'BsmtHalfBath', 'Functional_Sev', 'Exterior1st_Plywood', 'Heating_GasW', 'Neighborhood_Blueste', 'MiscVal', 'GarageType_2Types', 'Exterior2nd_AsphShn', 'Exterior2nd_CBlock', 'Exterior1st_CBlock', 'LowQualFinSF', 'Heating_OthW', 'HouseStyle_2.5Unf', 'FireplaceQu_Fa', 'GarageCond_Ex', 'Exterior1st_AsphShn', 'SaleType_ConLw', 'LandContour_Lvl', 'Alley_Pave', 'YrSold', 'GarageType_Basment', 'Exterior1st_Stucco', 'HeatingQC_Po', 'Functional_Maj1', 'ExterCond_Po', 'Condition2_Artery', 'SaleType_Oth', 'RoofStyle_Gambrel', 'Heating_Floor', 'Electrical_Mix', 'BsmtFinType2_LwQ', 'Neighborhood_NPkVill', 'HouseStyle_SLvl', 'Condition2_RRNn', 'BsmtFinType2_Rec', 'Exterior2nd_Wd Shng', 'MiscFeature_Othr', 'Street_Grvl', 'SaleType_ConLD', 'Exterior2nd_Stucco', 'MasVnrType_BrkCmn', 'GarageQual_Po', 'SaleCondition_Family', 'Condition1_RRAe', 'Exterior2nd_Brk Cmn', 'Electrical_FuseP', 'Condition2_Feedr', 'ExterCond_Gd', 'SaleCondition_AdjLand', 'LandSlope_Gtl', 'Fence_MnWw', 'Exterior1st_BrkComm', 'Exterior1st_WdShing', 'Exterior2nd_Plywood', 'BsmtCond_Po', 'BsmtFinType2_BLQ', 'Neighborhood_Mitchel', 'Heating_Wall', 'HouseStyle_1Story', 'GarageCond_Po', 'Neighborhood_SWISU', 'Functional_Min1', 'MSZoning_RH', 'Exterior2nd_HdBoard', 'MiscFeature_Shed', 'Functional_Maj2', 'GarageType_CarPort', 'Functional_Min2', 'FireplaceQu_Po', 'OverallCond', 'LotConfig_Inside', 'SaleType_COD', 'BsmtFinType1_Unf', 'MSSubClass', 'BsmtFinType1_LwQ', 'HouseStyle_1.5Unf', 'PavedDrive_P', 'Heating_Grav', 'HouseStyle_SFoyer', 'Exterior1st_HdBoard', 'BldgType_2fmCon', 'BldgType_Twnhs', 'Exterior2nd_AsbShng', 'Fence_GdWo', 'LandContour_Bnk', 'Neighborhood_BrDale', 'BsmtFinType1_ALQ', 'RoofMatl_CompShg', 'Condition1_Artery', 'Exterior1st_AsbShng', 'MSZoning_C (all)', 'Neighborhood_MeadowV', 'BldgType_Duplex', 'ExterQual_Fa', 'Condition1_Feedr', 'SaleCondition_Abnorml', 'Foundation_Slab', 'Electrical_FuseF', 'Neighborhood_Sawyer', 'EnclosedPorch', 'BsmtQual_Fa', 'GarageCond_Fa', 'BsmtCond_Fa', 'BsmtFinType1_BLQ', 'GarageQual_Fa', 'HeatingQC_Fa', 'HeatingQC_Gd', 'BsmtFinType1_Rec', 'KitchenAbvGr', 'ExterCond_Fa', 'Alley_Grvl', 'Fence_MnPrv', 'Neighborhood_BrkSide', 'SaleCondition_Normal', 'KitchenQual_Fa', 'Exterior1st_Wd Sdng', 'Exterior2nd_Wd Sdng', 'Exterior2nd_MetalSd', 'HouseStyle_1.5Fin', 'Neighborhood_IDOTRR', 'Exterior1st_MetalSd', 'Neighborhood_Edwards', 'Neighborhood_NAmes', 'Neighborhood_OldTown', 'Electrical_FuseA', 'Foundation_BrkTil', 'PavedDrive_N', 'RoofStyle_Gable', 'SaleType_WD', 'CentralAir_N', 'BsmtExposure_No', 'LotShape_Reg', 'MSZoning_RM', 'HeatingQC_TA', 'Foundation_CBlock', 'GarageType_Detchd', 'MasVnrType_None', 'GarageFinish_Unf', 'BsmtQual_TA', 'KitchenQual_TA', 'ExterQual_TA']
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    corr_mat = (
        df
        .corr()
    )
    upper_triangle_mask = np.triu(corr_mat)
    sns.heatmap(corr_mat, cmap='magma', ax=ax, mask=upper_triangle_mask)
    plt.title('correlation heatmap')
    plt.show()
    return corr_mat


def plot_2d_pca(X_pca_df: pd.DataFrame, pc0_upper_limit: int = 0) -> None:
    X_pca_top_10 = X_pca_df[X_pca_df['category'] == 1]
    X_pca_not_10 = X_pca_df[X_pca_df['category'] == 0]

    # plot it
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    not10scatter = ax.scatter(X_pca_not_10[0], X_pca_not_10[1],
                              c='#1f77b4',
                              label='not top 10',
                              alpha=0.3)
    top10scatter = ax.scatter(X_pca_top_10[0], X_pca_top_10[1],
                              c='orange',
                              label='top 10',
                              alpha=0.3)
    ax.set_title('scatter plot of pc1 vs pc0 shaded by SalePrice category')
    ax.set_xlabel('pc0, arbitrary units')
    ax.set_ylabel('pc1, arbitrary units')
    ax.yaxis.set_major_formatter(lambda x, y: '{:,}'.format(int(x)))
    ax.xaxis.set_major_formatter(lambda x, y: '{:,}'.format(int(x)))
    if pc0_upper_limit:
        ax.set_xlim(ax.get_xlim()[0], pc0_upper_limit)
    ax.legend()
    plt.show()
    

def demo_standardization(column_name: str, df: pd.DataFrame) -> None:
    raw = df[column_name].tolist()
    mean = sum(raw)/len(raw)
    variances = [(x-mean)**2 for x in raw]
    variance = sum(variances)/len(variances)
    stdev = math.sqrt(variance)
    standardized = [(x-mean)/stdev for x in raw]

    standardized_mean = sum(standardized)/len(standardized)
    s_variances = [(x-standardized_mean)**2 for x in standardized]
    s_variance = sum(s_variances)/len(s_variances)
    standardized_stdev = math.sqrt(s_variance)

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
    (raw_hts, raw_edges, raw_plot) = ax.hist(
        raw, 
        label='raw', 
        alpha=0.6
    )
    x_min = min(raw_edges)
    x_max = max(raw_edges)
    y_min = 0
    y_max = max(raw_hts)
    x_range = x_max - x_min
    y_range = y_max
    plt.title('raw')
    ax.set_ylabel('number of observations')
    ax.set_xlabel(column_name)
    plt.text(x_min + (x_range/10)*7.5, y_min + y_range/10*9, 'raw mean: {:.2f}'.format(mean))
    plt.text(x_min + (x_range/10)*7.5, y_min + y_range/10*8, 'raw stdev {:.2f}'.format(stdev))


    (std_hts, std_edges, std_plot) = ax.hist(
        standardized, 
        label='standardized', 
        alpha=0.6
    )
    x_min = min(std_edges)
    x_max = max(std_edges)
    y_min = 0
    y_max = max(std_hts)
    x_range = x_max - x_min
    y_range = y_max
    plt.text(x_min + (x_range/10)*7.5, y_min + y_range/10*9, 'std mean {:.2f}'.format(standardized_mean))
    plt.text(x_min + (x_range/10)*7.5, y_min + y_range/10*8, 'std stdev {:.2f}'.format(standardized_stdev))
    ax.legend()

    plt.show()
    

def create_feature_data(random_state: int) -> np.ndarray:
    '''create x and y correlated variable'''
    X1, Y1 = make_classification(
        n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
        random_state=random_state

    )

    feature_1 = [x for x, y in zip(X1, Y1) if y == 0]
    return np.stack(feature_1)


def create_normalized_feature(arr: np.ndarray) -> np.ndarray:
    """center data at 0"""
    normalized_arr = np.zeros_like(arr)
    for dim in range(arr.shape[-1]):
        data = arr[:, dim]
        normalized_arr[:, dim] = (data-np.mean(data))/np.std(data)
    return normalized_arr


def create_feature_scatter_plot(random_state: int) -> Tuple[np.ndarray, Tuple[float, float]]:
    # feature_data = create_feature_data()
    feature_data = create_normalized_feature(
            create_feature_data(random_state=random_state)
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.scatter(feature_data[:,0], feature_data[:,1], label='original features')
    # ensure square plot
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    min_ax_val = min(xlims[0], ylims[0])
    max_ax_val = max(xlims[1], ylims[1])
    ax.set_xlim(min_ax_val, max_ax_val)
    ax.set_ylim(min_ax_val, max_ax_val)
    # label and plot
    ax.set_title('random feature data scatter plot')
    ax.set_xlabel('feature x value, arbitrary units')
    ax.set_ylabel('feature y value, arbitrary units')
    return ax, feature_data, (min_ax_val, max_ax_val)


def plot_pca_features(features_pca: np.ndarray) -> Tuple[plt.Figure, plt.Axes, Tuple[float, float]]:
    """
    plot PCA data centreed in a square plot
    """
    # get extents, set to extremes
    min_x = min(features_pca[:,0])
    max_x = max(features_pca[:,0])
    min_y = min(features_pca[:,1])
    max_y = max(features_pca[:,1])
    ax_max = max(max_x, max_y)
    ax_min = min(min_x, min_y)

    # plot data within extents above
    fig, pca_ax = plt.subplots(1,1,figsize=(8, 8))
    pca_im = pca_ax.scatter(features_pca[:,0], features_pca[:, 1], color='orange')
    pca_ax.set_xlim(ax_min, ax_max)
    pca_ax.set_ylim(ax_min, ax_max)
    pca_ax.set_xlabel('pca component 0')
    pca_ax.set_ylabel('pca component 1')
    pca_ax.set_title('2 component pca')
    return fig, pca_ax, (ax_min, ax_max)


def add_pc_plot(ax: plt.Axes, p: PCA, 
                color_pc0: str = 'xkcd:pink', color_pc1: str = 'xkcd:black', 
                add_labels: bool = False) -> plt.Axes:
        """plot primary components"""
        for i, ((x, y), color) in enumerate(zip(p.components_, [color_pc0, color_pc1])):
            ax.plot([0, x], [0, y], color=color)
            if add_labels:
                ax.text(x, y, s="PC{}: {:.2f}, {:.2f}".format(i, x, y))
        return ax


def plot_pca_feature_comparison(features: np.ndarray, features_pca: np.ndarray,
                                ax_max: float, ax_min: float, p: PCA,
                                point_to_highlight: int=10) -> Tuple[plt.Figure, np.ndarray]:
    """
    compare 2D features pre and post PCA
    features: numpy ndarray of original features (2D)
    features_pca: numpy ndarray of PCA'd original features (2D)
    ax_max: float max axes length
    ax_min: float min axes length
    p: PCA model fit on features
    point_to_highlight: int choose a point to highlight in red on plot
    """
    color1 = "xkcd:bright blue"
    color2 = "xkcd:light orange"
    color_pc0 = "xkcd:pink"
    color_pc1 = "xkcd:black"

    gs_kw = {
        "height_ratios": [10, 1, 1],
        "width_ratios": [1, 1],
        "hspace": 0.1,
    }

    fig, axs = plt.subplots(3, 2, figsize=(8, 5.75), 
                            sharex=True, 
                            constrained_layout=True,
                            gridspec_kw=gs_kw)


    # plot orig data + PCA data
    feature_axs = [axs[0][0], axs[0][1]]
    feature_data = [features, features_pca]
    feature_titles = ["original features: x, y", "PCA features: PC0, PC1"]
    var_colors = [color1] + [color2]
    for ax, data, title, color in zip(feature_axs, feature_data, feature_titles, var_colors):
        ax.scatter(data[:,0], data[:,1], label=title, color=color)
        ax.set_title(title)
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        if point_to_highlight:
            ax.scatter(data[point_to_highlight,0], data[point_to_highlight,1], color='red', zorder=10)
            ax.text(data[point_to_highlight,0], data[point_to_highlight,1], s="{:.2f}, {:.2f}".format(data[point_to_highlight,0], data[point_to_highlight,1]))
        if "original" in title:
            # plot feature space eigenvectors
            ax = add_pc_plot(ax, p)
        if "PCA" in title:
            # hide y axis on PCA plot
            ax.yaxis.set_visible(False)
            # manually plot pca-space eigenvectors
            ax.plot([0, 1], [0, 0], color=color_pc0)
            ax.plot([0, 0], [0, 1], color=color_pc1)

    # plot feature variabilities
    var_axes = [axs[1][0], axs[2][0], axs[1][1], axs[2][1]]
    var_data = [features[:,0], features[:,1], features_pca[:,0], features_pca[:,1]]
    var_names = ["feature x", "feature y", "PC0", "PC1"]
    var_colors = [color1] * 2 + [color2] * 2
    for ax, data, name, color in zip(var_axes, var_data, var_names, var_colors):
        ax.scatter(data,[0]*len(data), alpha=0.6, color=color)
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(0.25, -0.25)
        ax.set_title(name)
        for spine in ["right", "top", "left"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.set_visible(False)
    return fig, axs


def show_pcs_on_unit_axes(p: PCA, color_pc0: str = 'xkcd:pink', color_pc1: str = 'xkcd:black'):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax = add_pc_plot(ax, p, color_pc0, color_pc1, add_labels=True)

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # set axes ticks to -1, 0, 1
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-1, 0, 1]))
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([-1, 0, 1]))

    # set axes lengths
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    plt.show()


def plot_eigenvectors(p: PCA) -> Tuple[plt.Figure, plt.Axes]:
    print(f'{p.components_.shape=}')
    fig, ax = plt.subplots(1,1)
    im = ax.pcolormesh(abs(p.components_.T), norm=mpl.colors.LogNorm(vmin=1e-6))
    ax.set_title('heat map of absolute values of PCA eigenvectors')
    ax.set_ylabel('eigenvector value')
    ax.set_xlabel('eigenvector')
    plt.colorbar(im)
    return plt.Figure, plt.Axes
