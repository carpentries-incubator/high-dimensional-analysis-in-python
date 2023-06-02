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


def plot_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
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
    

def create_feature_data() -> np.ndarray:
    '''create x and y correlated variable'''
    X1, Y1 = make_classification(
        n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1
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


def create_feature_scatter_plot() -> Tuple[np.ndarray, Tuple[float, float]]:
    # feature_data = create_feature_data()
    feature_data = create_normalized_feature(create_feature_data())
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


