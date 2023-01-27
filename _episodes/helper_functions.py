import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional


def plot_salesprice(df: pd.DataFrame) -> None:
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
        plt.title('SalePrice Distribution: Count vs. SalePrice')
        plt.ylabel('Count, n')
        plt.xlabel("Sales Price, USD")
        plt.xticks(rotation=45, ha='right')
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

