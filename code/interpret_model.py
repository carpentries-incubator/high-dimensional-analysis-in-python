import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
import numpy as np

def coef_plot(coefs: pd.Series, plot_const: bool) -> plt.Figure:
    """
    Plot coefficient values and feature importance based on sorted feature importance.

    Args:
        coeffs (pd.Series): Coefficient values.
        plot_const (bool): Whether or not to plot the y-intercept coef value

    Returns:
        plt.Figure: The figure containing the coefficient plots.
    """
    if not plot_const:
        coefs = coefs.drop('const')
        
    # Get feature importance based on coefficient magnitudes
    feature_importance = np.abs(coefs)
    sorted_indices = feature_importance.sort_values(ascending=False).index

    # Plot coefficient values based on feature importance order
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    coefs.loc[sorted_indices].plot(kind='bar', ax=axes[0])
    axes[0].set_title("Coefficient Values (Sorted Feature Importance)")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Coefficient Value")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot feature importance based on coefficient magnitudes
    feature_importance.loc[sorted_indices].plot(kind='bar', ax=axes[1])
    axes[1].set_title("Feature Importance based on Coefficient Magnitudes")
    axes[1].set_xlabel("Features")
    axes[1].set_ylabel("Magnitude")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return fig
