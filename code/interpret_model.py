import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def coef_plot(coefs: pd.Series, plot_const: bool = False, index: bool = None) -> plt.Figure:
    """
    Plot coefficient values and feature importance based on sorted feature importance.

    Args:
        coefs (pd.Series or np.ndarray): Coefficient values.
        plot_const (bool, optional): Whether or not to plot the y-intercept coef value. Default is False.
        index (list or pd.Index, optional): Index labels for the coefficients. Default is None.

    Returns:
        plt.Figure: The figure containing the coefficient plots.
    """
    if index is not None:
        coefs = pd.Series(coefs, index=index)
    
    if not isinstance(coefs, (pd.Series, np.ndarray)):
        raise ValueError("The 'coefs' argument must be a pandas Series or numpy array.")
    
    if not plot_const:
        coefs = coefs.drop('const', errors='ignore')
    
    # Check for perfect zeros and calculate sparsity
    num_zeros = np.sum(coefs == 0)
    sparsity = num_zeros / len(coefs) * 100    
    # Get feature importance based on coefficient magnitudes
    feature_importance = np.abs(coefs)
    # Exclude zeros from plot
    non_zero_indices = feature_importance[feature_importance > 0].index
    feature_importance = feature_importance.loc[non_zero_indices]
    coefs = coefs.loc[non_zero_indices]
    # Get sorted indices (non-zeros only)
    sorted_indices = feature_importance.sort_values(ascending=True).index
    
    # Plot coefficient values based on feature importance order
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot coefficient values subplot
    axes[0].barh(np.arange(len(sorted_indices)), coefs.loc[sorted_indices])
    axes[0].axvline(x=0, color='red', linestyle='--', label='Zero')
    axes[0].set_yticks(np.arange(len(sorted_indices)))
    axes[0].set_yticklabels(sorted_indices)
    axes[0].set_title("Coefficent Values")
    axes[0].set_xlabel("Coefficient Value")
    axes[0].set_ylabel("Features")
    
    # Plot feature importance subplot
    # feature_importance.loc[sorted_indices].plot(kind='bar', ax=axes[1])
    axes[1].barh(np.arange(len(sorted_indices)), feature_importance.loc[sorted_indices])

    axes[1].set_title("Feature Importance")
    axes[1].set_xlabel("Features")
    axes[1].set_ylabel("Magnitude")
    axes[1].tick_params(axis='x', rotation=45)
                      
    # Add suptitle if zeros are found
    if num_zeros > 0:
        suptitle = f"Number of Zeros: {num_zeros}, Sparsity: {sparsity:.2f}%"
        plt.suptitle(suptitle, fontsize=14)
    
    plt.tight_layout()
    
    return fig
