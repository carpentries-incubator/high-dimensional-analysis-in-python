# statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
# essentials
import numpy as np
import pandas as pd
from scipy import stats
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
# multicollinearity
from helper_functions import plot_corr_matrix 
# normal errors
import statsmodels.graphics.gofplots as smg # qqplot
# independent errors
from statsmodels.stats.stattools import durbin_watson


def normal_resid_test(resids) -> None:
    print('\n==========================')
    print('VERIFYING NORMAL ERRORS...')
    
    # Extract the residuals and calculate median — should lie close to 0 if it is a normal distribution
    print('Median of residuals:', np.median(resids))
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
    # fig.suptitle('Normal Residuals')

    # plot histogram
    ax1.hist(resids);
    # ax1.set_aspect('equal')
    ax1.set_title('Histogram of Residuals')

    # measure skew
    resids.skew() 

    # Plot the QQ-plot of residuals
    smg.qqplot(resids, line='s', ax=ax2)
    # ax2.set_aspect('equal')
    
    # Add labels and title
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.set_title('QQ-Plot of Residuals')

    # Hypothesis tests
    shapiro_stat, shapiro_p = stats.shapiro(resids)
    print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.10f}")

    # Perform the Kolmogorov-Smirnov test on the test_residuals
    ks_stat, ks_p = stats.kstest(resids, 'norm')
    print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_p:.10f}")
    
    plt.show()
    
    return fig

    
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_print_VIF(X):
    # Calculate VIF for each predictor in X
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Display the VIF values
    print(vif)
    
def multicollinearity_test(X: pd.DataFrame):
    print('\n==========================')
    print('VERIFYING MULTICOLLINEARITY...')
    
    # remove y-intercept column, if present
    X = X[X.columns.difference(['const'])]
    
    # caculate VIF
    calc_print_VIF(X)
    corr_matrix = X.corr()
    
    # plot results
    fig = plot_corr_matrix(corr_matrix)
    # fig.suptitle('Multicollinearity', y=1.05)

    # plt.savefig('..//fig//regression//assumptions//corrMat_multicollinearity2.png', bbox_inches='tight', dpi=300, facecolor='white');
    plt.show()
    return fig
    
    
def plot_pred_v_resid(y_pred, resids, ax):
    
    sns.regplot(x=y_pred, y=resids, lowess=True, ax=ax, line_kws={'color': 'red'})
    ax.axhline(0, color='blue', linestyle='dashed')  # Add a horizontal line at y=0

    # ax2.set_title('Residuals vs Fitted', fontsize=16)
    ax.set(xlabel='Predicted', ylabel='Residuals')
    ax.set_aspect('equal')
    return ax


def homoscedasticity_linearity_test(trained_model: sm.regression.linear_model.RegressionResultsWrapper, y: pd.Series, y_pred: pd.Series):
    '''
    Function for testing the homoscedasticity of residuals in a linear regression model.
    It plots residuals and standardized residuals vs. fitted values and runs Breusch-Pagan and Goldfeld-Quandt tests.
    
    Args:
    * model - fitted OLS model from statsmodels
    '''
    print('\n=======================================')
    print('VERIFYING LINEARITY & HOMOSCEDASTICITY...')

    sns.set_style('darkgrid')
    sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))#, sharex=True, sharey=True)
    # fig.suptitle('Linearity & Homoscedasticiy')

    # Predictions vs actual
    from helper_functions import plot_predictions 
    ax1 = plot_predictions(ax1, y, y_pred, log_transform_y=True, keep_tick_labels=True)
    resids = y_pred - y
    
    # Predictions vs residuals
    ax2 = plot_pred_v_resid(y_pred, resids, ax2)
    
    # GQ test
    gq_test = pd.DataFrame(sms.het_goldfeldquandt(resids, trained_model.model.exog)[:-1],
                           columns=['value'],
                           index=['F statistic', 'p-value'])

    print('\n Goldfeld-Quandt test (homoscedasticity) ----')
    print(gq_test)
    print('\n Residuals plots ----')
    plt.show()
    return fig


def independent_resid_test(y_pred, resids, include_plot=True) -> None:
    print('\n==========================')
    print('VERIFYING INDEPENDENT ERRORS...')

    durbin_watson_statistic = durbin_watson(resids)
    print(f"Durbin-Watson test statistic: {durbin_watson_statistic}")

    if include_plot:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        # fig.suptitle('Independent Errors')
        ax = plot_pred_v_resid(y_pred, resids, ax)
    else:
        fig = None
        
    plt.show()
    return fig


def eval_regression_assumptions(trained_model, X, y, y_pred):
    resids = y - y_pred
    multicollinearity_test(X)
    homoscedasticity_linearity_test(trained_model, y, y_pred)
    normal_resid_test(resids) 
    independent_resid_test(y_pred, resids, include_plot=False)
    
    return None

