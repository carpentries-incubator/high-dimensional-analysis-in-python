# statsmodels
import statsmodels.graphics.gofplots as smg
import statsmodels.api as sm
import statsmodels.stats.api as sms
# essentials
import numpy as np
import pandas as pd
from scipy import stats
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
from helper_functions import plot_corr_matrix 



def normal_resid_test(resids) -> None:
    
    # Extract the residuals and calculate median — should lie close to 0 if it is a normal distribution
    print('Median of residuals:', np.median(resids))
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))

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

    
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_print_VIF(X):
    # Calculate VIF for each predictor in X
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Display the VIF values
    print(vif)
    
def multicollinearity_test(X: pd.DataFrame):

    calc_print_VIF(X)
    corr_matrix = X.corr()
    fig = plot_corr_matrix(corr_matrix)
    # plt.savefig('..//fig//regression//assumptions//corrMat_multicollinearity2.png', bbox_inches='tight', dpi=300, facecolor='white');
    plt.show()
    

def homoscedasticity_linearity_test(trained_model: sm.regression.linear_model.RegressionResultsWrapper, y: pd.Series, y_pred: pd.Series):
    '''
    Function for testing the homoscedasticity of residuals in a linear regression model.
    It plots residuals and standardized residuals vs. fitted values and runs Breusch-Pagan and Goldfeld-Quandt tests.
    
    Args:
    * model - fitted OLS model from statsmodels
    '''
    sns.set_style('darkgrid')
    sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))#, sharex=True, sharey=True)

    from helper_functions import plot_predictions 
    ax1 = plot_predictions(ax1, y, y_pred, log_transform_y=True, keep_tick_labels=True)
    resids = y_pred - y

    sns.regplot(x=y_pred, y=resids, lowess=True, ax=ax2, line_kws={'color': 'red'})
    ax2.axhline(0, color='blue', linestyle='dashed')  # Add a horizontal line at y=0

    # ax2.set_title('Residuals vs Fitted', fontsize=16)
    ax2.set(xlabel='Predicted', ylabel='Residuals')
    ax2.set_aspect('equal')


    gq_test = pd.DataFrame(sms.het_goldfeldquandt(resids, trained_model.model.exog)[:-1],
                           columns=['value'],
                           index=['F statistic', 'p-value'])

    print('\n Goldfeld-Quandt test (homoscedasticity) ----')
    print(gq_test)
    print('\n Residuals plots ----')
    
    return fig