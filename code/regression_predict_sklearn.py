import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from typing import Optional, Tuple, List, Union


def train_linear_model(X_train, y_train, model_type):
    if model_type == "unregularized":
        reg = LinearRegression().fit(X_train,y_train)
    else:
        raise ValueError('Unexpected model_type encountered; model_type = ' + model_type)
  
    return reg

def get_train_test_pred(X_train, X_test, reg):
    # 1) get model predicitons based on transformed (z-scored) predictor vars
    y_pred_train=reg.predict(X_train)
    y_pred_test=reg.predict(X_test)
    
    return y_pred_train, y_pred_test

# baseline_pred: Union[np.Series, pd.DataFrame], 
def measure_model_err(y: Union[np.ndarray, pd.Series], baseline_pred: Union[float, np.float64, np.float32, int, np.ndarray, pd.Series],
                      y_train: Union[np.ndarray, pd.Series],
                      y_pred_train: Union[np.ndarray, pd.Series],
                      y_test: Union[np.ndarray, pd.Series],
                      y_pred_test: Union[np.ndarray, pd.Series],
                      metric: str, log_scaled: bool) -> Tuple[float, float]:
    """
    Measures the error of a regression model's predictions on train and test sets.
    
    Args:
        y (Union[np.ndarray, pd.Series]): Actual target values for full dataset (not transformed)
        baseline_pred (Union[float, np.float64, np.float32, int, np.ndarray, pd.Series]): Single constant or array of predictions equal to the length of y. Baseline is also not transformed.
        y_train (Union[np.ndarray, pd.Series]): Actual target values for the training set.
        y_pred_train (Union[np.ndarray, pd.Series]): Predicted target values for the training set.
        y_test (Union[np.ndarray, pd.Series]): Actual target values for the test set.
        y_pred_test (Union[np.ndarray, pd.Series]): Predicted target values for the test set.
        metric (str): The error metric to calculate ('RMSE', 'R-squared', or 'MAPE').
        log_scaled (bool): Whether the target values are log-scaled or not.

    Returns:
        Tuple[float, float]: A tuple containing the error values for the training set and test set.
    """
    # Check if baseline is single constant - convert to list
    continuous_numeric_types = (float, np.float64, np.float32, int)
    if isinstance(baseline_pred, continuous_numeric_types):
        baseline_pred = pd.Series(baseline_pred) 
        baseline_pred = baseline_pred.repeat(len(y))
    
    # reverse log transformation (exponential)
    if log_scaled:
        y_pred_train = np.exp(y_pred_train)
        y_pred_test = np.exp(y_pred_test)
        y_train = np.exp(y_train)
        y_test = np.exp(y_test)
    
    # calculate chosen metric
    if metric == 'RMSE':
        baseline_err = metrics.mean_squared_error(y, baseline_pred, squared=False)
        train_err = metrics.mean_squared_error(y_train, y_pred_train, squared=False)
        test_err = metrics.mean_squared_error(y_test, y_pred_test, squared=False)
    elif metric == 'R-squared':
        baseline_err = metrics.r2_score(y, baseline_pred)
        train_err = metrics.r2_score(y_train, y_pred_train)
        test_err = metrics.r2_score(y_test, y_pred_test)
    elif metric == 'MAPE':
        baseline_err = metrics.mean_absolute_percentage_error(y, baseline_pred)
        train_err = metrics.mean_absolute_percentage_error(y_train, y_pred_train)
        test_err = metrics.mean_absolute_percentage_error(y_test, y_pred_test)
    else:
        raise ValueError("Invalid metric. Choose from 'RMSE', 'R-squared', or 'MAPE'.")

    return baseline_err, train_err, test_err


def plot_predictions(ax: plt.Axes, y: np.ndarray, y_pred: np.ndarray,
                     log_transform_y: bool, keep_tick_labels: bool) -> plt.Axes:
    """
    Plot true vs. predicted values.

    Args:
        ax (plt.Axes): Matplotlib axis for plotting.
        y (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        log_transform_y (bool): Whether the target values are log-transformed.
        keep_tick_labels (bool): Whether to keep tick labels.

    Returns:
        plt.Axes: Matplotlib axis with the plot.
    """
    
    if log_transform_y:
        min_y = 11#np.percentile(all_y, 1)
        max_y = 13.6#np.percentile(all_y, 100)
        tick_dist = .5
    else:
        min_y = 50000#np.percentile(all_y, 2)
        max_y = 350001#np.percentile(all_y, 95)
        tick_dist = 50000
        

    # ax.set_aspect('equal')
    ax.scatter(y, y_pred, alpha=.1) 
    ax.set_aspect('equal')
    ax.set_xlim([min_y, max_y])
    ax.set_ylim([min_y, max_y])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='blue', linestyle='dashed')
    sns.regplot(x=y, y=y_pred, lowess=True, ax=ax, line_kws={'color': 'red'})

    ax.xaxis.set_ticks(np.arange(min_y, max_y, tick_dist))
    ax.yaxis.set_ticks(np.arange(min_y, max_y, tick_dist))
    
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    if keep_tick_labels:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    else:
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        
    return ax

def plot_train_test_predictions(predictors: List[str],
                                X_train: Union[np.ndarray, pd.Series, pd.DataFrame], X_test: Union[np.ndarray, pd.Series, pd.DataFrame],
                                y_train: Union[np.ndarray, pd.Series], y_test: Union[np.ndarray, pd.Series],
                                y_pred_train: Union[np.ndarray, pd.Series], y_pred_test: Union[np.ndarray, pd.Series],
                                log_scaled: bool,
                                err_type: Optional[str] = None,
                                train_err: Optional[float] = None,
                                test_err: Optional[float] = None) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    """
    Plot true vs. predicted values for train and test sets and line of best fit.

    Args:
        predictors (List[str]): List of predictor names.
        X_train (np.ndarray): Training feature data.
        X_test (np.ndarray): Test feature data.
        y_train (np.ndarray): Actual target values for the training set.
        y_test (np.ndarray): Actual target values for the test set.
        y_pred_train (np.ndarray): Predicted target values for the training set.
        y_pred_test (np.ndarray): Predicted target values for the test set.
        log_scaled (bool): Whether the target values are log-scaled or not.
        err_type (Optional[str]): Type of error metric.
        train_err (Optional[float]): Training set error value.
        test_err (Optional[float]): Test set error value.

    Returns:
        Tuple[Optional[plt.Figure], Optional[plt.Figure]]: Figures for true vs. predicted values and line of best fit.
    """
    
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
    if log_scaled:
        min_y = 10#np.percentile(all_y, 1)
        max_y = 14#np.percentile(all_y, 100)
        tick_dist = .5
    else:
        min_y = 50000#np.percentile(all_y, 2)
        max_y = 350001#np.percentile(all_y, 95)
        tick_dist = 50000
    
    # Fig1. True vs predicted sale price
    fig1, (ax1, ax2) = plt.subplots(1,2)#, sharex=True, sharey=True)
    if log_scaled:
        fig1.suptitle('True vs. Predicted log(Sale_Price)')
    else:
        fig1.suptitle('True vs. Predicted Sale Price')

    # train set
    if train_err is not None:
        ax1.title.set_text('Train ' + err_type + ' = ' + str(round(train_err,2)))
    else:
        ax1.title.set_text('Train Data')
    
    ax1 = plot_predictions(ax1, y_train, y_pred_train, log_scaled, keep_tick_labels=True)

    #test set
    if test_err is not None:
        ax2.title.set_text('Test ' + err_type + ' = ' + str(round(test_err,2)))
    else:
        ax2.title.set_text('Test Data')
    ax2 = plot_predictions(ax2, y_test, y_pred_test, log_scaled, keep_tick_labels=False)
    plt.show()
    # Fig2. Line of best fit 
    fig2 = None
    if X_train.shape[1]==1:
        predictor = predictors[0]
        #train data
        fig2, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
        if log_scaled:
            fig2.suptitle('Line of Best Fit - ' + predictor + ' vs. log(sale_price)')
        else:
            fig2.suptitle('Line of Best Fit - ' + predictor + ' vs. Sale Price')
        ax1.scatter(X_train,y_train,alpha=.1) 
        ax1.plot(X_train,y_pred_train,color='k') 
        ax1.set_ylim([min_y, np.max(all_y)])

        ax1.set_xlabel(predictor)
        if log_scaled:
            ax1.set_ylabel('log(sale_price)')
        else:
            ax1.set_ylabel('Sale Price')
        if train_err is not None:
            ax1.title.set_text('Train ' + err_type + ' = ' + str(round(train_err,2)))
        else:
            ax1.title.set_text('Train Data')
        #test data
        ax2.scatter(X_test,y_test,alpha=.1) 
        ax2.plot(X_test,y_pred_test,color='k') 
        if test_err is not None:
            ax2.title.set_text('Test ' + err_type + ' = ' + str(round(test_err,2)))
        else:
            ax2.title.set_text('Test Data')
        ax2.set_ylim([min_y, np.max(all_y)])
        plt.show()
        
    return (fig1, fig2)


def fit_eval_model(y, baseline_pred,
                   X_train, y_train, 
                   X_test, y_test, 
                   predictors, 
                   metric, log_scaled, 
                   model_type, include_plots):
    '''This function uses the predictor vars specified by predictor_vars to predict housing price. Function returns error metric for both train and test data'''
    
    # Convert response vectors from pandas series to numpy arrays. 
    # This is necessary for downstream analyses (required format for linear regression fucntion we'll use).
    y_train=np.array(y_train) 
    y_test=np.array(y_test) 

    if isinstance(predictors, str):
        # if working with a single predictor, convert to list (fit_eval_model expects a list of predictors, not a char)
        predictors = [predictors]
        
    # Index specific predictor vars. Use reshape to handle case of just one predictor var (convert to shape=[numRows,numvars] rather than shape=[numRows,] )

    X_train=np.array(X_train[predictors]).reshape(-1, len(predictors)) # index subset of predictor vars
    X_test=np.array(X_test[predictors]).reshape(-1, len(predictors)) # do the same for test set

    # report predictor var if there's only one
    if len(predictors)==1:
        preview_predict_var = ' (' + predictors[0] + ')'
    else:
        preview_predict_var = ''

    # print number of observations in train/test sets as well as number of features used to predict housing price
    print('# of predictor vars = ' + str(len(predictors)) + preview_predict_var)
    print('# of train observations = ' + str(X_train.shape[0]))
    print('# of test observations = ' + str(X_test.shape[0]))
    
    # fit model to training data
    reg = train_linear_model(X_train, y_train, model_type)

    # get predictions for train/test sets
    y_pred_train, y_pred_test = get_train_test_pred(X_train, X_test, reg)
    
    # get train and test set error
    baseline_err, train_err, test_err = measure_model_err(y, baseline_pred,
                                                          y_train, y_pred_train, 
                                                          y_test, y_pred_test, 
                                                          metric, log_scaled)

    # print results
    print('Baseline', metric, '=', baseline_err)
    print('Train', metric, '=', train_err)
    print('Holdout', metric, '=', test_err)
    perc_diff = (test_err-train_err)/train_err
    perc_diff = "{:.0%}".format(perc_diff)
    print('(Holdout-Train)/Train:', perc_diff)
    
    if include_plots:
        (fig1, fig2) = plot_train_test_predictions(predictors=predictors,
                                                   X_train=X_train, X_test=X_test,
                                                   y_train=y_train, y_test=y_test,
                                                   y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                                                   log_scaled=log_scaled);
    
    print('')
    
    return baseline_err, train_err, test_err


def compare_models_plot(df_model_err: pd.DataFrame, metric: str) -> List[str]:
    """
    Compare and plot model errors for different predictor variables.

    Args:
        df_model_err (pd.DataFrame): A DataFrame containing model errors for different predictor variables.
        metric (str): The error metric used for plotting.

    Returns:
        List[str]: A list of labels corresponding to the predictor variables sorted based on validation set performance (first index is best model).
    """
    
    # Let's take a closer look at the results by sorting the test error from best feature to worst. We'll then plot performance by feature for both train and test data.
    val_err = np.asarray(df_model_err['Validation Error'])
    sort_inds=[i[0] for i in sorted(enumerate(val_err), key=lambda x:x[1])]
    sort_inds = np.array(sort_inds)

    # now that we have the sort indices based on test set performance, we'll sort the trainErr, testErr, and feature name vectors
    train_err = np.asarray(df_model_err['Train Error'])
    all_feats = df_model_err['Predictor Variable']
    train_err = train_err[sort_inds]
    val_err = val_err[sort_inds]
    labels = all_feats[sort_inds] 
    
    # plot out top 10 features based on error; try tight layout or set fig size 
    num_feats_plot=min(30,len(labels))
    fig, ax = plt.subplots()
    ax.plot(train_err[0:num_feats_plot], linestyle='--', marker='o', color='b')
    ax.plot(val_err[0:num_feats_plot], linestyle='--', marker='o', color='r')
    
    ax.set_xticks(list(range(0, num_feats_plot)))
    ax.set_xticklabels(labels[0:num_feats_plot], rotation=45, ha='right')
    
    ax.set_ylabel(metric)
    ax.legend(['train','validation']);
    # increase fig size a bit
    fig = plt.gcf()
    fig.set_size_inches(14, 7) 
    # remind ourselves of train/test error for top-performing predictor variable
    print('Best model train error =', train_err[0])
    print('Best model validation error =',val_err[0])
    print('Worst model train error =', train_err[-1])
    print('Worst model validation error =',val_err[-1])
    
    return labels
    

def compare_models(y: Union[np.ndarray, pd.Series],
                   baseline_pred: Union[np.ndarray, pd.Series],
                   X_train: pd.DataFrame, y_train: Union[np.ndarray, pd.Series],
                   X_val: pd.DataFrame, y_val: Union[np.ndarray, pd.Series],
                   predictors_list: List[List[str]],
                   metric: str, log_scaled: bool,
                   model_type: str, include_plots: bool) -> pd.DataFrame:
    """
    Compare different models based on predictor variables and evaluate their errors.

    Args:
        y (Union[np.ndarray, pd.Series]): Target variable in its original scale (raw/untransformed).
        baseline_pred (Union[np.ndarray, pd.Series]): Baseline predictions (in same scale as original target, y).
        X_train (pd.DataFrame): Training feature data.
        y_train (Union[np.ndarray, pd.Series]): Actual target values for the training set.
        X_val (pd.DataFrame): Validation feature data.
        y_val (Union[np.ndarray, pd.Series]): Actual target values for the validation set.
        predictors_list (List[List[str]]): List of predictor variables for different models.
        metric (str): The error metric to calculate.
        log_scaled (bool): Whether the model was trained on log-scaled target values or not.
        model_type (str): Type of the model being used.
        include_plots (bool): Whether to include plots.

    Returns:
        pd.DataFrame: A DataFrame containing model errors for different predictor variables.
    """
    feat_index=0
    baseline_err_list=[None] * len(X_train.columns)
    train_err_list=[None] * len(X_train.columns)
    val_err_err_list=[None] * len(X_train.columns)
    for predictors in predictors_list:  
        baseline_err, train_err, val_err = fit_eval_model(y=y, baseline_pred=baseline_pred,
                                             X_train=X_train, y_train=y_train, 
                                             X_test=X_val, y_test=y_val,
                                             predictors=predictors, 
                                             metric=metric, log_scaled=log_scaled, 
                                             model_type=model_type, include_plots=include_plots)

        # store model errors
        baseline_err_list[feat_index] = baseline_err
        train_err_list[feat_index] = train_err
        val_err_err_list[feat_index] = val_err
        feat_index+=1

    # store errors in pandas dataframe for ease of access downstream
    df_model_err = pd.DataFrame()
    df_model_err['Predictor Variable'] = X_train.columns
    df_model_err['Baseline Error'] = baseline_err
    df_model_err['Train Error'] = train_err_list
    df_model_err['Validation Error'] = val_err_err_list
    
    return df_model_err

    