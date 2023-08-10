import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn import metrics


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
    
def measure_model_err(X_train, X_test,
                      y_train, y_pred_train, 
                      y_test, y_pred_test, 
                      reg, metric, log_scaled):
    
    # 2) reverse log transformation (exponential)
    if log_scaled:
        y_pred_train=np.exp(y_pred_train)
        y_pred_test=np.exp(y_pred_test)
        y_train=np.exp(y_train)
        y_test=np.exp(y_test)
    
    # 3) calculate RMSE for train and test sets
    if metric == 'RMSE':
        train_err = metrics.mean_squared_error(y_train, y_pred_train, squared=False) # squared=False to get RMSE instead of MSE
        test_err = metrics.mean_squared_error(y_test, y_pred_test, squared=False) 
    elif metric == 'R-squared':
        train_err = reg.score(X_train, y_train) # returns R^2 ("coef of determination")
        test_err = reg.score(X_test, y_test) # returns R^2 ("coef of determination")
    elif metric == 'MAPE':
        train_err = metrics.mean_absolute_percentage_error(y_train, y_pred_train) 
        test_err = metrics.mean_absolute_percentage_error(y_test, y_pred_test) 

    return train_err, test_err

def plot_predictions(ax, y, y_pred, log_transform_y, keep_tick_labels):
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

def plot_train_test_predictions(predictors,
                                X_train, X_test,
                                y_train, y_test,
                                y_pred_train, y_pred_test,log_scaled,
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


def fit_eval_model(X_train, y_train, 
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
    train_err, test_err = measure_model_err(X_train, X_test,
                                            y_train, y_pred_train, 
                                            y_test, y_pred_test, 
                                            reg, metric, log_scaled)

    # print results
    print('Train', metric, '=', train_err)
    print('Test', metric, '=', test_err)
    perc_diff = (test_err-train_err)/train_err
    perc_diff = "{:.0%}".format(perc_diff)
    print('(Test-Train)/Train:', perc_diff)
    

    if include_plots:
        (fig1, fig2) = plot_train_test_predictions(predictors=predictors,
                                                   X_train=X_train, X_test=X_test,
                                                   y_train=y_train, y_test=y_test,
                                                   y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                                                   log_scaled=log_scaled);
    
    print('')
    
    return train_err, test_err


def compare_models_plot(df_model_err, metric):

    # Let's take a closer look at the results by sorting the test error from best feature to worst. We'll then plot performance by feature for both train and test data.
    test_err = np.asarray(df_model_err['Test Error'])
    sort_inds=[i[0] for i in sorted(enumerate(test_err), key=lambda x:x[1])]
    sort_inds = np.array(sort_inds)

    # now that we have the sort indices based on test set performance, we'll sort the trainErr, testErr, and feature name vectors
    train_err = np.asarray(df_model_err['Train Error'])
    all_feats = df_model_err['Predictor Variable']
    train_err=train_err[sort_inds]
    test_err=test_err[sort_inds]
    labels=all_feats[sort_inds] 
    print(labels)
    print(len(labels))
    
    # plot out top 10 features based on error; try tight layout or set fig size 
    num_feats_plot=min(30,len(labels))
    fig, ax = plt.subplots()
    ax.plot(train_err[0:num_feats_plot], linestyle='--', marker='o', color='b')
    ax.plot(test_err[0:num_feats_plot], linestyle='--', marker='o', color='r')
    
    ax.set_xticks(list(range(0, num_feats_plot)))
    ax.set_xticklabels(labels[0:num_feats_plot], rotation=45, ha='right')
    
    ax.set_ylabel(metric)
    ax.legend(['train','test']);
    # increase fig size a bit
    fig = plt.gcf()
    fig.set_size_inches(14, 7) 
    # remind ourselves of train/test error for top-performing predictor variable
    print(train_err[0])
    print(test_err[0])
    
    return labels
    

def compare_models(X_train, y_train, 
                   X_test, y_test,
                   predictors_list, 
                   metric, log_scaled, 
                   model_type, include_plots):
    feat_index=0
    train_err_list=[None] * len(X_train.columns)
    test_err_list=[None] * len(X_train.columns)
    for predictors in predictors_list:  

        train_err, test_err = fit_eval_model(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test,
                                               predictors=predictors, 
                                               metric=metric, log_scaled=log_scaled, 
                                               model_type=model_type, include_plots=include_plots)

        # store model errors
        train_err_list[feat_index] = train_err
        test_err_list[feat_index] = test_err
        feat_index+=1

    # store errors in pandas dataframe for ease of access downstream
    df_model_err = pd.DataFrame()
    df_model_err['Predictor Variable'] = X_train.columns
    df_model_err['Train Error'] = train_err_list
    df_model_err['Test Error'] = test_err_list
    
    return df_model_err

    
