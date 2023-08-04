---
title: Regularization methods - lasso, ridge, and elastic net
teaching: 45
exercises: 2
keypoints:
- ""
objectives:
- ""
questions:
- ""
---

### Split data into train/test sets and zscore
We will now split our data into two separate groupings — one for fitting or training the model ("train set") and another for testing ("test set") the model's ability to generalize to data that was excluded during training. The amount of data you exclude for the test set should be large enough that the model can be vetted against a diverse range of samples. A common rule of thumb is to use 3/4 of the data for training, and 1/3 for testing.


```python
from sklearn.model_selection import train_test_split

# Perform train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.33, 
                                                    random_state=0)
print(X_train.shape)
print(X_test.shape)

print(type(y_train))
print(type(X_train))


```

    (978, 215)
    (482, 215)
    <class 'pandas.core.series.Series'>
    <class 'pandas.core.frame.DataFrame'>
    


```python
# sklearn version
from sklearn.linear_model import LinearRegression

# stats model version (for hypothesis testing)
from statsmodels.formula.api import ols

def train_linear_model(X_train, y_train, model_type):
    if model_type == "unregularized":
        reg = LinearRegression().fit(X_train,y_train)
#         reg = ols("dist ~ speed", data=cars).fit()
    else:
        raise ValueError('Unexpected model_type encountered; model_type = ' + model_type)
  
    # print number of estimated model coefficients. Need to add one to account for y-intercept (not included in reg.coef_ call)
    print('# model coefs = ' + str(len(reg.coef_)+1))

    return reg


```

Define a function `measure_model_err` to help us measure the model's performance (train/test RMSE)


```python
from sklearn import metrics

def measure_model_err(X_train, X_test, y_train, y_test, reg):
    
    # 1) get model predicitons based on transformed (z-scored) predictor vars
    y_pred_train=reg.predict(X_train)
    y_pred_test=reg.predict(X_test)
    
    # 2) reverse log transformation (exponential)
    y_pred_train=np.exp(y_pred_train)
    y_pred_test=np.exp(y_pred_test)
    y_train=np.exp(y_train)
    y_test=np.exp(y_test)
    
    # 3) calculate RMSE for train and test sets
    RMSE_train = metrics.mean_squared_error(y_train, y_pred_train,squared=False) # squared=False to get RMSE instead of MSE
    R2_train = reg.score(X_train, y_train) # returns R^2 ("coef of determination")
    RMSE_test = metrics.mean_squared_error(y_test, y_pred_test,squared=False) 
    R2_test = reg.score(X_test, y_test) # returns R^2 ("coef of determination")

    return RMSE_train, RMSE_test, R2_train, R2_test
```

Define a function `fit_eval_model` that will call both `train_linear_model` and `measure_model_err` and report back on model performance.


```python
def fit_eval_model(X_train, y_train, X_test, y_test, predictor_vars, model_type):
    '''This function uses the predictor vars specified by predictor_vars to predict housing price. Function returns RMSE for both train and test data'''
    # Convert response vectors from pandas series to numpy arrays. 
    # This is necessary for downstream analyses (required format for linear regression fucntion we'll use).
    y_train=np.array(y_train) 
    y_test=np.array(y_test) 

    # Index specific predictor vars. Use reshape to handle case of just one predictor var (convert to shape=[numRows,numvars] rather than shape=[numRows,] )
    X_train=np.array(X_train[predictor_vars]).reshape(-1, len(predictor_vars)) # index subset of predictor vars
    X_test=np.array(X_test[predictor_vars]).reshape(-1, len(predictor_vars)) # do the same for test set

    # report predictor var if there's only one
    if len(predictor_vars)==1:
        preview_predict_var = ' (' + predictor_vars[0] + ')'
    else:
        preview_predict_var = ''

    # print number of observations in train/test sets as well as number of features used to predict housing price
    print('# of predictor vars = ' + str(len(predictor_vars)) + preview_predict_var)
    print('# of train observations = ' + str(X_train.shape[0]))
    print('# of test observations = ' + str(X_test.shape[0]))
  
    # fit model to training data
    reg = train_linear_model(X_train, y_train, model_type)

    # get train and test set RMSE
    RMSE_train, RMSE_test = measure_model_err(X_train, X_test, y_train, y_test, reg)

    # print results
    print('Train RMSE:', RMSE_train)
    print('Test RMSE:', RMSE_test)
    perc_diff = (RMSE_test-RMSE_train)/RMSE_train
    perc_diff = "{:.0%}".format(perc_diff)
    print('(Test-Train)/Train:', perc_diff)
    return RMSE_train, RMSE_test

```


```python
import pandas as pd 

all_feats=X_train.columns
RMSE_train_list=[None] * len(all_feats)
RMSE_test_list=[None] * len(all_feats)

feat_index=0
for feat in all_feats:  
    # fit univariate model and return train/test RMSE
    RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, 
                                           X_test, y_test,
                                           [feat],'unregularized')
    print('')
    # store model errors
    RMSE_train_list[feat_index] = RMSE_train
    RMSE_test_list[feat_index] = RMSE_test#metrics.mean_squared_error(y_test, predicted_test,squared=False) # squared=False to get RMSE instead of MSE
    feat_index+=1
    
# store errors in pandas dataframe for ease of access downstream
df_model_err = pd.DataFrame()
df_model_err['Predictor Variable'] = all_feats
df_model_err['Train RMSE'] = RMSE_train_list
df_model_err['Test RMSE'] = RMSE_test_list

```

    # of predictor vars = 1 (MSSubClass_20.0)
    # of train observations = 978
    # of test observations = 482
    # model coefs = 2
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-63-e4a90b63b167> in <module>
          8 for feat in all_feats:
          9     # fit univariate model and return train/test RMSE
    ---> 10     RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, 
         11                                            X_test, y_test,
         12                                            [feat],'unregularized')
    

    <ipython-input-62-b8c93fc000b0> in fit_eval_model(X_train, y_train, X_test, y_test, predictor_vars, model_type)
         25 
         26     # get train and test set RMSE
    ---> 27     RMSE_train, RMSE_test = measure_model_err(X_train, X_test, y_train, y_test, reg)
         28 
         29     # print results
    

    ValueError: too many values to unpack (expected 2)


## Fit multivariate model using all predictor vars

#### Predictive Models VS Interpretable Models
* **Interpretable models**: Models trained with linear regression are the most interpretable kind of regression models available - meaning it’s easier to take action from the results of a linear regression model. However, if the assumptions are not satisfied, the interpretation of the results will not always be valid. This can be very dangerous depending on the application.

#### Assumptions of multivariate regression (for statistical/hypothesis testing)
1. Independence: All observations are independent
2. Linearity: The relationship between the dependent variable and the independent variables should be linear

    a. **Note**: In practice, linear models are often used to model nonlinear relationships due to complexity (number of model parameters/coefs that need to be estimated) of nonlinear models. When using a linear model to model nonlinear relationships, it usually best to use resulting model for predictive purposes only. 
3. Normality: For each value of the dependent variable, the distribution of the dependent variable must be normal.
4. Homoscedasticity: The residuals of a good model should be normally and randomly distributed i.e. the unknown does not depend on X ("homoscedasticity")


```python
print(len(labels)) 
```

    213
    


```python
help(fit_eval_model)
```

    Help on function fit_eval_model in module __main__:
    
    fit_eval_model(X_train, y_train, X_test, y_test, predictor_vars, model_type)
        This function uses the predictor vars specified by predictor_vars to predict housing price. Function returns RMSE for both train and test data
    
    


```python
# fit model using all features/predictors available
RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, X_test, y_test, labels, 'unregularized')
```

    # of predictor vars = 213
    # of train observations = 978
    # of test observations = 482
    # model coefs = 214
    Train RMSE: 21981.654614715466
    Test RMSE: 3562241001.482347
    (Test-Train)/Train: 16205418%
    

### Discuss
Is this a good model? Does this model encounter overfitting?

Flesh this out. How many features, how many observations, how many model coefs

## Regularized regression: ridge, lasso, elastic net


### Ridge and RidgeCV
- Show ridge optimization equation
- Default CV is Leave-One-Out. In this form of CV, all samples in the data except for one are used as the inital training set. The left out sample is used a validation set.
- One alpha value used for entire model; larger alphas give more weight to the penalty/regularization term of the loss function

Edit function below to use multiple regression techniques (add model_type input)






```python
# edit train_linear_model to train ridge models as well
def train_linear_model(X_train, y_train, model_type):
    if model_type == "unregularized":
        reg = LinearRegression().fit(X_train,y_train)
    elif model_type == 'ridge':
        reg = RidgeCV(alphas=[1e-3,1e-2,1e-1,1,10,100,1000], store_cv_values=True).fit(X_train,y_train)
        print(reg.cv_values_.shape) # num_datapoints x num_alphas
        print(np.mean(reg.cv_values_, axis=0))
        print(reg.alpha_)
    else:
        raise ValueError('Unexpected model_type encountered; model_type = ' + model_type)

    # print number of estimated model coefficients. Need to add one to account for y-intercept (not included in reg.coef_ call)
    print('# model coefs = ' + str(len(reg.coef_)+1))

    return reg


```


```python
# import sklearn's ridge model with built-in cross-validation
from sklearn.linear_model import RidgeCV 

# fit model using multivariate_model_feats and ridge regression
RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, X_test, y_test, labels, 'ridge')
```

    # of predictor vars = 213
    # of train observations = 978
    # of test observations = 482
    (978, 7)
    [1.01586692e+09 1.01401918e+09 9.99400573e+08 9.57029390e+08
     9.43452552e+08 1.02279420e+09 1.21826389e+09]
    10.0
    # model coefs = 214
    Train RMSE: 25463.82775189401
    Test RMSE: 39003.787373887266
    (Test-Train)/Train: 53%
    

- What is the model's train and test error? How does this compare to the unregularized model we fit using all predictor variables? How does this model compare to the best univariate model we fit?
  - The ridge model does much better (i.e., in terms of Test RMSE) than the unregularized model that uses all predictor vars.
  - Unregularized_all_predictors_testRMSE: 3562241001
  - Unregularized_best_univariate_testRMSE: 48243
  - Regularized_all_predictors_testRMSE: 39004

- What alpha value was selected using RidgeCV? Is it a lower or higher value? What does this value tell you about the model?
  - This model is highly regularized/penalized since it has a large alpha value



### LASSO
- explain why there's a random state param in LASSO but not ridge



```python
# edit train_linear_model to train ridge models as well
def train_linear_model(X_train, y_train, model_type):
    if model_type == "unregularized":
        reg = LinearRegression().fit(X_train,y_train)
    elif model_type == 'ridge':
        reg = RidgeCV(alphas=[1e-3,1e-2,1e-1,1,10,100,1000], store_cv_values=True).fit(X_train,y_train)
        print(reg.cv_values_.shape) # num_datapoints x num_alphas
        print(np.mean(reg.cv_values_, axis=0))
        print(reg.alpha_)
    elif model_type == 'lasso':
        reg = LassoCV(random_state=0, alphas=[1e-3,1e-2,1e-1,1,10,100,1000], max_iter=100000, tol=1e-3).fit(X_train,y_train)
        print(reg.alpha_)
        print(reg.alphas_)

    else:
        raise ValueError('Unexpected model_type encountered; model_type = ' + model_type)

    # print number of estimated model coefficients. Need to add one to account for y-intercept (not included in reg.coef_ call)
    print('# model coefs = ' + str(len(reg.coef_)+1))

    return reg


```


```python
# import sklearn's lasso model with built-in cross-validation
from sklearn.linear_model import LassoCV 

# fit model using multivariate_model_feats and ridge regression
RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, X_test, y_test, labels, 'lasso')
```

    # of predictor vars = 213
    # of train observations = 978
    # of test observations = 482
    

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 8284193614.777222, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 39790829366.316284, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 38718390339.6676, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 38647786272.54297, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 38613966518.50275, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 51724109695.34778, tolerance: 4985081801.251732
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 49532515142.88132, tolerance: 4985081801.251732
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 49422113573.12036, tolerance: 4985081801.251732
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 49474867747.30087, tolerance: 4985081801.251732
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 7255607704.328613, tolerance: 5231563918.29538
      positive,
    

    100.0
    [1.e+03 1.e+02 1.e+01 1.e+00 1.e-01 1.e-02 1.e-03]
    # model coefs = 214
    Train RMSE: 23844.411315377245
    Test RMSE: 41216.23320716389
    (Test-Train)/Train: 73%
    

Add elastic net option to function


```python
# edit train_linear_model to train ridge models as well
def train_linear_model(X_train, y_train, model_type):
    if model_type == "unregularized":
        reg = LinearRegression().fit(X_train,y_train)
    elif model_type == 'ridge':
        reg = RidgeCV(alphas=[1e-3,1e-2,1e-1,1,10,100,1000], store_cv_values=True).fit(X_train,y_train)
        print(reg.cv_values_.shape) # num_datapoints x num_alphas
        print(np.mean(reg.cv_values_, axis=0))
        print('alpha:', reg.alpha_)
    elif model_type == 'lasso':
        reg = LassoCV(random_state=0, alphas=[1e-3,1e-2,1e-1,1,10,100,1000], max_iter=100000, tol=1e-3).fit(X_train,y_train)
        print('alpha:', reg.alpha_)
        print('alphas:', reg.alphas_)
    elif model_type == 'elastic':
        reg = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],alphas=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10]).fit(X_train,y_train)
        print('alpha:', reg.alpha_)
        print('l1_ratio:', reg.l1_ratio_)
    else:
        raise ValueError('Unexpected model_type encountered; model_type = ' + model_type)

    # print number of estimated model coefficients. Need to add one to account for y-intercept (not included in reg.coef_ call)
    print('# model coefs = ' + str(len(reg.coef_)+1))

    return reg


```


```python
from sklearn.linear_model import ElasticNetCV

# fit model using multivariate_model_feats and ridge regression
RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, X_test, y_test, labels, 'elastic')
```

    # of predictor vars = 213
    # of train observations = 978
    # of test observations = 482
    

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 466501538000.58923, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 403270137725.9388, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 301577300752.58386, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 230081693672.962, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 191261132823.84335, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180015211573.03418, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178123858792.64304, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 517170940383.7549, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 455149703947.2829, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 356760757816.2706, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 271287951936.91687, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 208621382681.19177, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179934405122.62006, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174598046989.29358, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 497139127553.7737, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 439200852649.58246, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 339907195700.6849, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 258953594195.75394, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 200882179797.22876, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173374398849.39014, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168308116883.32767, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 531036794639.032, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 469827343258.0895, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 368933133080.9624, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 280605522374.60645, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 213947936192.50116, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 182087724825.03278, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176055602385.14227, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 572629384126.2748, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 501555480332.6684, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 391631272775.97107, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 298678305300.5673, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 227982526042.6555, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 195157942278.29935, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 189024550168.52182, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 444949892409.9793, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 365094575321.7053, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 271536336788.16092, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 211633099311.5671, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 186635909873.2761, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179121358447.05002, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178021230383.19012, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 493368405846.16003, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 414829399434.4205, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 322242581988.8907, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 246597333224.66333, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 196334081147.4532, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177416221956.2069, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174308462866.78204, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 480198849240.9201, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 405428439002.9113, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 309501576674.2682, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 236001205429.3958, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 190783090645.77213, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 170975409642.3658, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168035191309.2294, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 508714548362.01385, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 430015798460.9209, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 331451029685.684, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 253623789109.70438, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 201491454104.8407, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179241665013.0187, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175728390202.92365, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 547814708459.1266, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 460655259599.63354, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 357768526357.03766, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 272796444290.12274, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 214743064981.45016, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 192268977910.40497, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188690555192.06982, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 416677230542.26807, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 326470938261.0464, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 244818755605.2463, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 196565697156.34995, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 183909103175.64297, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178646713525.2646, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177969529190.60886, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 462432485621.19946, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 373224181082.7571, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 289057115458.06903, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 224721439217.2333, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 190549133808.97656, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176074535061.96317, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174162572900.09378, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 457774371126.0421, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 371054861296.9792, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 281551992503.0566, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 215505859575.48795, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 183583514120.1759, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 169702772421.62424, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167897783053.45456, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 479421550321.16235, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 388323215242.1754, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 296292818625.5564, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 228663127248.45554, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 194020395148.21582, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177724632997.83142, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175563562620.36926, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 514485326942.4092, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 421611389032.36316, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 326924241461.59247, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 250711795302.28946, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 207239240911.33615, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 190726062457.20798, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188522236475.86008, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 311797142313.5558, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 209862077232.94275, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179799225897.31848, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 162976250066.2721, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180226399106.9014, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178149146521.00806, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177917216985.7658, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 348771325443.47424, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 262676140169.38663, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 213606238041.7038, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 169325997730.12842, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180532958271.4085, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174670065202.66556, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174015846743.25925, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 369288213632.16943, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 286971076623.6909, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 215203533299.0186, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167078183835.83978, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173945243870.41464, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168375348268.6882, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167759303838.29395, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 369631488837.6472, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 274426855741.68402, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 212201514272.23502, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 154631300697.99414, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 182763227707.45428, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176136919103.90308, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175397761945.90024, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 399732561591.90906, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 325717700010.3299, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 256351885785.4635, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 201853296477.35535, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 195843376102.29, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 189107464528.57724, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188352732190.63144, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 220081626718.3236, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 155730175381.8516, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 154449120821.4552, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 181827917422.58365, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179122177438.0314, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178020909027.6598, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177904013510.02463, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 248813839151.22867, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 184846475897.15, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 156427193949.10315, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 163724362231.61304, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177417710095.71313, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174308423162.75894, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173978973823.79193, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 308635567974.98413, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 227821113321.46594, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 172002330293.1055, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188712555568.589, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 170975235250.40717, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168034122123.34964, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167724165476.4443, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 270167884662.31128, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 185075807794.65875, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 150140928105.46667, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 184244212674.26703, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179243266269.3932, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175728272974.75, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175356095927.54163, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 289957702789.10443, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 262688983586.6175, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 211292658771.88013, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 194232544058.9558, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 192270712977.04324, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188690421311.04477, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188310203297.35104, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 80443289126.99353, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 33940528950.43747, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 93696965405.23439, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180202740645.64984, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178146838446.83603, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177914128720.72086, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177890539075.50424, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 73876773879.9613, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 41155109697.85321, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 30215275736.665314, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180533254979.0658, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174670642077.96503, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174014803401.21664, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173948413577.51135, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168155841759.9858, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 124721798412.78943, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 96673988127.99841, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173936407575.40323, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168369125621.13373, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167752779273.1091, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167690449923.0291, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 88457878160.88037, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 44078712149.98291, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 43413779781.612, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 182757574338.25684, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176136877100.2423, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175396024287.3029, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175321047229.12817, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 194579646116.2381, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 154082096434.86902, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 124051165230.44101, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 195842065111.45575, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 189106603639.84344, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188350172728.8122, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188273563349.11105, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 69964040921.6865, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175528407923.5064, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176837080816.7248, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176867290496.7877, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177240763902.08405, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177003775798.41608, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176997329865.3684, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11322111222.890259, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 88359906052.406, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 156822284363.6528, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173347208032.45923, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173349890034.86722, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173360600099.71875, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173369456092.77753, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 114946317314.01196, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 169417380573.14777, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 166843406276.167, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167022131326.93918, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167004371179.29108, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167002615777.21475, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167002447844.92758, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 40265289368.52884, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176974601700.9984, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174920867566.5073, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174921819973.986, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174901178365.3262, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174899116163.26465, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174898918506.41833, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 81923366998.92291, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 190410356419.97, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188360311598.58383, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188204026079.0411, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188181829342.89905, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188179608568.28705, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188179387336.04922, tolerance: 523156391.829538
      positive,
    

    0.1
    0.95
    # model coefs = 214
    Train RMSE: 24561.74964572943
    Test RMSE: 39333.83728843832
    (Test-Train)/Train: 60%
    

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:648: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.409e+11, tolerance: 6.087e+08
      coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
# Diabetes dataset

# from sklearn import datasets
# example datasets from sklean: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes
# iris_X, iris_y = datasets.load_iris(return_X_y=True)
# more info on diabetes dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
# diabetes = datasets.load_diabetes(return_X_y=False,as_frame=False)
# print(type(diabetes))
# feat_names=diabetes['feature_names']
# print(feat_names)
# data=diabetes['data']
# target=diabetes['target'] # the target is a quantitative measure of disease progression one year after baseline
# print(data.shape)
# print(target.shape)
# print(diabetes_X.shape) # 442 observations, 10 features
# diabetes_y

# California housing dataset

# from sklearn.datasets import fetch_california_housing
# housing = fetch_california_housing()
# # housing
# feat_names=housing['feature_names']
# print(feat_names)
# print(len(feat_names))
```
