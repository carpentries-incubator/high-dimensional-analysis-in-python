---
title: Regularization methods - lasso, ridge, and elastic net
teaching: 45
exercises: 2
keypoints:
- ""
objectives:
- ""
questions:
- "How can LASSO regularization be used as a feature selection method?"
---

## Introduction to the LASSO Model in high-dimensional data analysis
In the realm of high-dimensional data analysis, where the number of predictors begins to approach or exceed the number of observations, traditional regression methods can become challenging to implement and interpret. The Least Absolute Shrinkage and Selection Operator (LASSO) offers a powerful solution to address the complexities of high-dimensional datasets. This technique, introduced by Robert Tibshirani in 1996, has gained immense popularity due to its ability to provide both effective prediction and feature selection.

The LASSO model is a regularization technique designed to combat overfitting by adding a penalty term to the regression equation. The essence of the LASSO lies in its ability to shrink the coefficients of less relevant predictors towards zero, effectively "shrinking" them out of the model. This not only enhances model interpretability by identifying the most important predictors but also reduces the risk of multicollinearity and improves predictive accuracy.

LASSO's impact on high-dimensional data analysis is profound. It provides several benefits:

* Feature Selection / Interpretability: The LASSO identifies and retains the most relevant predictors. With a reduced set of predictors, the model becomes more interpretable, enabling researchers to understand the driving factors behind the predictions.

* Regularization / Dimensionality Reduction: The L1 penalty prevents overfitting by constraining the coefficients, even in cases with a large number of predictors. The L1 penality inherently reduces the dimensionality of the model, making it suitable for settings where the number of predictors is much larger than the sample size.

* Improved Generalization: Related to the above point, LASSO's feature selection capabilities contribute to better generalization and prediction performance on unseen data.

* Data Efficiency: LASSO excels when working with limited samples, offering meaningful insights despite limited observations.

### The L1 penalty
The key concept behind the LASSO is its use of the L1 penalty, which is defined as the sum of the absolute values of the coefficients (parameters) of the model, multiplied by a regularization parameter (usually denoted as λ or alpha).

In the context of linear regression, the L1 penalty can be incorporated into the ordinary least squares (OLS) loss function as follows:

![LASSO Model](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Lasso.png)


Where:

* λ (lambda) is the regularization parameter that controls the strength of the penalty. Higher values of λ lead to stronger regularization and more coefficients being pushed towards zero.
* βi is the coefficient associated with the i-th predictor.

The L1 penalty has a unique property that it promotes sparsity. This means that it encourages some coefficients to be exactly zero, effectively performing feature selection. In contrast to the L2 penalty (Ridge penalty), which squares the coefficients and promotes small but non-zero values, the L1 penalty tends to lead to sparse solutions where only a subset of predictors are chosen. As a result, the LASSO automatically performs feature selection, which is especially advantageous when dealing with high-dimensional datasets where many predictors may have negligible effects on the outcome.

### Fit lasso model and run stats


### Fit lasso model and run stats



```python
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True, parser='auto') #
```


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import statsmodels.api as sm

# Load the Ames housing dataset (replace with your own data loading code)
# ames_data = pd.read_csv('ames_housing.csv')

# Define predictors and target variable
predictors = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'GarageArea', 'GarageCars']
X = ames_data[predictors]
y = ames_data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create LassoCV model with cross-validation for lambda selection
lasso_cv = LassoCV(alphas=np.logspace(-6, 6, 13), cv=5)
lasso_cv.fit(X_train, y_train)

# Calculate p-values for LASSO coefficients
X_train_with_constant = sm.add_constant(X_train)
lasso_model = sm.OLS(y_train, X_train_with_constant)
lasso_results = lasso_model.fit_regularized(alpha=lasso_cv.alpha_, L1_wt=1.0)

# Print the summary of LASSO results
print(lasso_results.summary())

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[2], line 12
          7 # Load the Ames housing dataset (replace with your own data loading code)
          8 # ames_data = pd.read_csv('ames_housing.csv')
          9 
         10 # Define predictors and target variable
         11 predictors = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'GarageArea', 'GarageCars']
    ---> 12 X = ames_data[predictors]
         13 y = ames_data['SalePrice']
         15 # Split the data into training and testing sets
    

    NameError: name 'ames_data' is not defined


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

Define a function `fit_eval_model` that will call both `train_linear_model` and `measure_model_err` and report back on model performance.

## Fit multivariate model using all predictor vars


```python
help(fit_eval_model)
```

## Regularized regression: ridge, lasso, elastic net


### Ridge and RidgeCV
- Show ridge optimization equation
- Default CV is Leave-One-Out. In this form of CV, all samples in the data except for one are used as the inital training set. The left out sample is used a validation set.
- One alpha value used for entire model; larger alphas give more weight to the penalty/regularization term of the loss function

Edit function below to use multiple regression techniques (add model_type input)






```python
# import sklearn's ridge model with built-in cross-validation
from sklearn.linear_model import RidgeCV 

# fit model using multivariate_model_feats and ridge regression
RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, X_test, y_test, labels, 'ridge')
```

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
