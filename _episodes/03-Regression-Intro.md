---
title: Predictive vs. explanatory regression
teaching: 45
exercises: 2
keypoints:
- "Linear regression models can be used to predict a target variable and/or to reveal relationships between variables"
- "Linear models are most effective when applied to linear relationships. Data transformation techniques can be used to help ensure that only linear relationships are modelled."
- "Train/test splits are used to assess under/overfitting in a model"
- "Different model evaluation metrics provide different perspectives of model error. Some error measurements, such as R-squared, are not as relevant for explanatory models."
objectives:
- "Review structure and goals of linear regression"
- "Know when to use different model evaluation metrics for different modeling goals"
- "Learn how to train and evaluate a predictive machine learning model"
- "Understand how to detect underfitting and overfitting in a machine learning model"
questions:
- "What are the two different goals to keep in mind when fitting machine learning models?"
- "What kinds of questions can be answered using linear regresion?"
- "How can we evaluate a model's ability to capture a true signal/relationship in the data versus spurious noise?"
---

# Linear Regression
Linear regression is powerful technique that is often used to understand whether and how certain *predictor variables* (e.g., garage size, year built, etc.) in a dataset **linearly relate** to some *target variable* (e.g., house sale prices). Starting with linear models when working with high-dimensional data can offer several advantages including:

* **Simplicity and Interpretability**: Linear models, such as linear regression, are relatively simple and interpretable. They provide a clear understanding of how each predictor variable contributes to the outcome, which can be especially valuable in exploratory analysis.

* **Baseline Understanding**: Linear models can serve as a baseline for assessing the predictive power of individual features. This baseline helps you understand which features have a significant impact on the target variable and which ones might be less influential.

* **Feature Selection**: Linear models can help you identify relevant features by looking at the estimated coefficients. Features with large coefficients are likely to have a stronger impact on the outcome, while those with small coefficients might have negligible effects

While linear models have their merits, it's important to recognize that they might not capture complex (nonlinear) relationships present in the data. However, they are often the best option available when working in a high-dimensional context unless data is extremely limited.

##  Goals of Linear Regression
By fitting linear models to the Ames housing dataset, we can...

1. **Predict**: Use predictive modeling to predict hypothetical/future sale prices based on observed values of the predictor variables in our dataset (e.g., garage size, year built, etc.).
2. **Explain**: Use statistics to make scientific claims concerning which predictor variables have a significant impact on sale price — the target variable (a.k.a. response / dependent variable)

**Terminology note:** "target" and "predictor" synonyms
* Predictor = independent variable = feature
* Target = dependent variable = response = outcome

In this workshop, we will explore how we can exploit well-established machine learning methods, including *feature selection*, and *regularization techniques* (more on these terms later), to achieve both of the above goals on high-dimensional datasets.

> ## To predict or explain. That is the question.
> When trying to model data you use in your work, which goal is typically more prevalent? Do you typically care more about (1) accurately predicting some target variable or (2) making scientific claims concerning the existence of certain relationships between variables?
> > ## Solution
> >
> > In a research setting, explaining relationships typically takes higher priority over predicting since explainations hold high value in science, but both goals are sometimes relevant. In industry, the reverse is typically true as many industry applications place predictive accuracy above explainability. We will explore how these goals align and sometimes diverge from one another throughout the remaining lessons.
> {:.solution}
{:.challenge}


## Predicting housing prices with a single predictor
We'll start with the first goal: prediction. How can we use regression models to predict housing sale prices? For clarity, we will begin this question through the lens of simple univariate regression models.

### General procedure for fitting and evaluating predictive models
We'll follow this general procedure to fit and evaluate predictive models:

1. **Extract predictor(s), X, and target, y, variables**
2. **Preprocess the data: check for NaNs and extreme sparsity**
3. **Visualize the relationship between X and y**
4. **Transform target variable, if necessary, to get a linear relationship between predictors**
5. **Train/test split the data**
6. **Fit the model to the training data**
7. **Evaluate model**

    a. Plot the data vs predictions - qualitative assessment

    b. Measure train/test set errors and check for signs of underfitting or overfitting


We'll start by loading in the Ames housing data as we have done previously in this workshop.


```python
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True, parser='auto') #
```

### 1) Extract predictor variable and target variable from dataframe
Next, we'll extract the two variables we'll use for our model — the target variable that we'll attempt to predict (SalePrice), and a single predictor variable that will be used to predict the target variable. For this example, we'll explore how well the "OverallQual" variable (i.e., the predictor variable) can predict sale prices.

**OverallQual**: Rates the overall material and finish of the house

       10	Very Excellent
       1	Very Poor


```python
# Extract x (predictor) and y (target)
y = housing['target']
predictor = 'OverallQual'
x = housing['data'][predictor]
```

### 2) Preprocess the data


```python
# remove columns with nans or containing > 97% constant values (typically 0's)
from preprocessing import remove_bad_cols
x_good = remove_bad_cols(x, .9)
```

    # of columns removed: 0
    Columns removed: []


### 3) Visualize the relationship between x and y
Before fitting any models in a univariate context, we should first explore the data to get a sense for the relationship between the predictor variable, "OverallQual", and the response variable, "SalePrice". If this relationship does not look linear, we won't be able to fit a good linear model (i.e., a model with low average prediction error in a predictive modeling context) to the data.


```python
import matplotlib.pyplot as plt
plt.scatter(x,y, alpha=.1)
plt.xlabel(predictor)
plt.ylabel('Sale Price');
# plt.savefig('..//fig//regression//scatterplot_x_vs_salePrice.png', bbox_inches='tight', dpi=300, facecolor='white');
```







<img src="../fig/regression/scatterplot_x_vs_salePrice.png"  align="center" width="30%" height="30%">

### 4) Transform target variable, if necessary
Unfortunately, sale price appears to grow almost exponentially—not linearly—with the predictor variable. Any line we draw through this data cloud is going to fail in capturing the true trend we see here.

##### Log scaling
How can we remedy this situation? One common approach is to log transform the target variable. We’ll convert the "SalePrice" variable to its logarithmic form by using the math.log() function. Pandas has a special function called apply which can apply an operation to every item in a series by using the statement y.apply(math.log), where y is a pandas series.


```python
import numpy as np
y_log = y.apply(np.log)
```


```python
plt.scatter(x,y_log, alpha=.1)
plt.xlabel(predictor)
plt.ylabel('Sale Price');
# plt.savefig('..//fig//regression//scatterplot_x_vs_logSalePrice.png', bbox_inches='tight', dpi=300, facecolor='white')
```







<img src="../fig/regression/scatterplot_x_vs_logSalePrice.png"  align="center" width="30%" height="30%">

This plot looks much better than the previous one. That is, the trend between OverallQual and log(SalePrice) appears fairly linear. Whether or not it is sufficiently linear can be addressed when we evaluate the model's performance later.

### 5) Train/test split
Next, we will prepare two subsets of our data to be used for *model-fitting* and *model evaluation*. This process is standard for any predictive modeling task that involves a model "learning" from observed data (e.g., fitting a line to the observed data).

During the model-fitting step, we use a subset of the data referred to as **training data** to estimate the model's coefficients (the slope of the model). The univariate model will find a line of best fit through this data.

Next, we can assess the model's ability to generalize to new datasets by measuring its performance on the remaining, unseen data. This subset of data is referred to as the **test data** or holdout set. By evaluating the model on the test set, which was not used during training, we can obtain an unbiased estimate of the model's performance.

If we were to evaluate the model solely on the training data, it could lead to **overfitting**. Overfitting occurs when the model learns the noise and specific patterns of the training data too well, resulting in poor performance on new data. By using a separate test set, we can identify if the model has overfit the training data and assess its ability to generalize to unseen samples. While overfitting is typically not likely to occur when using only a single predictor variable, it is still a good idea to use a train/test split when fitting univariate models. This can help in detecting unanticipated issues with the data, such as missing values, outliers, or other anomalies that affect the model's behavior.

![The above image is from Badillo et al., 2020. An Introduction to Machine Learning. Clinical Pharmacology & Therapeutics. 107. 10.1002/cpt.1796.](../fig/regression/under_v_over_fit.png)


The below code will split our dataset into a training dataset containing 2/3 of the samples, and a test set containing the remaining 1/3 of the data. We'll discuss these different subsets in more detail in just a bit.


```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y_log,
                                                    test_size=0.33,
                                                    random_state=0)

print(x_train.shape)
print(x_test.shape)
```

    (978,)
    (482,)


Reshape single-var predictor matrix in preparation for model-fitting step (requires a 2-D representation)


```python
x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)
print(x_train.shape)
print(x_test.shape)
```

    (978, 1)
    (482, 1)


### 6) Fit the model to the training dataset

During the model fitting step, we use a subset of the data referred to as **training data** to estimate the model's coefficients. The univariate model will find a line of best fit through this data.

##### The sklearn library
When fitting linear models solely for predictive purposes, the scikit-learn or "sklearn" library is typically used. Sklearn offers a broad spectrum of machine learning algorithms beyond linear regression. Having multiple algorithms available in the same library allows you to switch between different models easily and experiment with various techniques without switching libraries. Sklearn is also optimized for performance and efficiency, which is beneficial when working with large datasets. It can efficiently handle large-scale linear regression tasks, and if needed, you can leverage tools like NumPy and SciPy, which are well-integrated with scikit-learn for faster numerical computations.


```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train,y_train)
```

### 7) Evaluate model
#### a) Plot the data vs predictions - qualitative assessment


```python
y_pred_train=reg.predict(x_train)
y_pred_test=reg.predict(x_test)
```


```python
from regression_predict_sklearn import plot_train_test_predictions

?plot_train_test_predictions
```


    [1;31mSignature:[0m
    [0mplot_train_test_predictions[0m[1;33m([0m[1;33m
    [0m    [0mpredictors[0m[1;33m:[0m [0mList[0m[1;33m[[0m[0mstr[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mX_train[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mframe[0m[1;33m.[0m[0mDataFrame[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mX_test[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mframe[0m[1;33m.[0m[0mDataFrame[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0my_train[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0my_test[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0my_pred_train[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0my_pred_test[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mlog_scaled[0m[1;33m:[0m [0mbool[0m[1;33m,[0m[1;33m
    [0m    [0merr_type[0m[1;33m:[0m [0mOptional[0m[1;33m[[0m[0mstr[0m[1;33m][0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mtrain_err[0m[1;33m:[0m [0mOptional[0m[1;33m[[0m[0mfloat[0m[1;33m][0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mtest_err[0m[1;33m:[0m [0mOptional[0m[1;33m[[0m[0mfloat[0m[1;33m][0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m[1;33m)[0m [1;33m->[0m [0mTuple[0m[1;33m[[0m[0mOptional[0m[1;33m[[0m[0mmatplotlib[0m[1;33m.[0m[0mfigure[0m[1;33m.[0m[0mFigure[0m[1;33m][0m[1;33m,[0m [0mOptional[0m[1;33m[[0m[0mmatplotlib[0m[1;33m.[0m[0mfigure[0m[1;33m.[0m[0mFigure[0m[1;33m][0m[1;33m][0m[1;33m[0m[1;33m[0m[0m
    [1;31mDocstring:[0m
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
    [1;31mFile:[0m      c:\users\endemann\documents\github\high-dim-data-lesson\code\regression_predict_sklearn.py
    [1;31mType:[0m      function



```python
(fig1, fig2) = plot_train_test_predictions(predictors=[predictor],
                                           X_train=x_train, X_test=x_test,
                                           y_train=y_train, y_test=y_test,
                                           y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                                           log_scaled=True);

# print(type(fig1))
# import matplotlib.pyplot as plt
# import pylab as pl
# pl.figure(fig1.number)
# plt.savefig('..//fig//regression//univariate_truePrice_vs_predPrice.png',bbox_inches='tight', dpi=300)
# pl.figure(fig2.number)
# plt.savefig('..//fig//regression//univariate_x_vs_predPrice.png',bbox_inches='tight', dpi=300)

```













<img src="../fig/regression/univariate_truePrice_vs_predPrice.png"  align="left" width="40%" height="40%">
<img src="../fig/regression/univariate_x_vs_predPrice.png"  align="center" width="40%" height="40%">

> ## Inspect the plots
> 1. Does the model capture the variability in sale prices well? Would you use this model to predict the sale price of a house? Why or why not?
> 
> 2. Does the model seem to exhibit any signs of overfitting? What about underfitting?
> 
> 3. How might you improve the model?
> 
> > ## Solution
> >
> > 1. Based on visual inspection, this linear model does a fairly good job in capturing the relationship between "OverallQual" and sale price. However, there is a tendency for the model to underpredict more expensive homes and overpredict less expensive homes.
> > 
> > 2. Since the train and test set plots look very similar, overfitting is not a concern. Generally speaking, overfitting is not encountered with univariate models unless you have an incredily small number of samples to train the model on. Since the model follows the trajectory of sale price reasonably well, it also does not appear to underfit the data (at least not to an extreme extent).
> > 
> > 3. In order to improve this model, we can ask ourselves — is "OverallQual" likely the only variable that contributes to final sale price, or should we consider additional predictor variables? Most outcome variables can be influenced by more than one predictor variable. By accounting for all predictors that have an impact on sales price, we can improve the model.
> > 
> {:.solution}
{:.challenge}


#### b. Measure train/test set errors and check for signs of underfitting or overfitting
While qualitative examinations of model performance are extremely helpful, it is always a good idea to pair such evaluations with a quantitative analysis of the model's performance.

**Convert back to original data scale**
There are several error measurements that can't be used to measure a regression model's performance. Before we implement any of them, we'll first convert the log(salePrice) back to original sale price for ease of interpretation.


```python
expY_train = np.exp(y_train)
pred_expY_train = np.exp(y_pred_train)

expY_test = np.exp(y_test)
pred_expY_test = np.exp(y_pred_test)
```

**Measure baseline performance**


```python
from math import sqrt
import pandas as pd

baseline_predict = y.mean()
print('mean sale price =', baseline_predict)
# convert to series same length as y sets for ease of comparison
baseline_predict = pd.Series(baseline_predict)
baseline_predict = baseline_predict.repeat(len(y))
baseline_predict
```

    mean sale price = 180921.19589041095





    0    180921.19589
    0    180921.19589
    0    180921.19589
    0    180921.19589
    0    180921.19589
             ...
    0    180921.19589
    0    180921.19589
    0    180921.19589
    0    180921.19589
    0    180921.19589
    Length: 1460, dtype: float64



**Root Mean Squared Error (RMSE)**:
The RMSE provides an easy-to-interpret number that represents error in terms of the units of the target variable. With our univariate model, the "YearBuilt" predictor variable (a.k.a. model feature) predicts sale prices within +/- $68,106 from the true sale price. We always use the RMSE of the test set to assess the model's ability to generalize on unseen data. An extremely low prediction error in the train set is also a good indicator of overfitting.


```python
from sklearn import metrics

RMSE_baseline = metrics.mean_squared_error(y, baseline_predict, squared=False)
RMSE_train = metrics.mean_squared_error(expY_train, pred_expY_train, squared=False)
RMSE_test = metrics.mean_squared_error(expY_test, pred_expY_test, squared=False)

print(f"Baseline RMSE = {RMSE_baseline}")
print(f"Train RMSE = {RMSE_train}")
print(f"Test RMSE = {RMSE_test}")
```

    Baseline RMSE = 79415.29188606751
    Train RMSE = 45534.34940950763
    Test RMSE = 44762.77229823455


Here, both train and test RMSE are very similar to one another. As expected with most univariate models, we do not see any evidence of overfitting. This model performs substantially better than the baseline. However, an average error of +/- $44,726 is likely too high for this model to be useful in practice. That is, the model is underfitting the data given its poor ability to predict the true housing prices.

**Mean Absolute Percentage Error**:
What if we wanted to know the percent difference between the true sale price and the predicted sale price? For this, we can use the **mean absolute percentage error (MAPE)**...

#### Practice using helper function, `measure_model_err`
This code will be identical to the code above except for changing `metrics.mean_squared_error` to `metrics.mean_absolute_percentage_error`.

Rather than copying and pasting the code above, let's try using one of the helper functions provided for this workshop.


```python
from regression_predict_sklearn import measure_model_err
?measure_model_err
```


    [1;31mSignature:[0m
    [0mmeasure_model_err[0m[1;33m([0m[1;33m
    [0m    [0my[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mbaseline_pred[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mfloat[0m[1;33m,[0m [0mnumpy[0m[1;33m.[0m[0mfloat64[0m[1;33m,[0m [0mnumpy[0m[1;33m.[0m[0mfloat32[0m[1;33m,[0m [0mint[0m[1;33m,[0m [0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0my_train[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0my_pred_train[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0my_test[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0my_pred_test[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mmetric[0m[1;33m:[0m [0mstr[0m[1;33m,[0m[1;33m
    [0m    [0mlog_scaled[0m[1;33m:[0m [0mbool[0m[1;33m,[0m[1;33m
    [0m[1;33m)[0m [1;33m->[0m [0mTuple[0m[1;33m[[0m[0mfloat[0m[1;33m,[0m [0mfloat[0m[1;33m][0m[1;33m[0m[1;33m[0m[0m
    [1;31mDocstring:[0m
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
    [1;31mFile:[0m      c:\users\endemann\documents\github\high-dim-data-lesson\code\regression_predict_sklearn.py
    [1;31mType:[0m      function



```python
MAPE_baseline, MAPE_train, MAPE_test = measure_model_err(y=y, baseline_pred=baseline_predict,
                                                         y_train=expY_train, y_pred_train=pred_expY_train,
                                                         y_test=expY_test, y_pred_test=pred_expY_test,
                                                         metric='MAPE', log_scaled=False)

print(f"Baseline MAPE = {MAPE_baseline*100}")
print(f"Train MAPE = {MAPE_train*100}")
print(f"Test MAPE = {MAPE_test*100}")
```

    Baseline MAPE = 36.3222261212389
    Train MAPE = 18.75854039670096
    Test MAPE = 16.753971728816907


With the MAPE measurement (max value of 1 which corresponds to 100%), we can state that our model over/under estimates sale prices by an average of 23.41% (25.28%) across all houses included in the test set (train set). Certainly seems there is room for improvement based on this measure.

**R-Squared**: Another useful error measurement to use with regression models is the coefficient of determination — $R^2$. Oftentimes pronounced simply "R-squared",  this measure assesses the proportion of the variation in the target variable that is predictable from the predictor variable(s). Using sklearn's metrics, we can calculate this as follows:


```python
R2_baseline, R2_train, R2_test = measure_model_err(y=y, baseline_pred=baseline_predict,
                                                   y_train=expY_train, y_pred_train=pred_expY_train,
                                                   y_test=expY_test, y_pred_test=pred_expY_test,
                                                   metric='R-squared', log_scaled=False)

print(f"Baseline R-squared = {R2_baseline}")
print(f"Train R-squared = {R2_train}")
print(f"Test R-squared = {R2_test}")
```

    Baseline R-squared = 0.0
    Train R-squared = 0.6668752959029058
    Test R-squared = 0.6904634915571379


Our model predicts 70.1% (65.2%) of the variance across sale prices in the test set (train set). The R-squared for the baseline model is 0 because the numerator and denominator in the equation for R-squared are equivalent:

### R-squared equation: R-squared = 1 - (Sum of squared residuals) / (Total sum of squares)

**Sum of Squared Residuals (SSR)**:
SSR = Sum of (Actual Value - Predicted Value)^2 for all data points. The Sum of Squared Residuals (SSR) is equivalent to the variance of the residuals in a regression model. Residuals are the differences between the actual observed values and the predicted values produced by the model. Squaring these differences and summing them up yields the SSR.

**Total Sum of Squares (TSS)**:
TSS = Sum of (Actual Value - Mean of Actual Values)^2 for all data points. The TSS represents the total variability or dispersion in the observed values of the target variable. It measures the total squared differences between each data point's value and the mean of the observed values.

To read more about additional error/loss measurements, visit [sklearn's metrics documentation](https://scikit-learn.org/stable/modules/model_evaluation.html).

> ## More on R-squared
> Our above example model is able to explain roughly 70.1% of the variance in the test dataset. Is this a “good” value for R-squared?
> 
> > ## Solution
> >
> > The answer to this question depends on your objective for the regression model. This relates back to the two modeling goals of *explaining* vs *predicting*. Depending on the objective, the answer to "What is a good value for R-squared?" will be different.
> > 
> > **Predicting the response variable:**
> > If your main objective is to predict the value of the response variable accurately using the predictor variable, then R-squared is important. The value for R-squared can range from 0 to 1. A value of 0 indicates that the response variable cannot be explained by the predictor variable at all. A value of 1 indicates that the response variable can be perfectly explained without error by the predictor variable. In general, the larger the R-squared value, the more precisely the predictor variables are able to predict the value of the response variable. How high an R-squared value needs to be depends on how precise you need to be for your specific model's application. To find out what is considered a “good” R-squared value, you will need to explore what R-squared values are generally accepted in your particular field of study.
> > 
> > **Explaining the relationship between the predictor(s) and the response variable:**
> > If your main objective for your regression model is to explain the relationship(s) between the predictor(s) and the response variable, the R-squared is mostly irrelevant. A predictor variable that consistently relates to a change in the response variable (i.e., has a statistically significant effect) is typically always interesting — regardless of the the effect size. The exception to this rule is if you have a near-zero R-squared, which suggests that the model does not explain any of the variance in the data.
> > 
> {:.solution}
{:.challenge}


## Comparing univariate models
Let's see how well the other predictors in our dataset can predict sale prices. For simplicity, we'll compare just continous predictors for now.

### General procedure for comparing predictive models
We'll follow this general procedure to compare models:

1. Use get_feat_types() to get a list of continuous predictors
2. Create an X variable containing only continuous predictors from `housing['data']`
3. Extract sale prices from `housing['target']` and log scale it
4. Use the remove_bad_cols helper function to remove predictors with nans or containing > 97% constant values (typically 0's)
5. Perform a train/validation/test split using 60% of the data to train, 20% for validation (model selection), and 20% for final testing of the data
6. Use the `compare_models` helper function to quickly calculate train/validation errors for all possible single predictors. Returns a `df_model_err` df that contains the following data stored for each predictor: 'Predictor Variable', 'Train Error', 'Validation Error'.


```python
# preprocess
from preprocessing import get_feat_types
predictor_type_dict = get_feat_types()
continuous_fields = predictor_type_dict['continuous_fields']
X = housing['data'][continuous_fields]
y_log = np.log(housing['target'])

# remove columns with nans or containing > 97% constant values (typically 0's)
from preprocessing import remove_bad_cols
X_good = remove_bad_cols(X, .9)
```

    LotFrontage contains 259 NAs
    MasVnrArea contains 8 NAs
    LowQualFinSF sparsity = 0.9821917808219178
    BsmtHalfBath sparsity = 0.9438356164383561
    KitchenAbvGr sparsity = 0.0006849315068493151
    GarageYrBlt contains 81 NAs
    3SsnPorch sparsity = 0.9835616438356164
    ScreenPorch sparsity = 0.9205479452054794
    PoolArea sparsity = 0.9952054794520548
    # of columns removed: 9
    Columns removed: ['LotFrontage', 'MasVnrArea', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 'GarageYrBlt', '3SsnPorch', 'ScreenPorch', 'PoolArea']



```python
# train/holdout split
X_train, X_holdout, y_train, y_holdout = train_test_split(X_good, y_log,
                                                          test_size=0.4,
                                                          random_state=0)

# validation/test split
X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout,
                                                test_size=0.5,
                                                random_state=0)
```


```python
from regression_predict_sklearn import compare_models
?compare_models
```


    [1;31mSignature:[0m
    [0mcompare_models[0m[1;33m([0m[1;33m
    [0m    [0my[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mbaseline_pred[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mX_train[0m[1;33m:[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mframe[0m[1;33m.[0m[0mDataFrame[0m[1;33m,[0m[1;33m
    [0m    [0my_train[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mX_val[0m[1;33m:[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mframe[0m[1;33m.[0m[0mDataFrame[0m[1;33m,[0m[1;33m
    [0m    [0my_val[0m[1;33m:[0m [0mUnion[0m[1;33m[[0m[0mnumpy[0m[1;33m.[0m[0mndarray[0m[1;33m,[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mseries[0m[1;33m.[0m[0mSeries[0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mpredictors_list[0m[1;33m:[0m [0mList[0m[1;33m[[0m[0mList[0m[1;33m[[0m[0mstr[0m[1;33m][0m[1;33m][0m[1;33m,[0m[1;33m
    [0m    [0mmetric[0m[1;33m:[0m [0mstr[0m[1;33m,[0m[1;33m
    [0m    [0mlog_scaled[0m[1;33m:[0m [0mbool[0m[1;33m,[0m[1;33m
    [0m    [0mmodel_type[0m[1;33m:[0m [0mstr[0m[1;33m,[0m[1;33m
    [0m    [0minclude_plots[0m[1;33m:[0m [0mbool[0m[1;33m,[0m[1;33m
    [0m[1;33m)[0m [1;33m->[0m [0mpandas[0m[1;33m.[0m[0mcore[0m[1;33m.[0m[0mframe[0m[1;33m.[0m[0mDataFrame[0m[1;33m[0m[1;33m[0m[0m
    [1;31mDocstring:[0m
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
    [1;31mFile:[0m      c:\users\endemann\documents\github\high-dim-data-lesson\code\regression_predict_sklearn.py
    [1;31mType:[0m      function



```python
df_model_err = compare_models(y=y, baseline_pred=baseline_predict,
                              X_train=X_train, y_train=y_train,
                              X_val=X_val, y_val=y_val,
                              predictors_list=X_train.columns,
                              metric='RMSE', log_scaled=True,
                              model_type='unregularized', include_plots=False)
```

    # of predictor vars = 1 (LotArea)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 82875.38085495801
    Holdout RMSE = 84323.18923359209
    (Holdout-Train)/Train: 2%

    # of predictor vars = 1 (YearBuilt)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 67679.79091967695
    Holdout RMSE = 69727.34105729726
    (Holdout-Train)/Train: 3%

    # of predictor vars = 1 (YearRemodAdd)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 69055.74101358843
    Holdout RMSE = 70634.28565335085
    (Holdout-Train)/Train: 2%

    # of predictor vars = 1 (OverallQual)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 45516.18554163278
    Holdout RMSE = 46993.501005708364
    (Holdout-Train)/Train: 3%

    # of predictor vars = 1 (OverallCond)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 81016.56620745258
    Holdout RMSE = 84915.45225176154
    (Holdout-Train)/Train: 5%

    # of predictor vars = 1 (BsmtFinSF1)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 73380.6957528966
    Holdout RMSE = 93695.51432329496
    (Holdout-Train)/Train: 28%

    # of predictor vars = 1 (BsmtFinSF2)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 81032.9999001109
    Holdout RMSE = 84932.09816351396
    (Holdout-Train)/Train: 5%

    # of predictor vars = 1 (BsmtUnfSF)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 79102.09029562424
    Holdout RMSE = 82834.52706053828
    (Holdout-Train)/Train: 5%

    # of predictor vars = 1 (TotalBsmtSF)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 63479.544551733954
    Holdout RMSE = 220453.4404000341
    (Holdout-Train)/Train: 247%

    # of predictor vars = 1 (1stFlrSF)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 65085.562454919695
    Holdout RMSE = 105753.38603752904
    (Holdout-Train)/Train: 62%

    # of predictor vars = 1 (2ndFlrSF)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 75823.9442116895
    Holdout RMSE = 82198.0727208069
    (Holdout-Train)/Train: 8%

    # of predictor vars = 1 (GrLivArea)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 60495.94129708608
    Holdout RMSE = 106314.04818601975
    (Holdout-Train)/Train: 76%

    # of predictor vars = 1 (BsmtFullBath)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 78743.02284141965
    Holdout RMSE = 81578.57649938941
    (Holdout-Train)/Train: 4%

    # of predictor vars = 1 (FullBath)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 65268.05249448099
    Holdout RMSE = 71179.80571404072
    (Holdout-Train)/Train: 9%

    # of predictor vars = 1 (HalfBath)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 77596.37795287734
    Holdout RMSE = 80738.02793699679
    (Holdout-Train)/Train: 4%

    # of predictor vars = 1 (BedroomAbvGr)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 79493.20470404223
    Holdout RMSE = 85212.09548505505
    (Holdout-Train)/Train: 7%

    # of predictor vars = 1 (TotRmsAbvGrd)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 67840.8636310711
    Holdout RMSE = 71515.1768065365
    (Holdout-Train)/Train: 5%

    # of predictor vars = 1 (Fireplaces)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 72316.07195521964
    Holdout RMSE = 74450.34818815267
    (Holdout-Train)/Train: 3%

    # of predictor vars = 1 (GarageCars)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 59791.540810726234
    Holdout RMSE = 63397.45129071621
    (Holdout-Train)/Train: 6%

    # of predictor vars = 1 (GarageArea)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 62024.37703005484
    Holdout RMSE = 73482.26232929318
    (Holdout-Train)/Train: 18%

    # of predictor vars = 1 (WoodDeckSF)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 77392.84834191747
    Holdout RMSE = 79652.94391102252
    (Holdout-Train)/Train: 3%

    # of predictor vars = 1 (OpenPorchSF)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 77758.02983921244
    Holdout RMSE = 80447.97275506181
    (Holdout-Train)/Train: 3%

    # of predictor vars = 1 (EnclosedPorch)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 80431.83354350468
    Holdout RMSE = 83927.50566035754
    (Holdout-Train)/Train: 4%

    # of predictor vars = 1 (YrSold)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 80915.74311359474
    Holdout RMSE = 85361.89584710822
    (Holdout-Train)/Train: 5%

    # of predictor vars = 1 (MoSold)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 80954.47906260118
    Holdout RMSE = 84837.77118209275
    (Holdout-Train)/Train: 5%




```python
df_model_err.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictor Variable</th>
      <th>Baseline Error</th>
      <th>Train Error</th>
      <th>Validation Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LotArea</td>
      <td>79415.291886</td>
      <td>82875.380855</td>
      <td>84323.189234</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YearBuilt</td>
      <td>79415.291886</td>
      <td>67679.790920</td>
      <td>69727.341057</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YearRemodAdd</td>
      <td>79415.291886</td>
      <td>69055.741014</td>
      <td>70634.285653</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OverallQual</td>
      <td>79415.291886</td>
      <td>45516.185542</td>
      <td>46993.501006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OverallCond</td>
      <td>79415.291886</td>
      <td>81016.566207</td>
      <td>84915.452252</td>
    </tr>
  </tbody>
</table>
</div>




```python
from regression_predict_sklearn import compare_models_plot
sorted_predictors = compare_models_plot(df_model_err, 'RMSE');
```

    Best model train error = 45516.18554163278
    Best model validation error = 46993.501005708364
    Worst model train error = 63479.544551733954
    Worst model validation error = 220453.4404000341








### Examing the worst performers


```python
df_model_err = compare_models(y=y, baseline_pred=baseline_predict,
                              X_train=X_train, y_train=y_train,
                              X_val=X_val, y_val=y_val,
                              predictors_list=sorted_predictors[-3:],
                              metric='RMSE', log_scaled=True,
                              model_type='unregularized', include_plots=True)
```

    # of predictor vars = 1 (1stFlrSF)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 65085.562454919695
    Holdout RMSE = 105753.38603752904
    (Holdout-Train)/Train: 62%















    # of predictor vars = 1 (GrLivArea)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 60495.94129708608
    Holdout RMSE = 106314.04818601975
    (Holdout-Train)/Train: 76%















    # of predictor vars = 1 (TotalBsmtSF)
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 63479.544551733954
    Holdout RMSE = 220453.4404000341
    (Holdout-Train)/Train: 247%

















It appears the worst performing predictors do not have much of a linear relationship with log(salePrice) and have some extreme outliers in the test set data. We also see some mild indication that some of these predictors may have other external variables that cause them to produce multiple distributions between predictor and log(salePrice). For example, it looks like there may be two data clouds in the TotalBsmtSF vs log(sale_price) plot. The type of basement finish may intersect with that relationship — producing multiple distributions. This is one reason why it is essential to include as many relevant variables in a model as possible.

### Fitting all predictors
Let's assume all predictors in the Ames housing dataset are related to at least some extent to sale price, and fit a multivariate regression model using all continuous predictors.


```python
df_model_err = compare_models(y=y, baseline_pred=baseline_predict,
                              X_train=X_train, y_train=y_train,
                              X_val=X_val, y_val=y_val,
                              predictors_list=[X_train.columns],
                              metric='RMSE', log_scaled=True,
                              model_type='unregularized', include_plots=True)
```

    # of predictor vars = 25
    # of train observations = 876
    # of test observations = 292
    Baseline RMSE = 79415.29188606751
    Train RMSE = 34882.115731053294
    Holdout RMSE = 142964.987928765
    (Holdout-Train)/Train: 310%











### compare permutations of models with different numbers of predictors


```python
import itertools
import random

def generate_combinations(items, K):
    return list(itertools.combinations(items, K))

# Example usage
X_train_columns = list(X_train.columns)
K = 16  # Number of columns in each combination
num_samples = 20

all_combinations = generate_combinations(X_train_columns, K)
sampled_combinations = random.sample(all_combinations, min(num_samples, len(all_combinations)))
sampled_combinations = [list(combo) for combo in sampled_combinations] # convert to list of lists

for combo in sampled_combinations:
    print(combo)

df_model_err = compare_models(X_train=X_train, y_train=y_train,
                   X_test=X_test, y_test=y_test,
                   predictors_list=sampled_combinations,
                   metric='RMSE', log_scaled=True,
                   model_type='unregularized', include_plots=False)

sorted_predictors = compare_models_plot(df_model_err, 'RMSE')
```

    ['LotArea', 'YearRemodAdd', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'WoodDeckSF', 'EnclosedPorch', 'MoSold']
    ['LotArea', 'YearRemodAdd', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'FullBath', 'HalfBath', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'YrSold', 'MoSold']
    ['YearBuilt', 'YearRemodAdd', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'OpenPorchSF', 'EnclosedPorch', 'MoSold']
    ['YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'BsmtFinSF1', 'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'YrSold', 'MoSold']
    ['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'Fireplaces', 'GarageCars', 'GarageArea', 'OpenPorchSF', 'EnclosedPorch', 'YrSold', 'MoSold']
    ['LotArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'YrSold', 'MoSold']
    ['LotArea', 'YearBuilt', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'YrSold', 'MoSold']
    ['YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'WoodDeckSF', 'YrSold', 'MoSold']
    ['YearBuilt', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'EnclosedPorch', 'YrSold', 'MoSold']
    ['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']
    ['LotArea', 'YearBuilt', 'OverallQual', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'BedroomAbvGr', 'WoodDeckSF', 'YrSold', 'MoSold']
    ['YearRemodAdd', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'MoSold']
    ['YearBuilt', 'YearRemodAdd', 'OverallQual', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'EnclosedPorch', 'YrSold', 'MoSold']
    ['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF', 'GrLivArea', 'BsmtFullBath', 'BedroomAbvGr', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'YrSold']
    ['YearBuilt', 'YearRemodAdd', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'EnclosedPorch']
    ['LotArea', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'OpenPorchSF', 'EnclosedPorch', 'YrSold']
    ['YearBuilt', 'OverallQual', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'EnclosedPorch', 'YrSold', 'MoSold']
    ['LotArea', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'BsmtFinSF1', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'YrSold', 'MoSold']
    ['LotArea', 'YearBuilt', 'OverallQual', 'BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'YrSold', 'MoSold']
    ['LotArea', 'YearRemodAdd', 'OverallQual', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'HalfBath', 'TotRmsAbvGrd', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'MoSold']
    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 44501.907407993735
    Test RMSE = 165133.51857708755
    (Test-Train)/Train: 271%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 40080.24411630359
    Test RMSE = 90100.72801158622
    (Test-Train)/Train: 125%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 36353.8202812514
    Test RMSE = 154265.47547974318
    (Test-Train)/Train: 324%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 36696.2163499506
    Test RMSE = 118841.22847233835
    (Test-Train)/Train: 224%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 38474.08752561942
    Test RMSE = 89496.02093486271
    (Test-Train)/Train: 133%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 36358.51062647036
    Test RMSE = 98590.23875605517
    (Test-Train)/Train: 171%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 41404.04288173085
    Test RMSE = 177883.0959053497
    (Test-Train)/Train: 330%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 35503.42533146411
    Test RMSE = 66750.12003363922
    (Test-Train)/Train: 88%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 44441.22224450497
    Test RMSE = 157790.9042683851
    (Test-Train)/Train: 255%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 39099.062002021776
    Test RMSE = 174087.85418686905
    (Test-Train)/Train: 345%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 40959.287529578636
    Test RMSE = 136114.31830108684
    (Test-Train)/Train: 232%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 37150.51755208934
    Test RMSE = 117451.98416959836
    (Test-Train)/Train: 216%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 34249.835099681906
    Test RMSE = 114909.73628432401
    (Test-Train)/Train: 236%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 33555.77772436982
    Test RMSE = 55276.71640821062
    (Test-Train)/Train: 65%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 39565.008323192786
    Test RMSE = 139380.15750391112
    (Test-Train)/Train: 252%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 43038.96121037037
    Test RMSE = 191863.36663309642
    (Test-Train)/Train: 346%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 37676.15611392981
    Test RMSE = 118035.79633539334
    (Test-Train)/Train: 213%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 36166.58397878597
    Test RMSE = 80152.32788819952
    (Test-Train)/Train: 122%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 34233.86878865497
    Test RMSE = 45822.75695507571
    (Test-Train)/Train: 34%

    # of predictor vars = 16
    # of train observations = 973
    # of test observations = 487
    Train RMSE = 36971.77722218436
    Test RMSE = 106457.90933135293
    (Test-Train)/Train: 188%

    18       GarageCars
    13         FullBath
    7         BsmtUnfSF
    17       Fireplaces
    4       OverallCond
    1         YearBuilt
    5        BsmtFinSF1
    19       GarageArea
    12     BsmtFullBath
    11        GrLivArea
    16     TotRmsAbvGrd
    3       OverallQual
    10         2ndFlrSF
    14         HalfBath
    2      YearRemodAdd
    8       TotalBsmtSF
    0           LotArea
    9          1stFlrSF
    6        BsmtFinSF2
    15     BedroomAbvGr
    20       WoodDeckSF
    21      OpenPorchSF
    22    EnclosedPorch
    23           YrSold
    24           MoSold
    Name: Predictor Variable, dtype: object
    25
    34233.86878865497
    45822.75695507571









```python
df_model_err
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictor Variable</th>
      <th>Train Error</th>
      <th>Test Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LotArea</td>
      <td>44501.907408</td>
      <td>165133.518577</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YearBuilt</td>
      <td>40080.244116</td>
      <td>90100.728012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YearRemodAdd</td>
      <td>36353.820281</td>
      <td>154265.475480</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OverallQual</td>
      <td>36696.216350</td>
      <td>118841.228472</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OverallCond</td>
      <td>38474.087526</td>
      <td>89496.020935</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BsmtFinSF1</td>
      <td>36358.510626</td>
      <td>98590.238756</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BsmtFinSF2</td>
      <td>41404.042882</td>
      <td>177883.095905</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BsmtUnfSF</td>
      <td>35503.425331</td>
      <td>66750.120034</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TotalBsmtSF</td>
      <td>44441.222245</td>
      <td>157790.904268</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1stFlrSF</td>
      <td>39099.062002</td>
      <td>174087.854187</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2ndFlrSF</td>
      <td>40959.287530</td>
      <td>136114.318301</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GrLivArea</td>
      <td>37150.517552</td>
      <td>117451.984170</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BsmtFullBath</td>
      <td>34249.835100</td>
      <td>114909.736284</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FullBath</td>
      <td>33555.777724</td>
      <td>55276.716408</td>
    </tr>
    <tr>
      <th>14</th>
      <td>HalfBath</td>
      <td>39565.008323</td>
      <td>139380.157504</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BedroomAbvGr</td>
      <td>43038.961210</td>
      <td>191863.366633</td>
    </tr>
    <tr>
      <th>16</th>
      <td>TotRmsAbvGrd</td>
      <td>37676.156114</td>
      <td>118035.796335</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Fireplaces</td>
      <td>36166.583979</td>
      <td>80152.327888</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GarageCars</td>
      <td>34233.868789</td>
      <td>45822.756955</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GarageArea</td>
      <td>36971.777222</td>
      <td>106457.909331</td>
    </tr>
    <tr>
      <th>20</th>
      <td>WoodDeckSF</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>OpenPorchSF</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>EnclosedPorch</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>YrSold</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>MoSold</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
sorted_predictors = compare_models_plot(df_model_err, 'RMSE')
```

    18       GarageCars
    13         FullBath
    7         BsmtUnfSF
    17       Fireplaces
    4       OverallCond
    1         YearBuilt
    5        BsmtFinSF1
    19       GarageArea
    12     BsmtFullBath
    11        GrLivArea
    16     TotRmsAbvGrd
    3       OverallQual
    10         2ndFlrSF
    14         HalfBath
    2      YearRemodAdd
    8       TotalBsmtSF
    0           LotArea
    9          1stFlrSF
    6        BsmtFinSF2
    15     BedroomAbvGr
    20       WoodDeckSF
    21      OpenPorchSF
    22    EnclosedPorch
    23           YrSold
    24           MoSold
    Name: Predictor Variable, dtype: object
    25
    34233.86878865497
    45822.75695507571








#### 7) Explaining models
At this point, we have assessed the predictive accuracy of our model. However, what if we want to interpret our model to understand which predictor(s) have a consistent or above chance (i.e., statistically significant) impact sales price? For this kind of question and other questions related to model interpretability, we need to first carefully validate our model. The next two episodes will explore some of the necessary checks you must perform before reading too far into your model's estimations.
